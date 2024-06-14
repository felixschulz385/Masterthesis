import numpy as np
import pandas as pd
import dask.dataframe as dd
import geopandas as gpd
from itertools import chain

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.river_network import calculate_distance_from_estuary

def preprocess_land_cover_stations(root_dir = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/"):
    """
    Preprocess land cover data to match it with water quality stations.

    Parameters:
    root_dir (str): The root directory where the data files are located. Default is 
                    "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/".

    Returns:
    None

    Outputs:
    - A parquet file containing processed land cover data for water quality stations, 
      saved to the specified root directory.
    """
    # import data
    rivers = gpd.read_feather(f"{root_dir}data/river_network/shapefile.feather")
    topology = gpd.read_feather(f"{root_dir}data/river_network/topology.feather")
    stations_rivers = pd.read_pickle(f"{root_dir}data/water_quality/stations_rivers.pkl")
    water_quality_panel = pd.read_parquet(f"{root_dir}data/water_quality/quality_indicators_panel.parquet")
    drainage_polygons = gpd.read_feather(f"{root_dir}data/drainage/extracted_drainage_polygons.feather")
    land_cover = dd.read_parquet(f"{root_dir}data/land_cover/temp_extracted_land_cover/", columns=["year", "deforestation", "deforestation_f", "deforestation_p", "deforestation_u", "deforestation_m", "forest", "farming", "pasture", "urban", "mining", "total"]).astype(np.uint32)

    # calculate distance from estuary
    rivers = calculate_distance_from_estuary(rivers, topology)
    del topology

    # compute lookup from node to list of polygons
    drainage_polygons_tmp = drainage_polygons[((~ drainage_polygons.is_empty) & (~ drainage_polygons.geometry.isna()))].drop(columns="geometry").reset_index(names = ["grid", "index"])
    drainage_polygons_tmp = pd.merge(drainage_polygons_tmp, rivers.drop(columns = ["NORIOCOMP", "CORIO", "geometry"]), on = ["estuary", "river", "segment", "subsegment"])
    drainage_polygons_tmp["identifier"] = drainage_polygons_tmp.apply(lambda x: [x["grid"], x["index"]], axis = 1)
    node_polygon_lookup = drainage_polygons_tmp.set_index("upstream_node_id").groupby(level=0).apply(lambda x: x["identifier"].tolist()).to_dict()
    del drainage_polygons, drainage_polygons_tmp
    
    # compute lookup from station to list of upstream nodes
    reachability_lookup = stations_rivers.set_index("Codigo", drop = True).reachability.apply(lambda x: list(x) if x else None)
    reachability_lookup = reachability_lookup[~reachability_lookup.index.duplicated(keep='first')]
    
    # compute lookup from node to distance from estuary
    distance_lookup = rivers[["upstream_node_id", "distance_from_estuary"]].rename(columns={"upstream_node_id": "node"}).reset_index(drop = True)
    del rivers
    
    # compute lookup from station to distance from estuary
    station_distance_lookup = stations_rivers.set_index("Codigo", drop=True).distance_from_estuary
    station_distance_lookup = station_distance_lookup[~station_distance_lookup.index.duplicated(keep='first')]
    del stations_rivers
    
    # get all station codes
    station_codes = water_quality_panel.station.unique()
    del water_quality_panel
    
    # get upstream nodes for all stations
    t_node_ids = {x: reachability_lookup.loc[x] for x in station_codes}
    del reachability_lookup, station_codes
    
    # prepare deforestation data in dask dataframe
    t_deforestation = land_cover.groupby(["grid_id", "index", "year"]).sum().persist()
    del land_cover
    
    ## chunk all stations into chunks if 1M nodes
    # assign chunks of 1M nodes
    t_chunks = np.cumsum([len(x) if x is not None else 0 for i, x in t_node_ids.items()]) // 1e6
    # get indices of chunks
    t_chunks = [(int(np.argmax(t_chunks == i)), int(len(t_chunks) - np.argmax(t_chunks[::-1] == i) - 1)) for i in np.unique(t_chunks)]
    # get nodes split into chunks
    c_node_ids = [{y: t_node_ids[y] for y in list(t_node_ids.keys())[x[0]:x[1]]} for x in t_chunks]
    #c_polygon_ids = [{station_id: {node_id: node_polygon_lookup.get(node_id, [None, None]) for node_id in node_ids if node_id is not None} for station_id, node_ids in chunk.items() if node_ids is not None} for chunk in c_node_ids]
    del t_node_ids
    
    # iterate over chunks
    out_df = [None] * len(c_node_ids)
    for i in range(len(c_node_ids)):
        ## prepare data frame for final estimation
        # get polygon ids
        t_index_prep = {(key, int(value)): node_polygon_lookup.get(value, [None, None]) for key, values in c_node_ids[i].items() if values is not None for value in values}
        # combine in tuple for index
        t_index_prep = [(key[0], key[1], int(value[0]), int(value[1])) for key, values in t_index_prep.items() for value in values if value is not None]         
        # create dataframe with indices
        c_final_df = dd.from_pandas(pd.DataFrame().from_records(t_index_prep, columns = ["station", "node", "grid_id", "index"]))
        # merge with deforestation data
        c_final_df = dd.merge(c_final_df, t_deforestation.reset_index(), on = ["grid_id", "index"], how = "left")
        # merge with distance data
        c_final_df = dd.merge(c_final_df, dd.from_pandas(distance_lookup.dropna().astype({"node": int})), on="node")
        # calculate distance from station
        c_final_df["distance_from_station"] = (c_final_df.distance_from_estuary - c_final_df.station.map(station_distance_lookup, meta=('station', 'uint32'))).astype(np.uint32)
        
        # Define the windows
        max_ = 200 * 1e3; step_ = 20 * 1e3
        windows = np.arange(0, max_, step_)
        # Create bin labels
        bin_labels = [f"[{start},{start + step_})" for start in windows]
        # Bin the sorting column
        c_final_df['bins'] = c_final_df["distance_from_station"].map_partitions(pd.cut, bins=np.concatenate([windows, [max_ + step_]]), labels=bin_labels, right=False, include_lowest=True)
        
        # Group by the bins and sum the value column
        out_df[i] = c_final_df.drop(columns=["distance_from_station"]).groupby(["station", "year", "bins"]).sum().compute()
        out_df[i] = out_df[i].groupby(["station", "year", "bins"]).sum()
    
    # combine all chunks
    out_df = pd.concat(out_df)
    
    # save to parquet
    out_df.reset_index().astype({"year": np.int16}).to_parquet(f"{root_dir}data/land_cover/land_cover_stations.parquet")
            
if __name__ == "__main__":
    preprocess_land_cover_stations()