import os
import pickle
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr

from tqdm import tqdm

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.drainage_polygons.aux_functions import expand_bounds

def extract(x, x_1):
    #
    land_cover = np.unique(x, return_counts=True)
    #
    df_g = np.sum(np.isin(x_1, class_labels["forest"]) & (~ np.isin(x, class_labels["forest"])))
    df_f = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["farming"])))
    df_p = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["pasture"])))
    df_a = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["agriculture"])))
    df_u = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["urban"])))
    df_m = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["mining"])))
    deforestation = np.array([df_g, df_f, df_p, df_a, df_u, df_m])
    #
    return land_cover, deforestation
 
    
def worker(payload):
    #print(f"*** Processing grid_id: {payload[0]} at {datetime.now().strftime('%H:%M:%S')} ***")

    # load extraction masks
    polygons_raster = pickle.load(open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks/{payload[0]}.pkl", "rb"))

     # extract raw pixel values
    c_extracted = {i: payload[2].values[:,polygons_raster[i].astype(bool).todense()] for i in range(len(polygons_raster))}
    
    out_df = pd.DataFrame()  
    for year in range(1986, 2023):
        # calculate land cover and deforestation data
        c_extracted_processed = [extract(c_extracted[i][year-1985], c_extracted[i][year-1985-1]) for i in range(len(polygons_raster))]
        
        ## post-process land cover data
        # prepare dataframe
        lc_df = pd.DataFrame(columns = legend.ID, index = payload[1])
        # fill dataframe
        for i in range(len(c_extracted_processed)):
            lc_df.loc[payload[1][i], c_extracted_processed[i][0][0]] = c_extracted_processed[i][0][1]
        # fill missing values with 0 and calculate total area 
        lc_df = lc_df.astype(float).fillna(0)
        lc_df["total"] = lc_df.sum(axis=1)

        # sum columns based on IDs and create new columns with names from the dictionary
        for name, id_ in class_labels.items():
            lc_df[name] = lc_df[id_].sum(axis=1)
            
        ## post-process deforestation data
        df_df = pd.DataFrame().from_dict({payload[1][i]: c_extracted_processed[i][1] for i in range(len(c_extracted_processed))}, 
                                        orient = "index",
                                        columns = ["deforestation", "deforestation_f", "deforestation_p", "deforestation_a", "deforestation_u", "deforestation_m"])
        
        # merge dataframes
        merged_df = pd.concat([lc_df, df_df], axis=1).astype(np.uint32)
        # add year and grid_id
        merged_df["year"] = year; merged_df["grid_id"] = grid_id
        # set index
        out_df = pd.concat([out_df, merged_df.reset_index().set_index(["grid_id", "index"])])
    
    out_df.to_feather(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/temp_extracted_land_cover/{payload[0]}.feather")

def main():
    # import drainage polygons
    extracted_drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/extracted_drainage_polygons.feather")
    extracted_drainage_polygons = extracted_drainage_polygons.set_crs(4326)[~extracted_drainage_polygons.is_empty].dropna(subset=["geometry"])
    
    # sort by size for efficient chunking
    sizes = extracted_drainage_polygons.groupby(level=0).size()
    sorted_first_index = sizes.sort_values(ascending=False).index
    extracted_drainage_polygons = extracted_drainage_polygons.loc[sorted_first_index]
    
    # 
    files = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/temp_extracted_land_cover")
    files = pd.Series(files).str.extract(r"(^\d*)").squeeze().astype(int).unique()
    extract_indices = extracted_drainage_polygons.index.get_level_values(0).unique()
    extract_indices = np.array([extract_indices for extract_indices in extract_indices if extract_indices not in files])
        
    # create chunks of size 8
    chunks = np.array_split(extract_indices, np.floor(extract_indices.size / 8))
    
    # iterate over chunks
    for chunk in chunks:
        # prepare data for multiprocessing
        data = [[grid_id, extracted_drainage_polygons.loc[grid_id].index, {}] for grid_id in chunk]
        
        for year in range(1985, 2023):
            # load land cover data    
            with rxr.open_rasterio(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/raw/lc_mapbiomas8_30/mapbiomas_brasil_coverage_{year}.tif", chunks=True).squeeze() as lc_t:       
                for idx, grid_id in enumerate(chunk):
                    # clip land cover data
                    data[idx][2][year] = lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[grid_id].total_bounds)).load()
        
        for idx, grid_id in enumerate(chunk):
            data[idx][2] = xr.concat(data[idx][2].values(), pd.Index(data[idx][2].keys(), name = "year"))
        
        # run multiprocessing
        with Pool(8) as p:
            p.map(worker, data)

if __name__ == "__main__":
    
    # import the legend
    legend = pd.read_excel("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/legend/mapbiomas_legend.xlsx")

    # create class labels to summarize the land cover data
    class_labels = {
        "forest": legend.loc[legend.Class.str.match(r"(^1\.)"), "ID"].values,
        "non-forest/natural": legend.loc[legend.Class.str.match(r"(^2\.)"), "ID"].values,
        "farming": legend.loc[legend.Class.str.match(r"(^3\.)"), "ID"].values,
        "pasture": legend.loc[legend.Class.str.match(r"(^3\.1)"), "ID"].values,
        "agriculture": legend.loc[legend.Class.str.match(r"(^3\.2)"), "ID"].values,
        "non-vegetated": legend.loc[legend.Class.str.match(r"(^4\.)"), "ID"].values,
        "urban": legend.loc[legend.Class.str.match(r"(^4\.2)"), "ID"].values,
        "mining": legend.loc[legend.Class.str.match(r"(^4\.3)"), "ID"].values,
        "water": legend.loc[legend.Class.str.match(r"(^5\.)"), "ID"].values,
    }
    
    main()