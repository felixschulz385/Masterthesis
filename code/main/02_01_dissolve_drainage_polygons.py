import pickle
import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.river_network import river_network, calculate_distance_from_estuary
from data.preprocess.drainage_polygons.extract_detailed_drainage_polygons import extract_polygons_river, expand_bounds

# connect to database
conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
# get grid data
grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
# get grid geometries
grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
# merge data
grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on = "CellID"))
# close connection
conn.close()

# Import the fixed drainage polygons
drainage_polygons = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/raw/drainage_polygons.geojson", engine="pyogrio")
drainage_polygons_projected = drainage_polygons.to_crs(5641)
drainage_polygons_gridded = grid_data.sjoin(gpd.GeoDataFrame(geometry = drainage_polygons_projected.centroid, index = drainage_polygons_projected.index), how = "right").dropna(subset = ["index_left"])

# Import the raw update set
with open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/temp_update_set.pkl", "rb") as f:
    update_set = pickle.load(f)
    
# there was a bug in the script for the 10hr run of the fix
# if there was no polygon to fix in a given grid cell, the script returned None
# this bug is now fixed for future re-runs; I impute these values here
for idx in [idx for idx, val in update_set.items() if val is None]:
    update_set[idx] = drainage_polygons.loc[drainage_polygons_gridded.index[drainage_polygons_gridded.index_left == idx].values].to_crs(5641)

# Combine update set
drainage_polygons = pd.concat(update_set).reset_index(drop = True)

# Read the river network
rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")

# Join the polygons with the river network
joined = drainage_polygons.sjoin(rivers_brazil_shapefile, how="left", predicate = "intersects")
# Remove polygons with no corresponding river
joined.dropna(subset = ["index_right0", "index_right1", "index_right2"], inplace = True)
# Create a tuple of river indices
joined["index_right"] = joined.apply(lambda x: tuple([int(x.index_right0), int(x.index_right1), int(x.index_right2)]), axis = 1)
# Calculate the intersection length
joined["intersection_length"] = joined.apply(lambda x: x.geometry.intersection(rivers_brazil_shapefile.loc[x.index_right].geometry).length, axis = 1)
# Assign drainage polygons to rivers with longest intersection
tmp = joined.reset_index().groupby(["index", "estuary", "river"]).intersection_length.sum().groupby("index").idxmax()
joined_assignment = pd.DataFrame({"estuary": tmp.apply(lambda x: x[1]),
                                  "river": tmp.apply(lambda x: x[2])},
                                 index = tmp.apply(lambda x: x[0]))


# Join in the assignment and dissolve by river
drainage_polygons = drainage_polygons.join(joined_assignment).to_crs(4326)
drainage_polygons["geometry"] = drainage_polygons.geometry.buffer(0)
drainage_polygons_dissolved = drainage_polygons.dissolve(["estuary", "river"]).reset_index()

# Filter out those polygons for which their centroid does not intersect the grid data
drainage_polygons_dissolved_filtered = drainage_polygons_dissolved[drainage_polygons_dissolved.centroid.intersects(grid_data.to_crs(4326).unary_union)].copy()

# Clip the polygons to the expanded boundaries of Brazil
boundaries_limits = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").to_crs(5641).buffer(200 * 1e3).to_crs(4326)
drainage_polygons_dissolved_filtered = gpd.clip(drainage_polygons_dissolved_filtered, boundaries_limits)

# Save the dissolved and filtered polygons
drainage_polygons_dissolved.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved.feather")
drainage_polygons_dissolved_filtered.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved_filtered.feather")