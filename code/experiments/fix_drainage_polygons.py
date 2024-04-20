# Standard library imports
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from itertools import chain

# Third-party imports for data handling and computation
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm

# Geospatial and image processing libraries
import cv2
import shapely
from pysheds.grid import Grid
from rasterio.io import MemoryFile

# Multiprocessing imports
from multiprocessing import Pool

# Local module imports for specific functionality
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.river_network import river_network, calculate_distance_from_estuary
from data.preprocess.drainage_areas import fix_rivers_in_grid

# Database connection to fetch grid data
conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
conn.close()

# Reading and processing river data
rivers = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/msc_rivers.feather").to_crs(5641)
rivers_subset = gpd.clip(rivers, grid_data.geometry.iloc[525])

# Loading administrative boundaries and clipping to the grid area
boundaries = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio").to_crs(5641)
boundaries_subset = gpd.clip(boundaries, grid_data.geometry.iloc[525])

# Loading river network data and computing distances from the estuary
rivers_brazil = pickle.load(open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/network.pkl", "rb"))
rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

# Preparing digital elevation model data
filedir_dem_cop = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/DEM_GLO-90/"
files_dem_cop = pd.Series(os.listdir(filedir_dem_cop))
files_dem_cop = files_dem_cop[~files_dem_cop.str.contains(r"\.tar$")]
files_dem_cop = files_dem_cop.map(lambda x: f"{filedir_dem_cop}{x}/DEM/{x}_DEM.tif")
mfdataset = [rxr.open_rasterio(file, chunks=True)[0,:-1,:-1] for file in files_dem_cop]
height_profile = xr.combine_by_coords(mfdataset)

# Reading and processing drainage areas
drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage_area.feather")
drainage_polygons_projected = drainage_polygons.to_crs(5641)
drainage_polygons_gridded = grid_data.sjoin(gpd.GeoDataFrame(geometry=drainage_polygons_projected.centroid, index=drainage_polygons_projected.index), how="right").dropna(subset=["index_left"])

# Checking for previously saved work and updating the grid calculations
if os.path.exists(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/temp_update_set.pkl"):
    with open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/temp_update_set.pkl", "rb") as f:
        update_set = pickle.load(f)    
else:
    update_set = {key: None for key in drainage_polygons_gridded.index_left.unique()}
    
def wrapper(i):
    try:
        return fix_rivers_in_grid(i, rivers_brazil_shapefile, rivers_brazil_topology, drainage_polygons, drainage_polygons_gridded, height_profile)
    except:
        return None

for idx, i in tqdm(enumerate([key for key, value in update_set.items() if value is None])):
    update_set[i] = wrapper(i)
    if idx % 10 == 0:  # Assuming the intent was to save periodically based on iterations, not the index 'i'
        with open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/temp_update_set.pkl", "wb") as f:
            pickle.dump(update_set, f)
