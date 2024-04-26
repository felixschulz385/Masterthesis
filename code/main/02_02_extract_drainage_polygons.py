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
from data.preprocess.drainage_polygons.extract_detailed_drainage_polygons import extract_polygons_river, expand_bounds

# A function to calculate a drainage area
def worker(chunk, drainage_polygons_dissolved, rivers_brazil_shapefile):
    results = [None] * len(chunk)
    for i in chunk:
        try: 
            results[i % len(chunk)] = extract_polygons_river(drainage_polygons_dissolved.iloc[i], rivers_brazil_shapefile)
        except:
            pass
    pickle.dump(results, open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_processed_chunks/{chunk[0]}_{chunk[-1]}.pkl", "wb"))

def process_chunk(chunk):
    worker(chunk, drainage_polygons_dissolved, rivers_brazil_shapefile)
    
if __name__ == '__main__':
    # Database connection to fetch grid data
    conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
    grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
    grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
    grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
    conn.close()

    # Loading river network data and computing distances from the estuary
    rivers_brazil = pickle.load(open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/network.pkl", "rb"))
    rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
    rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
    rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

    # Get the dissolved drainage polygons
    drainage_polygons_dissolved = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved_filtered.feather")
    
    # Create a iterable of chunks
    chunksize = 100
    chunks = [list(range(i, min(i + chunksize, len(drainage_polygons_dissolved)))) for i in range(0, len(drainage_polygons_dissolved), chunksize)]

    with Pool(processes = int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
        # Apply the function to each chunk using multiprocessing
        pool.map(process_chunk, chunks)