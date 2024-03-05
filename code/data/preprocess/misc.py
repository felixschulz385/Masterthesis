import os
import pandas as pd
import geopandas as gpd

def preprocess_boundaries(root_dir, area):
    if os.path.exists('data/misc/raw/gadm_410-levels.gpkg'):
        # Load the boundaries data
        boundaries = gpd.read_file(root_dir + 'data/boundaries/raw/gadm_410-levels.gpkg')

        # Filter for country
        boundaries = boundaries[boundaries.GID_0 == area]

        # Save the preprocessed data
        boundaries.to_feather(root_dir + 'data/boundaries/gadm_410-' + area + '.feather')
        
def preprocess_rivers(root_dir, area):
    if os.path.exists('data/misc/raw/GEOFT_BHO_REF_RIO.shp'):
        # Load the boundaries data
        rivers = gpd.read_file(root_dir + 'data/misc/raw/GEOFT_BHO_REF_RIO.shp', engine = "pyogrio")

        # Save the preprocessed data
        rivers.to_feather(root_dir + 'data/misc/msc_rivers.feather')