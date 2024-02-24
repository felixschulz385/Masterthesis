import os
import pandas as pd
import geopandas as gpd

def preprocess_boundaries(root_dir, area):
    if os.path.exists('data/boundaries/raw/gadm_410-levels.gpkg'):
        # Load the boundaries data
        boundaries = gpd.read_file(root_dir + 'data/boundaries/raw/gadm_410-levels.gpkg')

        # Filter for country
        boundaries = boundaries[boundaries.GID_0 == area]

        # Save the preprocessed data
        boundaries.to_file(root_dir + 'data/boundaries/gadm_410-' + area + '.geojson', driver='GeoJSON')