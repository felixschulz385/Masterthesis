import os
import geopandas as gpd

def preprocess_landcover(root_dir, area):
    
    # check if IBGE data is available
    if os.path.exists(root_dir + 'data/land_cover/raw/Brasil_2010.shp'):
        # Load the boundaries data
        lc = gpd.read_file(root_dir + 'data/land_cover/raw/Brasil_2010.shp',
                           engine="pyogrio")

        # Reproject and export to feather
        lc.to_crs("EPSG:5641").to_feather(root_dir + 'data/land_cover/raw/lc_ibge_2010.feather')
        
