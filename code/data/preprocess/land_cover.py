import os
import geopandas as gpd

def preprocess_landcover(root_dir, area):
    
    # check if IBGE data is available
    if os.path.exists(root_dir + 'data/land_cover/raw/Cobertura_uso_terra_Brasil_serie_revisada.shp'):
        # Load the boundaries data
        lc = gpd.read_file(root_dir + 'data/land_cover/raw/Cobertura_uso_terra_Brasil_serie_revisada.shp',
                           engine="pyogrio")

        # Reproject and export to feather
        lc.to_crs("EPSG:5641").to_feather(root_dir + 'data/land_cover/raw/Cobertura_uso_terra_Brasil_serie_revisada.feather')
        
