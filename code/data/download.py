import os
import shutil
import urllib.request
import zipfile


class download:
    
    def __init__(self, root_dir, area):
        self.root_dir = root_dir
        self.area = area
    
    def prepare_filesystem(self):
        for directory in ("data", 
                          "data/biome_clusters/", "data/boundaries/", "data/DTM/", "data/imagery/", "data/land_cover/"):
            os.makedirs(os.path.dirname(self.root_dir + directory), exist_ok = True)
    
    ###
    # Imagery data
    
    ###
    # Land Cover data
    
    def download_land_cover(self):
        #
        urllib.request.urlretrieve("https://zenodo.org/record/3939038/files/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif?download=1", 
                                   self.root_dir + "data/land_cover/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif")
        
    
    ###
    # Auxiliary data
        
    def download_boundaries(self):
        #
        os.makedirs(os.path.dirname(self.root_dir + "data/boundaries/raw/"), exist_ok = True)
        #
        urllib.request.urlretrieve("https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip", 
                                   self.root_dir + "data/boundaries/raw/gadm_410-levels.zip")
        #
        with zipfile.ZipFile(self.root_dir + "data/boundaries/raw/gadm_410-levels.zip", 'r') as zip_ref:
            zip_ref.extractall(self.root_dir + "data/boundaries")
            
    def download_biome_clusters(self):
        #
        os.makedirs(os.path.dirname(self.root_dir + "data/biome_clusters/raw/"), exist_ok = True)
        #
        urllib.request.urlretrieve("https://zenodo.org/record/5848610/files/biome_cluster_shapefile.zip?download=1", 
                                   self.root_dir + "data/biome_clusters/raw/biome_cluster_shapefile.zip")
        #
        with zipfile.ZipFile(self.root_dir + "data/biome_clusters/raw/biome_cluster_shapefile.zip", 'r') as zip_ref:
            zip_ref.extractall(self.root_dir + "data/biome_clusters")
            
    def download_DTM(self):
        #
        os.makedirs(os.path.dirname(self.root_dir + "data/DTM/raw/"), exist_ok = True)
        #
        for lon in range(40, 44 + 1):
            for lat in range(17, 18 + 1):
                #
                urllib.request.urlretrieve(f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{lon}_{lat}.zip", 
                                           self.root_dir + f"data/DTM/raw/srtm_{lon}_{lat}.zip")
                #
                with zipfile.ZipFile(self.root_dir + f"data/DTM/raw/srtm_{lon}_{lat}.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir + "data/DTM")
            

dl_agent = download("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/Masterthesis/", "africa")
dl_agent.download_boundaries()