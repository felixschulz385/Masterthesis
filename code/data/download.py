import os
import time
import re
import tarfile
from functools import partial
from itertools import compress
import urllib.request
import zipfile
import requests
import geopandas as gpd
import shapely
from tqdm import tqdm
from multiprocessing import Pool

# -- this function needs to be defined on __main__ so that multiprocessing is able to pickle it
def imagery_downloader(link, root_dir):
    tmp = root_dir + "data/imagery/raw/" + re.compile("(?<=\=).*(?=[\&]requestSignature\=)").search(link).group(0) + ".tar"
    urllib.request.urlretrieve(link, tmp)

class download:
    
    def __init__(self, root_dir, area):
        self.root_dir = root_dir
        self.area = area
    
    def prepare_filesystem(self):
        for directory in ("data", 
                          "data/biome_clusters/", "data/boundaries/", "data/DTM/", "data/imagery/raw/", "data/land_cover/"):
            os.makedirs(os.path.dirname(self.root_dir + directory), exist_ok = True)
    
    ###
    # Imagery data
    
    def download_imagery(self, year, clear_list = True):
        # resolve timeframe
        if (year >= 2015):
            query_dataset_name = "landsat_etm_c2_l2"
            query_list_id = self.area + "_ls7_" + str(year)
        if (year < 2015):
            query_dataset_name = "landsat_tm_c2_l2"
            query_list_id = self.area + "_ls45_" + str(year)
            
        # resolve area
        if (self.area == "za"):
            polygon_to_bounds = self.root_dir + "data/boundaries/gadm_za.gpkg"
        
        # get boundaries
        print("--- Loading Boundaries ---")
        boundaries = gpd.read_file(polygon_to_bounds)
        
        # get credentials
        with open("code/data/api_creds", "r") as file:
            exec("self.creds = " + file.read())
        # login
        print("--- Login ---")
        response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/login",
                                json = self.creds)
        API_key = response.json()["data"]
        
        # get all scenes within boundaries
        print("--- Querying scenes ---")
        scene_list = []
        it_starting_number = 0
        it_next_record = 1001
        it_total_hits = 0
        while (it_total_hits != it_next_record):
            print(f"*** Querying scenes {it_starting_number} to {it_next_record - 1} ***")
            time.sleep(1)
            # do the API request
            response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-search",
                                    json = {"datasetName": query_dataset_name,
                                            "maxResults": 1000,
                                            "startingNumber": it_starting_number,
                                            "sceneFilter": {"acquisitionFilter": {"start": "2015-01-01", "end": "2015-12-31"},
                                                            "spatialFilter": {"filterType": "mbr", 
                                                                            "lowerLeft": {"latitude": boundaries.total_bounds[1], "longitude": boundaries.total_bounds[0]},
                                                                            "upperRight": {"latitude": boundaries.total_bounds[3], "longitude": boundaries.total_bounds[2]}}}},
                                    headers = {"X-Auth-Token": API_key})
            # set for the next iteration
            if it_starting_number == 0:
                it_total_hits = response.json()["data"]["totalHits"]
            it_starting_number = it_next_record
            it_next_record = response.json()["data"]["nextRecord"]
            # process results
            res_polygons = gpd.GeoDataFrame({"productId": [x["browse"][0]["id"] for x in response.json()["data"]["results"]],
                                            "entityId": [x["entityId"] for x in response.json()["data"]["results"]],
                                        "geometry": [shapely.Polygon(x["spatialBounds"]["coordinates"][0]) for x in response.json()["data"]["results"]]},
                                        crs = "EPSG:4326")
            #
            scene_list += res_polygons.loc[res_polygons.intersects(boundaries.geometry.iloc[0]), "entityId"].values.tolist()
        
        # check scene list
        response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-list-summary",
                                json = {"listId": query_list_id,
                                        "datasetName": query_dataset_name},
                                headers = {"X-Auth-Token": API_key})
        query_list_unempty = (response.json()["data"]["datasets"][0]["sceneCount"] > 0)
        
        # clear scene list
        if (query_list_unempty & clear_list):
            print("--- Removing existing list ---")
            response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-list-remove",
                        json = {"listId": query_list_id},
                        headers = {"X-Auth-Token": API_key})
        
        # add to scene list if cleared or empty
        if ((query_list_unempty & clear_list) | ~query_list_unempty):
            print("--- Adding scenes to list ---")
            response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-list-add",
                                    json = {"listId": query_list_id,
                                            "datasetName": query_dataset_name,
                                            "entityIds": scene_list},
                                    headers = {"X-Auth-Token": API_key})
        
        # get productIds for download
        print("--- Querying download products ---")
        response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/download-options",
                                json = {"listId": query_list_id,
                                        "datasetName": query_dataset_name},
                                headers = {"X-Auth-Token": API_key}) 
        downloads = [{"label": query_list_id + str(idx), 
                      "productId": x["id"], 
                      "entityId": x["entityId"]} for idx, x in enumerate(response.json()["data"]) if x["available"]]

        
        # get download links
        print("--- Requesting download links ---")
        response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/download-request",
                                json = {"downloads": downloads},
                                headers = {"X-Auth-Token": API_key})
        downloads_links = [x["url"] for x in response.json()["data"]["availableDownloads"]]
        # filter list for files not already downloaded
        downloads_links = list(compress(downloads_links, [not os.path.exists(self.root_dir + "data/imagery/raw/" + re.compile("(?<=\=).*(?=[\&]requestSignature\=)").search(link).group(0) + ".tar") for link in downloads_links]))
        
        # download files
        print("--- Downloading files ---") 
        with Pool(4) as p:
            list(tqdm(p.map(partial(imagery_downloader, root_dir = self.root_dir), downloads_links, chunksize = 100), total = len(downloads_links)))

        # logout
        print("--- Logging out ---")
        response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/logout",
                                headers = {"X-Auth-Token": API_key})
        
        # unpack
        print("--- Unpacking ---")
        for filename in [re.compile("(?<=\=).*(?=[\&]requestSignature\=)").search(link).group(0) for link in downloads_links]:
            with tarfile.open(self.root_dir + "data/imagery/raw/" + filename + ".tar", "r") as file:
                outdir = self.root_dir + "data/imagery/" + query_list_id
                os.makedirs(outdir, exist_ok = True)
                file.extractall(outdir)
        
        
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
            

dl_agent = download("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/Masterthesis/", "za")
dl_agent.download_imagery(2015, clear_list = False)