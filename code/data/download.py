# Standard library imports
import os
import time
import re
import tarfile
import zipfile
import urllib.request
import json

# Third-party library imports
import geopandas as gpd
import shapely
from tqdm import tqdm
from multiprocessing import Pool
import ee

# Specific imports
from functools import partial
from itertools import compress
import requests

class DataAgent:
    """
    DataAgent class for handling data preparation and fetching.
    """
# -- this function needs to be defined on __main__ so that multiprocessing is able to pickle it
def imagery_downloader(link, root_dir):
    tmp = root_dir + "data/imagery/raw/" + re.compile("(?<=\=).*(?=[\&]requestSignature\=)").search(link).group(0) + ".tar"
    urllib.request.urlretrieve(link, tmp)

class download_agent:
    
    def __init__(self, root_dir, area = "BRA", year = 2010):
        """
        Constructor for DataAgent.

        Parameters:
        root_dir (str): Root directory where data will be stored.
        area (str): Area identifier.
        """
        self.root_dir = root_dir
        self.area = area
        self.year = year
            
    def fetch(self, dataset):
        """
        Fetch data from the provided dataset.

        Parameters:
        dataset (dict): Dataset containing setup details.
        """
        def prepare_filesystem(dataset):
            """
            Prepare filesystem directories for storing data.
            """
            os.makedirs(os.path.dirname(self.root_dir + dataset["path"]), exist_ok = True)

        def dl(dataset):
            """
            Download function to fetch data from a URL.
            """
            if type(dataset["url"]) == dict:
                for file in dataset["url"]["files"]:
                    urllib.request.urlretrieve(dataset["url"]["base"] + file, self.root_dir + dataset["path"] + "/" + file)
            if not type(dataset["url"]) == dict:
                urllib.request.urlretrieve(dataset["url"], self.root_dir + dataset["path"])
            
        def ex(dataset):
            """
            Extract function to extract data from a zip file.
            """
            if type(dataset["url"]) == dict:
                for file in dataset["url"]["files"]:
                    with zipfile.ZipFile(self.root_dir + dataset["path"] + "/" + file, 'r') as zip_ref:
                        zip_ref.extractall(self.root_dir + dataset["path"])
            if not type(dataset["url"]) == dict:
                with zipfile.ZipFile(self.root_dir + dataset["path"], 'r') as zip_ref:
                    zip_ref.extractall((self.root_dir + dataset["path"]).split("raw")[0] + "raw")
                    
        def fe_de_co(dataset):
            """
            Fetch Copernicus DEM data.
            """
            # create shapely polygons on a grid, spanning the entire planet with a resolution of 1 degrees
            grid = gpd.GeoDataFrame(geometry = [shapely.geometry.box(i, j, i+1, j+1) for i in range(-180, 180, 1) for j in range(-90, 90, 1)])
            # filter for the area of interest
            grid["in_bbox"] = grid.intersects(shapely.geometry.box(*gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").total_bounds))
            grid["in_boundaries"] = grid.intersects(gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").geometry.iloc[0])
            grid_filtered = grid.copy()[grid.in_boundaries]
            grid_filtered = grid_filtered[grid_filtered.centroid.map(lambda x: x.coords[0][0]) < -32.5]
            # format the coordinates
            grid_filtered["formatted_lon"] = grid_filtered.bounds.minx.map(lambda x: "W" + str(int(x*-1)).zfill(3) if x < 0 else "E" + str(int(x)).zfill(3))
            grid_filtered["formatted_lat"] = grid_filtered.bounds.miny.map(lambda x: "S" + str(int(x*-1)).zfill(2) if x < 0 else "N" + str(int(x)).zfill(2))
            # get the links
            grid_filtered["filename"] = grid_filtered.apply(lambda x: f"Copernicus_DSM_30_{x.formatted_lat}_00_{x.formatted_lon}_00.tar", axis = 1)
            grid_filtered["dl_link"] = grid_filtered.apply(lambda x: f"https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload/COP-DEM_GLO-90-DGED__2021_1/{x.filename}", axis = 1)
            # download and extract the files
            for idx, row in tqdm(grid_filtered.iterrows(), total = grid_filtered.shape[0]):
                if os.path.exists(self.root_dir + dataset["path"] + "/" + row.filename):
                    continue
                urllib.request.urlretrieve(row.dl_link, self.root_dir + dataset["path"] + "/" + row.filename)
                tarfile.open(self.root_dir + dataset["path"] + "/" + row.filename, "r").extractall(self.root_dir + dataset["path"])
                
        def fe_mb_mo(dataset):
            """
            Fetch Mapbiomas Mosaics to Google cloud.
            
            This function downloads MapBiomas mosaics to Google Cloud using the Earth Engine Python API. 
            It selects specific bands from the mosaics, creates an image, and exports it to Google Cloud Storage
            as GeoTIFF files.

            Note:
            - Ensure that Earth Engine API is initialized and authenticated before using this function.
            - Ensure the necessary permissions and credentials to access Google Cloud Storage are set up.

            """
            
            # Initialize the Earth Engine API
            ee.Initialize(project='master-thesis-414809')

            # Read the grid shapefile
            boundaries = gpd.read_file(self.root_dir + "data/boundaries/gadm_410-BRA.geojson")
            boundaries = ee.Geometry.Rectangle(boundaries.bounds.to_numpy().tolist()[0], proj = "EPSG:4326", evenOdd = False)
            
            # Access the MapBiomas mosaic collection
            mb_mosaics = ee.ImageCollection('projects/nexgenmap/MapBiomas2/LANDSAT/BRAZIL/mosaics-2')
            
            # Define the bands to be selected from the mosaics
            bands = [
                "blue_median",
                "green_median",
                "red_median",
                "nir_median",
                "swir1_median",
                "swir2_median",
            ]
            
            # Filter the mosaic collection by year, select bands, create a mosaic, and convert to int32
            img = mb_mosaics.\
                filterMetadata("year", "equals", self.year).\
                    select(bands).\
                        mosaic().\
                            int32()
                                    

            try:
                # Define the export task
                task = ee.batch.Export.image.toCloudStorage(
                    image=img,
                    description=f'mapbiomas_{self.year}',
                    bucket="master-thesis-lulc",
                    fileNamePrefix=f'mapbiomas/{self.year}_',
                    scale=30,
                    maxPixels=1e13,
                    crs="EPSG:5641",
                    crsTransform=[30, 0, 0, 0, -30, 0],
                    shardSize=208,
                    fileDimensions=[3328, 3328],
                    region=boundaries,
                    fileFormat='GeoTIFF'
                    )
                # Start the export task
                task.start()
                print("*** Started export task ***")
                print("--- Task status ---")
                print(task.status())
                print("--- Task status ---")
            except:
                print("*** Failed to start export task ***")
                
        def dl_ls(dataset, clear_list = True):
            # resolve timeframe
            if (self.year >= 2015):
                query_dataset_name = "landsat_etm_c2_l2"
                query_list_id = self.area + "_ls7_" + str(self.year)
            if (self.year < 2015):
                query_dataset_name = "landsat_tm_c2_l2"
                query_list_id = self.area + "_ls45_" + str(self.year)
                
            # resolve area
            if (self.area == "za"):
                polygon_to_bounds = self.root_dir + "data/boundaries/gadm_za.gpkg"
            elif (self.area == "br"):
                polygon_to_bounds = self.root_dir + "data/boundaries/gadm_410-BRA.geojson"
            
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
            it_lower = 0
            while True:
                time.sleep(1)
                # do the API request
                response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-search",
                                        json = {"datasetName": query_dataset_name,
                                                "maxResults": 1000,
                                                "startingNumber": it_lower,
                                                "sceneFilter": {"acquisitionFilter": {"start": str(self.year) + "-02-01", "end": str(self.year) + "-08-30"},
                                                                "CloudCoverFilter": {"max": 75, "includeUnknown": "true"},
                                                                "spatialFilter": {"filterType": "mbr", 
                                                                                "lowerLeft": {"latitude": boundaries.total_bounds[1], "longitude": boundaries.total_bounds[0]},
                                                                                "upperRight": {"latitude": boundaries.total_bounds[3], "longitude": boundaries.total_bounds[2]}}}},
                                        headers = {"X-Auth-Token": API_key})

                # process results
                res_polygons = gpd.GeoDataFrame({"productId": [x["browse"][0]["id"] for x in response.json()["data"]["results"]],
                                                "entityId": [x["entityId"] for x in response.json()["data"]["results"]],
                                            "geometry": [shapely.Polygon(x["spatialBounds"]["coordinates"][0]) for x in response.json()["data"]["results"]]},
                                            crs = "EPSG:4326")
                # filter for precise boundaries of the area
                scene_list += res_polygons.loc[res_polygons.intersects(boundaries.geometry.iloc[0]), "entityId"].values.tolist()
                
                # set for the next iteration
                if it_lower == 0:
                    print(f"*** Total hits: {response.json()['data']['totalHits']} ***")
                print(f"*** Queried scenes from {it_lower} ***")
                if it_lower + 1000 > response.json()["data"]["totalHits"]:
                    print(f"*** Gathered {len(scene_list)} scenes ***")
                    break
                it_lower = response.json()["data"]["nextRecord"]

            
            # check scene list
            response = requests.post("https://m2m.cr.usgs.gov/api/api/json/stable/scene-list-summary",
                                    json = {"listId": query_list_id,
                                            "datasetName": query_dataset_name},
                                    headers = {"X-Auth-Token": API_key})
            query_list_unempty = (not response.json()["data"]["datasets"] == [])
            if query_list_unempty:
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
                    
        prepare_filesystem(dataset)
        # Perform operations defined in the dataset setup
        for ops in dataset["setup"].split("+"):
            eval(ops)(dataset)

    # Auxiliary data
        
    # TODO: Implement download of DTM
    def download_DTM(self):
        """
        Download Digital Terrain Model (DTM) data.
        """
        # Create DTM directory if not exist
        os.makedirs(os.path.dirname(self.root_dir + "data/DTM/raw/"), exist_ok = True)
        
        # Download DTM data for specified longitude and latitude range
        for lon in range(40, 44 + 1):
            for lat in range(17, 18 + 1):
                # Construct URL and download
                urllib.request.urlretrieve(f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{lon}_{lat}.zip", 
                                           self.root_dir + f"data/DTM/raw/srtm_{lon}_{lat}.zip")
                # Extract downloaded ZIP file
                with zipfile.ZipFile(self.root_dir + f"data/DTM/raw/srtm_{lon}_{lat}.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.root_dir + "data/DTM")