# Standard library imports
import os
import time
from time import sleep
import re
import requests
import tarfile
import zipfile
import urllib.request
import json
from functools import partial
from itertools import compress

# Third-party library imports
import pandas as pd
import geopandas as gpd
import shapely
from tqdm import tqdm
from multiprocessing import Pool
#import ee
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import cdsapi

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
            boundaries_limits = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").to_crs(5641).buffer(200 * 1e3).to_crs(4326)
            grid["in_bbox"] = grid.intersects(shapely.geometry.box(*boundaries_limits.total_bounds))
            grid["in_boundaries"] = grid.intersects(boundaries_limits.geometry.iloc[0])
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
                try:
                    urllib.request.urlretrieve(row.dl_link, self.root_dir + dataset["path"] + "/" + row.filename)
                    tarfile.open(self.root_dir + dataset["path"] + "/" + row.filename, "r").extractall(self.root_dir + dataset["path"])
                except:
                    pass
                
        def fe_wq_ana(dataset):
            """
            Fetch (scrape) water quality data from the SNIRH website.
            """           
            
            # Check if the queries.json file already exists in the specified path
            if not os.path.exists(f"{self.root_dir}data/water_quality/queries.json"):
                # Load geographic data of stations from a Feather file
                stations_geo = gpd.read_feather(f"{self.root_dir}data/water_quality/stations.feather")
                
                # Filter stations where 'TipoEstacaoQualAgua' equals 1 and select only the 'Codigo' column
                stations_to_query = stations_geo.loc[stations_geo["TipoEstacaoQualAgua"] == 1, ["Codigo"]]
                
                # Initialize a new column 'success' with None values to track query success
                stations_to_query.loc[:, "success"] = None
                
                # Reset the index for a clean DataFrame and set 'Codigo' as the new index
                stations_to_query = stations_to_query.reset_index(drop=True).set_index("Codigo")
                
                # List all files in the raw data directory
                existing_files = os.listdir(f"{self.root_dir}data/water_quality/raw")
                
                # Extract numeric IDs from the file names and filter unique IDs
                existing_file_ids = pd.Series(existing_files).str.extract(r"(^\d*)").squeeze().astype(int).unique()
                
                # Update the 'success' status for stations where corresponding raw data files exist
                stations_to_query.loc[existing_file_ids, "success"] = True

                # Save the DataFrame to a JSON file for later use
                stations_to_query.to_json(f"{self.root_dir}data/water_quality/queries.json")
                
            stations_to_query = pd.read_json(f"{self.root_dir}data/water_quality/queries.json")
            
            downloadPath = f"{self.root_dir}data/water_quality/raw"
            os.makedirs(downloadPath, exist_ok=True)

            # docker run -d -p 4444:4444 -p 7900:7900 -v "/home/ubuntu/ext_drive/scraping/Masterthesis/data/water_quality/raw":"/home/seluser/downloads" --shm-size="2g" selenium/standalone-chrome:latest
            options = webdriver.ChromeOptions()
            options.add_argument('--ignore-ssl-errors=yes')
            options.add_argument('--ignore-certificate-errors')
            #options.add_argument('--headless')
            options.add_argument("--disable-extensions") 
            options.add_argument("--disable-gpu") 

            prefs = {}
            prefs["profile.default_content_settings.popups"]=0
            prefs["download.default_directory"]="/home/seluser/downloads"
            options.add_experimental_option("prefs", prefs)
            # Connect to the WebDriver
            driver = webdriver.Remote(command_executor='http://localhost:4444/wd/hub', options=options)
            
            def download_by_ID(id, driver):
                n_tries = 0
                while n_tries < 5:
                    n_tries += 1
                    
                    try:
                        # clear field
                        driver.find_element(By.XPATH, '//button[@type="reset"]').click()
                        sleep(.5)
                        # enter ID
                        driver.find_element(By.XPATH, '//*[@name="codigoEstacao"]').send_keys(str(id))
                        sleep(.5)
                        # click to search
                        driver.find_element(By.XPATH, '//button[@color="primary"]').click()
                        # wait for results to load
                        element = WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.XPATH, '//td[contains(@class, "mat-column-csv")]/button'))
                        )
                        # download csv
                        element.click()
                        sleep(1)
                        return 1
                    except:
                        sleep(1)
                        if ((n_tries == 2) | (n_tries == 4)):
                            disconnected = True
                            while disconnected:
                                try:
                                    driver.get("https://www.snirh.gov.br/hidroweb/serieshistoricas")
                                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//button[@type="reset"]')))
                                    disconnected = False
                                except:
                                    pass
                return 0
            
            driver.get("https://www.snirh.gov.br/hidroweb/serieshistoricas")
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//button[@type="reset"]')))

            indices_to_fetch = stations_to_query[stations_to_query.success.isna()].index
            for idx in tqdm(range(len(indices_to_fetch))):
                stations_to_query.loc[indices_to_fetch[idx], "success"] = download_by_ID(indices_to_fetch[idx], driver)
                if idx % 10 == 0:
                    stations_to_query.to_json(f"{self.root_dir}data/water_quality/queries.json")
            
            driver.close()
        
        def fe_cc_co(dataset):
            """
            Fetch Copernicus Cloud Cover data
            
            Note: Requires a Copernicus Climate Data Store account and API key installed on the device.
            """
            
            # Create a CDS API client
            c = cdsapi.Client()
            # Retrieve the data
            c.retrieve(
                'satellite-cloud-properties',
                {
                    'format': 'zip',
                    'product_family': 'clara_a3',
                    'origin': 'eumetsat',
                    'variable': 'cloud_fraction',
                    'climate_data_record_type': 'thematic_climate_data_record',
                    'time_aggregation': 'monthly_mean',
                    'year': [
                        '1979', '1980', '1981',
                        '1982', '1983', '1984',
                        '1985', '1986', '1987',
                        '1988', '1989', '1990',
                        '1991', '1992', '1993',
                        '1994', '1995', '1996',
                        '1997', '1998', '1999',
                        '2000', '2001', '2002',
                        '2003', '2004', '2005',
                        '2006', '2007', '2008',
                        '2009', '2010', '2011',
                        '2012', '2013', '2014',
                        '2015', '2016', '2017',
                        '2018', '2019', '2020',
                    ],
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                },
                self.root_dir + dataset["path"])
            
            # extract the files
            with zipfile.ZipFile(self.root_dir + dataset["path"], 'r') as zip_ref:
                    zip_ref.extractall((self.root_dir + dataset["path"]).split("raw")[0] + "raw")
          
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
                    
if __name__ == "__main__":
    test = download_agent("/home/ubuntu/ext_drive/scraping/Masterthesis/")
    test.fetch({"setup": "fe_wq_ana", "path": "data/water_quality"})