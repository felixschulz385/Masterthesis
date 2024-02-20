import os
#import shutil
import urllib.request
import zipfile

class DataAgent:
    """
    DataAgent class for handling data preparation and fetching.
    """
    
    def __init__(self, root_dir, area):
        """
        Constructor for DataAgent.

        Parameters:
        root_dir (str): Root directory where data will be stored.
        area (str): Area identifier.
        """
        self.root_dir = root_dir
        self.area = area
        self.prepare_filesystem()
            
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

        def download(dataset):
            """
            Download function to fetch data from a URL.
            """
            urllib.request.urlretrieve(dataset["url"], self.root_dir + dataset["path"])
            
        def extract(dataset):
            """
            Extract function to extract data from a zip file.
            """
            with zipfile.ZipFile(self.root_dir + dataset["path"], 'r') as zip_ref:
                zip_ref.extractall(self.root_dir + dataset["path"].split("/raw")[0])
        
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