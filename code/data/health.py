import time
import io
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class health:
    """
    A class to preprocess health data.
    """
    
    def __init__(self):
        pass
    
    def fetch(self):
        """
        Scrapes health data from the DATASUS TABNET website.
        Requires up to 3 hours to run
        """

        def fe_he_mo():
            """
            Fetch (scrape) mortality data from the DATASUS TABNET website.
            """   
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
            
            def worker(mode):
                # Connect to the WebDriver
                driver = webdriver.Remote(command_executor='http://localhost:4444/wd/hub', options=options)
                
                # Years to query
                if mode == "pre":
                    years = list(range(79, 95))
                elif mode == "post":
                    years = list(range(96, 100)) + list(range(0, 22))
                years = [str(x).zfill(2) for x in years]

                # Dictionary to store the data
                out_df = {year: None for year in years}
                
                try:
                    for year in years:
                        # Open the URL
                        if mode == "pre":
                            driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt09br.def")
                        elif mode == "post":
                            driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10br.def")
                            
                        # Wait for the page to load
                        time.sleep(3)  # Adjust the sleep time as needed
                        
                        # Select 'Faixa Etária' from the 'Coluna' dropdown
                        driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Faixa_Etária']").click()
                        
                        # If the year is not "22", select the corresponding year option
                        if ((not year == "22") and (mode == "pre")) or ((not year == "95") and (mode == "post")):
                            driver.find_element(By.XPATH, f"//option[@value='obtbr{year}.dbf']").click()
                        
                        # Select the 'prn' format
                        driver.find_element(By.XPATH, "//input[@name='formato' and @value='prn']").click()
                        
                        # Click the submit button
                        driver.find_element(By.XPATH, "//input[@class='mostra']").click()
                        
                        # Switch to the new window
                        driver.switch_to.window(driver.window_handles[-1])
                        
                        # Wait for the data to be displayed
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
                        
                        # Extract the data from the 'pre' tag
                        data = driver.find_element(By.XPATH, "//pre").text
                        
                        # Read the data as a CSV from the string and store it in the dictionary
                        out_df[year] = pd.read_csv(io.StringIO(data), sep=';', encoding='latin1')
                        
                        # Close the current window
                        driver.close()
                        
                        # Switch back to the original window
                        driver.switch_to.window(driver.window_handles[0])
                        
                        # Optional: wait a bit before the next iteration
                        time.sleep(2)  # Adjust the sleep time as needed
                        
                        # Click the reset button
                        driver.find_element(By.XPATH, "//input[@class='limpa']").click()

                finally:
                    # Quit the WebDriver
                    driver.quit()
                    
                ## Data Postprocessing

                # Concatenate all dataframes in the dictionary into a single dataframe
                out_df = pd.concat(out_df)

                # Reset index and set 'year' as a column
                out_df = out_df.reset_index(level=0, names=["year"])

                # Adjust the 'year' column values (assuming years > 22 are in the 1900s and the rest are in the 2000s)
                out_df["year"] = out_df.year.astype(int).apply(lambda x: x + 1900 if x > 22 else x + 2000)

                # List of columns that need fixing (converting '-' to '0' and then to float)
                fix_cols = [
                    'Menor 1 ano', '1 a 4 anos', '5 a 9 anos',
                    '10 a 14 anos', '15 a 19 anos', '20 a 29 anos', '30 a 39 anos',
                    '40 a 49 anos', '50 a 59 anos', '60 a 69 anos', '70 a 79 anos',
                    '80 anos e mais', 'Idade ignorada'
                ]
                # Replace '-' with '0' and convert columns to float32
                out_df[fix_cols] = out_df[fix_cols].apply(lambda x: x.str.replace("-", "0"), axis=0).astype("float32")

                # Extract municipality ID and name from the 'Município' column
                out_df["mun_id"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)
                out_df["mun_name"] = out_df.Município.str.extract(r"\d{6}(.*)")[0].str.strip()

                # Drop the original 'Município' column as it's no longer needed
                out_df.drop(columns=["Município"], inplace=True)

                # Reorder columns to make 'mun_id', 'mun_name', and 'year' the first columns
                out_df = out_df[["mun_id", "mun_name", "year"] + [col for col in out_df.columns if col not in ["mun_id", "mun_name", "year"]]]

                # Rename columns to more parsable English names
                out_df.columns = [
                    'mun_id', 'mun_name', 'year', 'under_1', '1_to_4', '5_to_9', '10_to_14', '15_to_19',
                    '20_to_29', '30_to_39', '40_to_49', '50_to_59', '60_to_69', '70_to_79',
                    '80_and_more', 'age_unknown', 'total'
                ]

                # Drop rows with any missing values and save the cleaned dataframe to a CSV file
                out_df.dropna().to_csv(f"data/health/scraping_{mode}_1996.csv", index=False)

            worker("pre")
            worker("post")

        def fe_he_ho():
            """
            Fetch (scrape) hospital data from the DATASUS TABNET website.
            """   
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
            
            def worker(mode):
                ### --- OPTION "waterborne" NOT YET IMPLEMENTED ---
                
                # Connect to the WebDriver
                driver = webdriver.Remote(command_executor='http://localhost:4444/wd/hub', options=options)
                
                years = list(range(8, 22 + 1))
                years = [str(x).zfill(2) for x in years]

                # Dictionary to store the data
                out_df = {year: None for year in years}
                
                try:
                    for year in years:
                        # Open the URL
                        driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/qibr.def")
                        
                        # Wait for the page to load
                        time.sleep(3)  # Adjust the sleep time as needed
                        
                        # Select 'Faixa Etária' from the 'Coluna' dropdown
                        #driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Faixa_Etária']").click()
                        
                        # Select 'Valor aprovado' from the 'Incremento' dropdown
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='AIH_aprovadas']").click()
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='Internações']").click()
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='Valor_total']").click()
                        
                        # choose time period to query
                        driver.find_element(By.XPATH, "//option[@value='qibr2404.dbf']").click()
                        months = driver.find_elements(By.XPATH, f"//option[contains(@value, 'qibr{year}')]")
                        for month in months:
                            time.sleep(.2)
                            month.click()
                            
                        if mode == "waterborne":
                            # List of IDs corresponding to the queried medical procedures
                            procedure_ids = [
                                "0202040119", "0202040127", "0202040178",  # Stool Examination
                                "0213010240", "0213010275", "0213010216", "0213010453", "0202030750", "0202030873", "0202030776", "0213010020",  # Blood Tests
                                "0202080153", "0202020037", "0202020029", "0202020118", "0202010651", "0202010643",  # Blood Tests continued
                                "0214010120", "0214010139", "0214010180", "0214010058", "0214010104", "0214010090",  # Rapid Diagnostic Tests (RDTs)
                                "0213010208", "0213010194", "0213010186", "0213010011",  # PCR (Polymerase Chain Reaction)
                                "0301100209",  # Hydration Therapy
                                "0301100241", "0303010045", "0303010061",  # Antibiotic Treatment
                                "0303010100", "0303010150",  # Antiparasitic Treatment
                                "0303010118",  # Antiviral and Supportive Care
                                "0213010216", "0213010267",  # Antimalarial Treatment
                                "0303010142", "0303020032", "0303060301", "0303070129"  # Symptomatic Treatment
                            ]
                            
                            driver.find_element(By.XPATH, "//img[@id='fig15']").click()
                            time.sleep(1)
                            
                            driver.find_element(By.XPATH, f"//option[contains(text(), 'Todas as categorias')]").click()
                            for option_str in procedure_ids:
                                # select the procedure by its name
                                driver.find_element(By.XPATH, f"//option[contains(text(), '0101010010')]").click()
                                
                                driver.find_element(By.XPATH, f"//option[contains(text(), '{option_str}')]").click()
                        
                        # Select the 'prn' format
                        driver.find_element(By.XPATH, "//input[@name='formato' and @value='prn']").click()
                        
                        # Click the submit button
                        driver.find_element(By.XPATH, "//input[@class='mostra']").click()
                        
                        # Switch to the new window
                        driver.switch_to.window(driver.window_handles[-1])
                        
                        # Wait for the data to be displayed
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
                        
                        # Extract the data from the 'pre' tag
                        data = driver.find_element(By.XPATH, "//pre").text
                        
                        # Read the data as a CSV from the string and store it in the dictionary
                        out_df[year] = pd.read_csv(io.StringIO(data), sep=';', encoding='latin1')
                        
                        # Close the current window
                        driver.close()
                        
                        # Switch back to the original window
                        driver.switch_to.window(driver.window_handles[0])
                        
                        # Optional: wait a bit before the next iteration
                        time.sleep(2)  # Adjust the sleep time as needed
                        
                        # Click the reset button
                        driver.find_element(By.XPATH, "//input[@class='limpa']").click()

                        pickle.dump(out_df, open(f"/home/ubuntu/ext_drive/scraping/Masterthesis/data/hospital/tmp_scraping.pkl", "wb"))
                        
                    # Concatenate out_df
                    out_df = pd.concat(out_df)
                    
                    # Reset index and set 'year' as a column
                    out_df = out_df.reset_index(level=0, names=["year"])

                    # Adjust the 'year' column values (assuming years > 22 are in the 1900s and the rest are in the 2000s)
                    out_df["year"] = out_df.year.astype(int).apply(lambda x: x + 1900 if x > 22 else x + 2000)

                    # Extract municipality ID and name from the 'Município' column
                    out_df["CC_2r"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)

                    # Drop the original 'Município' column as it's no longer needed
                    out_df.drop(columns=["Município"], inplace=True)

                    # Reorder columns to make 'CC_2r', 'mun_name', and 'year' the first columns
                    out_df = out_df[["CC_2r", "year"] + [col for col in out_df.columns if col not in ["CC_2r", "mun_name", "year"]]]

                    # Rename columns to more parsable English names
                    out_df.columns = [
                        'CC_2r', 'year', 'n_approved', 'hospitalizations', 'total_value'
                    ]

                    if not mode == "waterborne":
                        out_df.dropna().to_parquet("data/health/hospitalizations.parquet", index=False)
                    if mode == "waterborne":
                        out_df.dropna().to_parquet("data/health/hospitalizations_waterborne.parquet", index=False)
                
                finally:
                    # Quit the WebDriver
                    driver.quit()
            
            worker()
        
        # ===========================================================
        # Execute scraping
        # ===========================================================
        
        fe_he_mo()
        fe_he_ho()  
            
    
    def preprocess(self):
        pass