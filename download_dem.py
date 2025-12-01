from gee_datasets.dem import GEEdem
from pathlib import Path

import ee
import os
import sys
from tqdm import tqdm
import concurrent.futures

import yaml
import pandas as pd
import numpy as np
import xarray
import rioxarray as rio

def initialize_ee(project_id):
    ee.Initialize(project_id)
    

def main(config_path):
    
    assert os.path.exists(config_path), "the path does not exist"
    

    print(f'-------> Starting: ', config_path)
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    

    ee.Initialize(project = config_dict['GENERAL_SETTINGS']['ee_project_name'])
    
    data_downloader = GEEdem(config_dict['DATA_DOWNLOAD']['ADM0_NAME'])

    
    ouput_path = config_dict['GENERAL_SETTINGS']['output_path']
    if not os.path.exists(ouput_path): os.makedirs(ouput_path)
        
    if config_dict['GENERAL_SETTINGS']['donwnload_as_raster']:
        data_downloader.initialize_query(config_dict['DATA_DOWNLOAD']['source'])
        adm_level = config_dict['DATA_DOWNLOAD']['adm_level']    
        sp_scale = config_dict['DATA_DOWNLOAD']['scale']
        possiblespscales = [30,250,1000,5000]
        output_fn = os.path.join(ouput_path, config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME'].lower() + f'_{sp_scale}.tif')
        while(not os.path.exists(output_fn)):
            print(sp_scale)
            dem_image = data_downloader.get_adm_level_data(adm_level='ADM1', feature_name = config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME'])
            try:
                data_downloader.download_data(dem_image, output_fn,  scale = sp_scale)
            except:
                for i in range(len(possiblespscales)-1): 
                    if sp_scale >= possiblespscales[i] and sp_scale < possiblespscales[i+1]:
                        sp_scale= possiblespscales[i+1]
                        output_fn = os.path.join(ouput_path, config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME'].lower() + f'_{sp_scale}.tif')
                        break
                
                print(f"The resolution is to high for the gee capacity, the resolution was changed to: {sp_scale}")
        
        print(f'-------> DEM data sownloaded in: ', output_fn)
        
    if config_dict['GENERAL_SETTINGS']['donwnload_coordinatedata']:
        if not os.path.exists(config_dict['GENERAL_SETTINGS']['output_path']): os.makedirs(config_dict['GENERAL_SETTINGS']['output_path'])
        coordinate = config_dict['DATA_DOWNLOAD']['coordinate']
        sp_scale = config_dict['DATA_DOWNLOAD']['scale']
        df = data_downloader.terraindata_using_point(coordinate,  scale = sp_scale)
        output_fn = os.path.join(ouput_path, 'point.csv')
        if os.path.exists(output_fn):
            prev_df = pd.read_csv(output_fn)
            df = pd.concat([prev_df[['band', 'value', 'x', 'y']], df])
        
        df.to_csv(output_fn)
        

if __name__ == '__main__':
    print('''\
      
            ========================================
            |                                      |
            |         AGWISE DATA SOURCING         |    
            |               GEEdem                 |
            |                                      |
            ========================================      
      ''')

    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)

        
        