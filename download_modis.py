from datetime import datetime

import ee
import os
import sys

import yaml
import pandas as pd
import numpy as np
import xarray
import rioxarray as rio

from gee_datasets.modis import GEEMODIS
from gee_datasets.gee_data import GEECropMask


def initialize_ee(project_id):
    ee.Initialize(project_id)

def get_crop_mask(configuration):
    cropmask_downloader = GEECropMask(configuration['DATA_DOWNLOAD']['ADM0_NAME'], configuration['PREPROCESSING']['crop_mask_product'])
    cropmask_downloader.initialize_query()
    return ee.Image(cropmask_downloader.query.first()).clip(cropmask_downloader.country_filter).eq(
        cropmask_downloader.crop_mask_value[cropmask_downloader.product])

def main(config_path):
    
    assert os.path.exists(config_path), "the path does not exist"
    

    print(f'-------> Starting: ', config_path)
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    

    ee.Initialize(project = config_dict['GENERAL_SETTINGS']['ee_project_name'])
    
    data_downloader = GEEMODIS(config_dict['DATA_DOWNLOAD']['ADM0_NAME'], config_dict['DATA_DOWNLOAD']['product'])
    data_downloader.initialize_query(config_dict['DATA_DOWNLOAD']['starting_date'], config_dict['DATA_DOWNLOAD']['ending_date'])
    
    ## crop mask
    if config_dict['PREPROCESSING'].get('crop_mask', False):
        crop_mask = get_crop_mask(config_dict)
    else:
        crop_mask = False
    
    # get administration level data
    adm_level = config_dict['DATA_DOWNLOAD'].get('adm_level', 'ADM0')
    feature_name = config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME']
    band = config_dict['DATA_DOWNLOAD'].get('band', 'NDVI')
    filldata = config_dict['PREPROCESSING'].get('data_filling', False)
    smoothing = config_dict['PREPROCESSING'].get('sg_smoothing', False)
    sg_window = config_dict['PREPROCESSING'].get('sg_window', 3)
    
    
    img_collection = data_downloader.get_adm_level_data(adm_level= adm_level, 
                                                        feature_name=feature_name,  
                                                        band = band, 
                                                        fill_data = filldata, 
                                                        smooth_data = smoothing, 
                                                        crop_mask = crop_mask, 
                                                        adm_filter = None, window_size = sg_window)
    
    # download to local
    
    output_tmpdir = os.path.join(config_dict['GENERAL_SETTINGS']['output_path'], config_dict['DATA_DOWNLOAD']['ADM1_NAME'].lower())
    scale = config_dict['DATA_DOWNLOAD']['scale']
    raster_list = data_downloader.download_data(img_collection, output_dir= output_tmpdir, 
                            feature_geometry = data_downloader._adm_filter.geometry(), scale = scale, img_property = 'system:id')

    
    ## create data cube
    
    band_name, xrdata_list = [], []

    for i in range(len(raster_list)):
        with rio.open_rasterio(raster_list[i]) as xrdata:
            xr_loaded = xrdata.load() 
        band_name.append(os.path.basename(raster_list[i])[:-4])
        xrdata_list.append(xr_loaded)
        
    xrstacked = xarray.concat(xrdata_list, dim = 'band')

    xrstacked = xrstacked.assign_coords({'band': band_name})

    ## export png image
    grid = xrstacked.plot(col = 'band', col_wrap=5)
    grid.fig.savefig(os.path.join(output_tmpdir,'mlt_bands.png'))
    
    ## export as netCDF
    usecase = config_dict['GENERAL_SETTINGS'].get('use_case', None)
    init_date = config_dict['DATA_DOWNLOAD']['starting_date']
    end_date = config_dict['DATA_DOWNLOAD']['ending_date']

    init_year = datetime.strptime(init_date, '%Y-%m-%d').year
    ending_year = datetime.strptime(end_date, '%Y-%m-%d').year

    if not usecase:
        usecase = config_dict['DATA_DOWNLOAD']['ADM1_NAME'].title()
        
    output_fn = '{country}_{usecase}_MODIS_NDVI_{init_year}_{ending_year}_{sg}.tif'.format(
        country = data_downloader.country.title(),
        usecase = usecase,
        init_year = init_year,
        ending_year = ending_year,
        sg = 'SG' if config_dict['PREPROCESSING']['sg_smoothing'] else '')


    xrstacked = xrstacked.assign_attrs(band_names=xrstacked.band.values.tolist())
    fn = os.path.join(output_tmpdir, output_fn)
    xrstacked.rio.to_raster(fn)
    print(f'Stacked file saved in {fn}')
    ## remove files
    for i in raster_list: os.remove(i)

if __name__ == '__main__':
    print('''\
      
            ========================================
            |                                      |
            |         AGWISE DATA SOURCING         |    
            |               GEEMODIS               |
            |                                      |
            ========================================      
      ''')

    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)

        
        