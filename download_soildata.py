from gee_datasets.soil import GEESoilGrids
from pathlib import Path

import ee
import os
import sys
from tqdm import tqdm
import concurrent.futures

import yaml
import numpy as np
import xarray
import rioxarray as rio

def initialize_ee(project_id):
    ee.Initialize(project_id)
    

def get_soil_data_as_table(data_downloader, coordinate, soil_properties, depths):

    df = data_downloader.soildata_using_point(soil_properties,
                                        coordinate,
                                        depths= depths)
    
    for column_name in df.columns:
        if column_name.startswith('wv'):
            df[column_name] = df[column_name] * 1000
        if column_name == 'nitrogen':
            df[column_name] = df[column_name] / 10
    
    return df

def export_coordinate_referenceasdssat(data_downloader,  coordinate_id, coordinate, soil_properties, depths, soil_id, output_path, output_fn = 'SOL', site = 'AFR'):
    from crop_modeling.dssat.files_export import from_soil_to_dssat
    ind_pixel_path= os.path.join(output_path, str(coordinate_id))
    
    if not os.path.exists(ind_pixel_path): os.mkdir(ind_pixel_path)
    
    soil_df = get_soil_data_as_table( data_downloader, coordinate, soil_properties, depths)
    
    from_soil_to_dssat(soil_df, 
                    outputpath = ind_pixel_path, 
                    outputfn = output_fn, soil_id=soil_id, 
                    country=data_downloader.country.title(), site = site)
    

def export_dssat_table(data_downloader, coordinate, soil_properties, depths, soil_id, output_path, output_fn = 'SOL', site = 'AFR'):
    from crop_modeling.dssat.files_export import from_soil_to_dssat
    
    soil_df = get_soil_data_as_table( data_downloader, coordinate, soil_properties, depths)
    
    from_soil_to_dssat(soil_df, 
                    outputpath = output_path, 
                    outputfn = output_fn, soil_id=soil_id, 
                    country=data_downloader.country.title(), site = site)

def export_data_cube(data_downloader, output_path, soil_properties, adm_level, locality_name, depths, scale = 250 ):
    
    tmp_dir = 'soil'
    if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)
    
    data_downloader.download_multiple_properties('soil', 
                    soil_properties,
                    adm_level=adm_level,
                    feature_name = locality_name,
                    scale = scale,
                    depths= depths)
    
    raster_list = [os.path.join(tmp_dir,i) for i in os.listdir('soil') if i.endswith('tif')]
    xrdata_list = []
    
    for i in range(len(raster_list)):
        xrdata = rio.open_rasterio(raster_list[i]).rename({'band': 'depth'})
        xrdata.name = os.path.basename(raster_list[i])[:-4]

        xrdata_list.append(xrdata)

    soilm = xarray.merge(xrdata_list).assign_coords({'depth': np.array([i.replace('_', '-') for i in depths])})
    
    soilm.to_netcdf(output_path)
    
    return soilm

def export_individual_pixelasdssatformat(soilm, idpx, xcoord, ycoord, soil_id, output_path, locality_name):
    
    from crop_modeling.dssat.files_export import from_soil_to_dssat
    
    dfdata = soilm.sel(x = xcoord, y = ycoord, method = 'nearest').to_dataframe().reset_index().dropna()
    if all(dfdata.wv0010<-32000):
        dfdata = dfdata.drop(columns='wv0010')
    if all(dfdata.wv0033<-32000):
        dfdata = dfdata.drop(columns='wv0033')
    if all(dfdata.wv1500<-32000):
        dfdata = dfdata.drop(columns='wv1500')
    
    from_soil_to_dssat(dfdata, depth_name= 'depth',
                                        outputpath= output_path, 
                                        outputfn='SOL', 
                                        country = 'Kenya'.upper(),
                                        site = locality_name, 
                                        soil_id=soil_id, 
                                        sub_working_path = str(idpx), verbose = False) 
    

def main(config_path):
    
    assert os.path.exists(config_path), "the path does not exist"
    

    print(f'-------> Starting: ', config_path)
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    
    cm_path = Path(config_dict['DSSAT_process'].get('dssat_processor_path', None))

    if cm_path is None:
        path = os.path.abspath(os.path.join(os.getcwd(),'/WeatherSoilDataProcessor'))
    else:
        path = os.path.abspath(cm_path / Path('WeatherSoilDataProcessor'))
    
    print('dssat processor path: ', path)
    assert os.path.exists(path), "the dssat processor path does not exist"
    sys.path.append(path)

    ee.Initialize(project = config_dict['GENERAL_SETTINGS']['ee_project_name'])
    
    data_downloader = GEESoilGrids(config_dict['DATA_DOWNLOAD']['ADM0_NAME'])

    
    if config_dict['GENERAL_SETTINGS']['donwnload_data_cube']:
        
        adm_level = config_dict['DATA_DOWNLOAD']['adm_level']
        ouput_path = os.path.join(config_dict['GENERAL_SETTINGS']['output_path'], config_dict['DATA_DOWNLOAD']['ADM1_NAME'])
        if not os.path.exists(ouput_path): os.makedirs(ouput_path)
        
        xrdata = export_data_cube(data_downloader, 
                config_dict['DATA_DOWNLOAD']['output_path'], 
                config_dict['DATA_DOWNLOAD']['properties'], 
                adm_level = config_dict['DATA_DOWNLOAD']['adm_level'], 
                locality_name = config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME'],
                depths = config_dict['DATA_DOWNLOAD']['depths'], 
                scale = config_dict['DATA_DOWNLOAD']['scale'])
        
        
        xrref = xrdata.isel(depth = 0)

        xrmask = xrref.notnull()[list(xrdata.data_vars)[0]]
        xgrid, ygrid = np.meshgrid(xrref.x,xrref.y)
        xgrid = np.where(xrmask.values,xgrid,np.nan).flatten()
        ygrid = np.where(xrmask.values,ygrid,np.nan).flatten()
        pxswithdata = np.where(~np.isnan(ygrid))[0]
        
        
    if config_dict['GENERAL_SETTINGS']['donwnload_coordinatedata_asdssat']:
        if not os.path.exists(config_dict['GENERAL_SETTINGS']['output_path']): os.makedirs(config_dict['GENERAL_SETTINGS']['output_path'])
        export_dssat_table(data_downloader, 
                        coordinate = config_dict['DATA_DOWNLOAD']['coordinate'], 
                        soil_properties = config_dict['DSSAT_process']['soil_properties'], 
                        depths = config_dict['DATA_DOWNLOAD']['depths'], 
                        soil_id = config_dict['DSSAT_process']['soil_id'], 
                        output_path = config_dict['GENERAL_SETTINGS']['output_path'],
                        output_fn = config_dict['DSSAT_process']['output_fn'],
                        site = 'AFR')
        
    if config_dict['GENERAL_SETTINGS'].get('donwnload_area_asdssat', False):
        
        if not os.path.exists(config_dict['GENERAL_SETTINGS']['output_path']): os.makedirs(config_dict['GENERAL_SETTINGS']['output_path'])
        output_dir = config_dict['GENERAL_SETTINGS']['output_path']
        max_workers = config_dict['DATA_DOWNLOAD'].get('n_workers', 5)
        ##create raster ref
        soil_property = 'sand'
        data_downloader.initialize_query(soil_property, depths= ['0_5'])
        adm_level = config_dict['DATA_DOWNLOAD']['adm_level']
        feature_name = config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME']
        soil_image = data_downloader.get_adm_level_data(adm_level=adm_level, feature_name = feature_name)
        fn = os.path.join(output_dir, soil_property + '.tif')
        
        data_downloader.download_data(soil_image, fn, scale = 1000)
        
        dataref = rio.open_rasterio(os.path.join(output_dir, f'{soil_property}.tif'))
        
        dataref.isel(band = 0).plot()

        xrmask = dataref.notnull()
        xgrid, ygrid = np.meshgrid(dataref.x,dataref.y)
        xgrid = np.where(xrmask.values,xgrid,np.nan).flatten()
        ygrid = np.where(xrmask.values,ygrid,np.nan).flatten()

        pxswithdata = np.where(~np.isnan(ygrid))[0]
        
        with tqdm(total=len(pxswithdata)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tr ={executor.submit(export_coordinate_referenceasdssat, data_downloader, 
                        coordinate_id = idpx,
                        coordinate = [xgrid[pxswithdata[idpx]], ygrid[pxswithdata[idpx]]], 
                        soil_properties = config_dict['DSSAT_process']['soil_properties'], 
                        depths = config_dict['DATA_DOWNLOAD']['depths'], 
                        soil_id = config_dict['DSSAT_process']['soil_id'], 
                        output_path = config_dict['GENERAL_SETTINGS']['output_path'],
                        output_fn = config_dict['DSSAT_process']['output_fn'],
                        site = 'AFR'): (idpx) for idpx in pxswithdata}

                for future in concurrent.futures.as_completed(future_to_tr):
                    idpx = future_to_tr[future]
                    try:
                        future.result()
                            
                    except Exception as exc:
                            print(f"Request for treatment {idpx} generated an exception: {exc}")
                    pbar.update(1)
                    
if __name__ == '__main__':
    print('''\
      
            ============================================
            |                                          |
            |           AGWISE DATA SOURCING           |    
            |               GEESOILData                |
            |              Crop Modeling               |
            ============================================      
      ''')

    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)



    


    