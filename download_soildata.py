from gee_datasets.soil import GEESoilGrids
import ee
import os
import sys

import yaml
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

    xrdata = xarray.merge(xrdata_list)
    
    xrdata.to_netcdf(output_path)
    
    return xrdata


def main(config_path):
    
    assert os.path.exists(config_path), "the path does not exist"
    

    print(f'-------> Starting: ', config_path)
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    cm_path = config_dict['GENERAL_SETTINGS'].get('dssat_processor_path', None)
    if cm_path is None:
        path = os.path.abspath(os.path.join(os.getcwd(),'/WeatherSoilDataProcessor'))
    else:
        path = os.path.abspath(os.path.join(cm_path,'/WeatherSoilDataProcessor'))
    
    sys.path.append(path)

    ee.Initialize(project = config_dict['GENERAL_SETTINGS']['ee_project_name'])
    
    data_downloader = GEESoilGrids(config_dict['DATA_DOWNLOAD']['ADM0_NAME'])

    
    if config_dict['GENERAL_SETTINGS']['donwnload_data_cube']:
        adm_level = config_dict['DATA_DOWNLOAD']['adm_level']
        export_data_cube(data_downloader, 
                config_dict['DATA_DOWNLOAD']['output_path'], 
                config_dict['DATA_DOWNLOAD']['properties'], 
                adm_level = config_dict['DATA_DOWNLOAD']['adm_level'], 
                locality_name = config_dict['DATA_DOWNLOAD'][f'{adm_level}_NAME'],
                depths = config_dict['DATA_DOWNLOAD']['depths'], 
                scale = config_dict['DATA_DOWNLOAD']['scale'])
        
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
                    
if __name__ == '__main__':
    print('''\
      
      ==============================================
      |                                            |
      |           AGWISE DATA SOURCING             |    
      |               GEESOILData                  |
      |              Crop Modeling                 |
      ==============================================      
      ''')

    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)



    


    