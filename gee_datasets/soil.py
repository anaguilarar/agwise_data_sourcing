import os
from typing import List

import ee
import pandas as pd
import requests

from .gee_data import GEEDataDownloader

class GEESoilGrids(GEEDataDownloader):
  def __init__(self, country) -> None:
  
    self.country = country.lower()
    self._adm_filter = None
    self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

    
    dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
    self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

    count = self.country_filter.size().getInfo()
    assert count == 1, f'No info for {self.country.title()}'

  @staticmethod  
  def extract_data_using_coordinate(image, point_coordinate, soil_property, scale = 250):
    ee_point = ee.Geometry.Point(point_coordinate)  
    sample = image.sample(region=ee_point, scale=scale).first().getInfo()
    samplesdf = []
    for k, v in sample['properties'].items():
        if soil_property in ['wv0010', 'wv0033', 'wv1500']:
            depth = k[len('val')+1:k.index('cm')].replace('_','-')
        elif soil_property in ['calcium', 'phosphorus', 'potassium']:
          depth = k[len('mean')+1:].replace('_','-')
        else:
            depth = k[len(soil_property)+1:k.index('cm')].replace('_','-')
        df = pd.DataFrame({'depth': [depth], soil_property: [v], 'x': [point_coordinate[0]], 'y': [point_coordinate[1]]})
        samplesdf.append(df)
        
    return pd.concat(samplesdf)
    
  @property
  def list_of_products(self):
    return {
        'bdod': "projects/soilgrids-isric/bdod_mean",
        'cec': "projects/soilgrids-isric/cec_mean",
        'cfvo': "projects/soilgrids-isric/cfvo_mean",
        'clay': "projects/soilgrids-isric/clay_mean",
        'sand': "projects/soilgrids-isric/sand_mean",
        'silt': "projects/soilgrids-isric/silt_mean",
        'nitrogen': "projects/soilgrids-isric/nitrogen_mean",
        'soc': "projects/soilgrids-isric/soc_mean",
        'phh2o': "projects/soilgrids-isric/phh2o_mean",
        'wv0010': "ISRIC/SoilGrids250m/v2_0/wv0010",
        'wv0033': "ISRIC/SoilGrids250m/v2_0/wv0033",
        'wv1500': "ISRIC/SoilGrids250m/v2_0/wv1500",
        'calcium': "ISDASOIL/Africa/v1/calcium_extractable",
        'phosphorus': "ISDASOIL/Africa/v1/phosphorus_extractable",
        'potassium': "ISDASOIL/Africa/v1/potassium_extractable"
    }

  def initialize_query(self, soil_property: str = None, depths: List[str] = None):
    """
    function to inisitalize the query
    """
        
    assert soil_property in self.list_of_products, 'Product not available'

    self.query = ee.Image(self.list_of_products[soil_property])#.first()

    self.query = self.query.clip(self.country_filter)
    if depths is not None:
      if soil_property in ['wv0010', 'wv0033', 'wv1500']:
        band_names = ['val_' + depth_name + 'cm_mean' for depth_name in depths]
      elif soil_property in ['calcium', 'phosphorus', 'potassium']:
        band_names = [ 'mean_' + depth_name for depth_name in depths]
      else:
        band_names = [soil_property + '_' + depth_name.replace('_','-') + 'cm_mean' for depth_name in depths]
      
      self.query = self.query.select(band_names)
      
    return self.query
  
  def download_data(self, soil_image, output_fn,  scale = 250):
    soil_image = soil_image.reproject(crs="EPSG:4326", scale=scale)
    url = soil_image.getDownloadURL({
      'scale': scale,
      'region': self._adm_filter.geometry(),
      'format': 'GEO_TIFF',
    })
    
    #fn = os.path.dirname(output_fn)
    response = requests.get(url)
    with open(output_fn, 'wb') as f:
      f.write(response.content)
  
  def soildata_using_point(self, soil_properties, point_coordinate, depths = None, scale = 250):
    soildf = None
    
    for soil_property in soil_properties:
        image = self.initialize_query(soil_property, depths= depths)
        dfval = self.extract_data_using_coordinate(image, point_coordinate, soil_property, scale = scale)
        soildf = dfval if soildf is None else pd.merge(soildf, dfval, on = ['depth', 'x', 'y'])
      
    return soildf
    
  def download_multiple_properties(self, output_dir, soil_properties = None, adm_level = None, feature_name = None, scale = 250, depths = None):
    if soil_properties is None: soil_properties = self.list_of_products.keys()
    
    for soil_property in soil_properties:
      self.initialize_query(soil_property, depths= depths)
      if feature_name is not None:
        soil_image = self.get_adm_level_data(adm_level=adm_level, feature_name = feature_name)
      else:
        soil_image = self.query
        
      fn = os.path.join(output_dir, soil_property + '.tif')
      if not os.path.exists(output_dir): os.mkdir(output_dir)
      
      self.download_data(soil_image, fn, scale = scale)
      print(f'{soil_property}: data was downloaded in {fn}')
  
  def get_adm_level_data(self, adm_level = None, feature_name = None):
    if adm_level is not None and feature_name is not None:
      dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level=adm_level))
      adm_filter = dataset.filter(ee.Filter.eq('shapeName', feature_name.lower().title()))
      print(f'data will be processed for: {feature_name}')
      self._adm_filter = adm_filter
      return self.query.clip(adm_filter)
    else:
      return self.query
    
