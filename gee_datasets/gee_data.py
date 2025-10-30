import ee

import pandas as pd
import matplotlib.pyplot as plt
from .processing_funs import summarize_collection_tots, fill_gaps_linear, smooth_ts_using_savitsky_golay_modis
from abc import ABC, abstractmethod
from datetime import datetime

from typing import List
class GEEDataDownloader(ABC):
  @abstractmethod
  def list_of_products(self):
    return

  @abstractmethod
  def initialize_query(self, starting_date: str, ending_date: str):
    pass

  @abstractmethod
  def download_data(self):
    pass


class GEESoilGrids(GEEDataDownloader):
  def __init__(self, country) -> None:
      
        self.country = country.lower()
        self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

        
        dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
        self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

        count = self.country_filter.size().getInfo()
        assert count == 1, f'No info for {self.country.title()}'
        
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
        'wv1500': "ISRIC/SoilGrids250m/v2_0/wv1500" 
    }


  def initialize_query(self, soil_property: str = None, depths: List[str] = None):
    """
    function to inisitalize the query
    """
        
    assert soil_property in self.list_of_products, 'Product not available'

    self.query = ee.Image(self.list_of_products[soil_property])#.first()

    self.query = self.query.clip(self.country_filter)
        
    return self.query
  
  def download_data(self):
    pass
  
  def get_adm_level_data(self, adm_level = None, feature_name = None):
    if adm_level is not None and feature_name is not None:
      dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level=adm_level))
      adm_filter = dataset.filter(ee.Filter.eq('shapeName', feature_name.lower().title()))
      print(f'data will be processed for: {feature_name}')
      return self.query.clip(adm_filter)
    else:
      return self.query
    
      
  

class GEECropMask(GEEDataDownloader):
  @property
  def list_of_products(self):
    return {
        'ESA': 'ESA/WorldCover/v200',
        'DYNAMICWORLD': 'GOOGLE/DYNAMICWORLD/V1'
    }

  @property
  def crop_mask_value(self):
    return {
        'ESA': 40,
        'DYNAMICWORLD': 4
    }

  def __init__(self, country, product) -> None:
    self.country = country.lower()
    self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

    assert product in self.list_of_products, 'Product not available'
    self.product = product
    dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
    self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

    count = self.country_filter.size().getInfo()
    assert count == 1, f'No info for {self.country.title()}'


  def initialize_query(self, year: str = None):
    """
    function to inisitalize the query
    """
    if self.product == 'DYNAMICWORLD':
      if year is None: year = datetime.now().year

      self.query = ee.ImageCollection(self.list_of_products[self.product]).filterDate(f'{year}-01-01', f'{year}-12-31')
      #self.query = self.query.mean()

    else:
      self.query = ee.ImageCollection(self.list_of_products[self.product])#.first()

    self.query = self.query.filterBounds(self.country_filter)
    return self.query

  def download_data(self):
    pass

class GEEMODIS(GEEDataDownloader):
  @staticmethod
  def mask_low_quality_pixels(image):

    """Masks pixels that are not of the highest quality."""
    qa_band = image.select('DetailedQA')


    vi_quality = qa_band.rightShift(0).bitwiseAnd(3)
    good_quality_mask = vi_quality.eq(0)

    # Bits 2-5: VI Usefulness. Highest quality is '0000'.
    vi_usefulness = qa_band.rightShift(2).bitwiseAnd(15)
    highest_usefulness_mask = vi_usefulness.lte(2)

    # Combine both masks to get the best quality pixels.
    quality_mask = good_quality_mask.And(highest_usefulness_mask)
    return image.updateMask(quality_mask)


  @property
  def list_of_products(self):
    return {
        'MYD13Q1': 'MODIS/061/MYD13Q1',
        'MOD13Q1': 'MODIS/061/MOD13Q1',
        'MOD13A2': 'MODIS/061/MOD13A2',
        'VNP13A1': 'NASA/VIIRS/002/VNP13A1'
    }

  def __init__(self, country, product):
    self.country = country.lower()
    self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

    assert product in self.list_of_products, 'Product not available'
    self.product = product
    dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
    self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

    count = self.country_filter.size().getInfo()
    assert count == 1, f'No info for {self.country.title()}'


  def initialize_query(self, starting_date: str, ending_date: str):
    """
    function to inisitalize the query
    """
    self.starting_date = starting_date
    self.ending_date = ending_date

    self.query = ee.ImageCollection(self.list_of_products[self.product]).filterDate(self.starting_date, self.ending_date)
    self.query = self.query.filterBounds(self.country_filter)
    self.query = self.query.map(self.mask_low_quality_pixels)
    return self.query

  def plot_time_series(self, band = 'NDVI', adm_level = 'ADM0'):
    if band is None: band = 'NDVI'

    time_series_features = self.query.map(lambda image: summarize_collection_tots(image, self.country_filter, band))

    features = time_series_features.getInfo()['features']
    print(features)
    df = pd.DataFrame([
        {
            'date': f['properties']['date'],
            band: f['properties'][band]
        }
        for f in features if f['properties'][band] is not None
    ])

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df[band], marker='o', linestyle='-')
    plt.title(f"{band} Time Series for {self.country.title()}")
    plt.xlabel("Date")
    plt.ylabel(band)
    plt.grid(True)
    plt.show()

    return df

  def get_adm_timeseries(self, adm_level = None, feature_name = None, band = 'NDVI', fill_gaps = True, sg = True, **kwargs):
    if adm_level is not None and feature_name is not None:
      dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level=adm_level))
      adm_filter = dataset.filter(ee.Filter.eq('shapeName', feature_name.lower().title()))
      print(f'data will be processed for: {feature_name}')
      query = self.query.filterBounds(adm_filter)
      
    else:
      query = self.query
      adm_filter = self.country_filter
      print('data will be processed at country level')
      
    if fill_gaps: query = fill_gaps_linear(query, band)
      
    if sg:
      query = smooth_ts_using_savitsky_golay_modis(query.select(band), **kwargs)
      band = band  + '_smooth'
      
    time_series_features = query.map(lambda image: summarize_collection_tots(image, adm_filter, band))
    
    features = time_series_features.getInfo()['features']
    
    df = pd.DataFrame([
        {
            'date': f['properties']['date'],
            band: f['properties'][band]
        }
        for f in features if f['properties'][band] is not None
    ])

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df
    
  def download_data(self):
    pass