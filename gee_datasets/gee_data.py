import ee

import pandas as pd
import matplotlib.pyplot as plt

def fill_gaps_linear(collection, band):
    # Sort collection by time
    collection = collection.sort('system:time_start')

    def interp(img):
        prev = collection.filterDate(
            ee.Date(img.get('system:time_start')).advance(-32, 'day'),
            ee.Date(img.get('system:time_start'))
        ).limit(1, 'system:time_start', False)

        next_ = collection.filterDate(
            ee.Date(img.get('system:time_start')),
            ee.Date(img.get('system:time_start')).advance(32, 'day')
        ).limit(1)

        prev_val = ee.Image(prev.first()).select(band)
        next_val = ee.Image(next_.first()).select(band)

        # Interpolation fraction
        t = ee.Image(img).date().millis()
        t0 = ee.Number(prev.first().get('system:time_start'))
        t1 = ee.Number(next_.first().get('system:time_start'))

        frac = t.subtract(t0).divide(t1.subtract(t0))

        interp_val = prev_val.add(next_val.subtract(prev_val).multiply(frac))
        return img.addBands(interp_val.rename(band + '_interp'), overwrite=True)

    return collection.map(interp)

def moving_average(collection, band, window=3):
    # Must be odd
    half = (window - 1) // 2

    def smooth(img):
        date = ee.Date(img.get('system:time_start'))
        start = date.advance(-16 * half, 'day')
        end = date.advance(16 * half, 'day')

        neigh = collection.filterDate(start, end).select(band)
        mean = neigh.mean().rename(band + '_smoothed')
        return img.addBands(mean, overwrite=True)

    return collection.map(smooth)

def summarize_collection_tots(image, roi, band, scale = 250):

  reduced_value = image.select(band).reduceRegion(
      reducer=ee.Reducer.mean(),
      geometry=roi.geometry(),
      scale=scale,
      maxPixels = 1e13
  )

  return ee.Feature(None, {
      'date': image.date().format(),
      band: reduced_value.get(band)
  })


from abc import ABC, abstractmethod
from datetime import datetime

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

  def download_data(self):
    pass