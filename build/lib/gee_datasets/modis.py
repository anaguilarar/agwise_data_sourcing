
import os
import requests

import ee
import pandas as pd
import matplotlib.pyplot as plt

from .processing_funs import summarize_collection_tots, fill_gaps_linear, smooth_ts_using_savitsky_golay_modis
from .gee_data import GEEDataDownloader


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
        

    @staticmethod
    def preprocess_image_collection( imc_query, band = 'NDVI', fill_data = True, smooth_data = True, crop_mask = None, adm_filter = None, **kwargs):
        imgc = fill_gaps_linear(imc_query, band) if fill_data else imc_query
        imgc = smooth_ts_using_savitsky_golay_modis(imgc, band = band, **kwargs) if smooth_data else imgc
        
        if crop_mask is not None:
            imgc = imgc.map(lambda image: image_masking(image, crop_mask, adm_filter))
            
        return imgc
            
    def get_adm_level_data(self, adm_level = None, feature_name = None, band = 'NDVI', fill_data = True, smooth_data = True, crop_mask = None, adm_filter = None, **kwargs):
        if adm_level is not None and feature_name is not None:
            dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level=adm_level))
            adm_filter = dataset.filter(ee.Filter.eq('shapeName', feature_name.lower().title()))
            print(f'data will be processed for: {feature_name}')
            self._adm_filter = adm_filter
            if crop_mask:
                crop_mask = crop_mask.clip(adm_filter)
            
        else:
            self._adm_filter = self.country_filter
            
        return self.preprocess_image_collection(self.query, band=band, fill_data= fill_data, smooth_data=smooth_data, crop_mask=crop_mask, adm_filter = self._adm_filter,**kwargs)
        
    def download_data(self, modis_collection, output_dir, feature_geometry,  scale = 250, img_property = None, band = 'NDVI'):
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        images = modis_collection.toList(modis_collection.size())
        n = modis_collection.size().getInfo()
        listfns = []
        for i in range(n):
            img = ee.Image(images.get(i))
            if img_property:
                img = set_new_id(img, self.country.title(), img_property = img_property, band=band)
            
            img_id = img.get('system:id').getInfo()
            
            output_fn = os.path.join(output_dir, f"{img_id}.tif")
            
            listfns.append(
                download_ind_image(img.select(band), output_fn, feature_geometry, scale=scale))
            
        return listfns
    
    
def set_new_id(image, country, band = 'NDVI', img_property = 'system:index'):
    # system:index as ee.String
    original_id = ee.String(image.get(img_property))
    formatted = original_id.replace('_', '-').replace('_', '-')

    date = ee.Date(formatted)
    doy = date.getRelative('day', 'year').add(1)  # DOY is zero-based → add 1
    new_id = ee.String(country) \
        .cat('_').cat(ee.String(band)).cat('_') \
        .cat(date.get('year')) \
        .cat('_') \
        .cat(doy.format('%03d'))  # pad to 3 digits

    return image.set('system:id', new_id)

def image_masking(image, crop_mask, geometry = None):
    """Calculates and adds an NDVI band to an image."""
    if geometry is not None:
        return image.clip(geometry).updateMask(crop_mask)
    else:
        return image.updateMask(crop_mask)
    
def download_ind_image(image, output_fn, geomety, scale, crs = "EPSG:4326"):
    image = image.reproject(crs=crs, scale=scale)
    url = image.getDownloadURL({
    'scale': scale,
    'region': geomety,
    'format': 'GEO_TIFF',
    })
    
    #fn = os.path.dirname(output_fn)
    response = requests.get(url)
    with open(output_fn, 'wb') as f:
        f.write(response.content)
        
    print(f'Image saved in {output_fn}')
    
    return output_fn