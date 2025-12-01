import os
from typing import List

import ee
import pandas as pd
import requests

from .gee_data import GEEDataDownloader

class GEEdem(GEEDataDownloader):
    def __init__(self, country) -> None:
    
        self.country = country.lower()
        self._adm_filter = None
        self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

        
        dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
        self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

        count = self.country_filter.size().getInfo()
        assert count == 1, f'No info for {self.country.title()}'

    
    @staticmethod  
    def extract_data_using_coordinate(image, point_coordinate, scale = 250):
        ee_point = ee.Geometry.Point(point_coordinate)  
        sample = image.sample(region=ee_point, scale=scale).first().getInfo()
        samplesdf = []
        for k, v in sample['properties'].items():
            df = pd.DataFrame({ 'band': [k], 'value':[v], 'x': [point_coordinate[0]], 'y': [point_coordinate[1]]})
            samplesdf.append(df)
            
        return pd.concat(samplesdf)

    @property
    def list_of_products(self):
        return {
            'nasadem': "NASA/NASADEM_HGT/001",
            'cgiardem': "CGIAR/SRTM90_V4"
        }
        
    def initialize_query(self, dem_product: str = 'nasadem'):
        """
        function to inisitalize the query
        """
            
        self.query = ee.Image(self.list_of_products[dem_product]).select('elevation')

        self.query = self.query.clip(self.country_filter)
        slope = ee.Terrain.slope(self.query)
        aspect = ee.Terrain.aspect(self.query)
        
        self.query = self.query.addBands(slope).addBands(aspect)
        
        return self.query

    def download_data(self, raster_data, output_fn,  scale = 250):
        raster_data = raster_data.reproject(crs="EPSG:4326", scale=scale)
        url = raster_data.getDownloadURL({
        'scale': scale,
        'region': self._adm_filter.geometry(),
        'format': 'GEO_TIFF',
        })
        
        #fn = os.path.dirname(output_fn)
        response = requests.get(url)
        with open(output_fn, 'wb') as f:
            f.write(response.content)
    
    def get_adm_level_data(self, adm_level = None, feature_name = None):
        if adm_level is not None and feature_name is not None:
            dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level=adm_level))
            adm_filter = dataset.filter(ee.Filter.eq('shapeName', feature_name.lower().title()))
            print(f'data will be processed for: {feature_name}')
            self._adm_filter = adm_filter
            return self.query.clip(adm_filter)
        else:
            return self.query

    def terraindata_using_point(self, point_coordinate, dem_product: str = 'nasadem', scale = 30):
        
        image = self.initialize_query(dem_product)
        
        return self.extract_data_using_coordinate(image, point_coordinate, scale = scale)
        
