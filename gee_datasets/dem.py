import os
from typing import Dict, List, Optional, Tuple

import ee
import pandas as pd
import requests

from .gee_data import GEEDataDownloader

class GEEdem(GEEDataDownloader):
    """
    Google Earth Engine (GEE) downloader for DEM-based terrain products.
    
    Parameters
    ----------
    country : str
        Name of the country to extract administrative boundaries from.
    
    Attributes
    ----------
    country : str
        Lowercase country name.
    country_filter : ee.FeatureCollection
        FeatureCollection containing the ADM0 boundary for the country.
    query : ee.Image
        DEM image containing elevation, slope, aspect, and TPI bands.
    _adm_filter : ee.FeatureCollection or None
        Optional administrative area subset for export operations.
    """
    
    def __init__(self, country) -> None:
    
        self.country = country.lower()
        self._adm_filter = None
        self._global_adiminstrative_data = 'WM/geoLab/geoBoundaries/600/{adm_level}'

        
        dataset = ee.FeatureCollection(self._global_adiminstrative_data.format(adm_level='ADM0'))
        self.country_filter = dataset.filter(ee.Filter.eq('shapeName', self.country.title()))

        count = self.country_filter.size().getInfo()
        assert count == 1, f'No info for {self.country.title()}'

    
    @staticmethod  
    def extract_data_using_coordinate(
        image: ee.Image,
        point_coordinate: Tuple[float, float],
        scale: int = 250
        ) -> pd.DataFrame:
        """
        Extract DEM-derived values at a specific coordinate.

        Parameters
        ----------
        image : ee.Image
            Image containing DEM values.
        point_coordinate : tuple(float, float)
            (longitude, latitude) coordinate pair.
        scale : int, optional
            Sampling resolution in meters. Default is 250.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing band name, value, and point geometry.
        """
        ee_point = ee.Geometry.Point(point_coordinate)  
        sample = image.sample(region=ee_point, scale=scale).first().getInfo()
        samplesdf = []
        for k, v in sample['properties'].items():
            df = pd.DataFrame({ 'band': [k], 'value':[v], 'x': [point_coordinate[0]], 'y': [point_coordinate[1]]})
            samplesdf.append(df)
            
        return pd.concat(samplesdf)

    @property
    def list_of_products(self):
        """
        Available DEM products.

        Returns
        -------
        dict
            Dictionary mapping product keys to their GEE asset IDs.
        """
        return {
            'nasadem': "NASA/NASADEM_HGT/001",
            'cgiardem': "CGIAR/SRTM90_V4"
        }
        
    def initialize_query(self, dem_product: str = "nasadem") -> ee.Image:
        """
        Initialize DEM query and compute derived terrain metrics.

        Parameters
        ----------
        dem_product : str, optional
            One of the available DEM products. Default is "nasadem".

        Returns
        -------
        ee.Image
            Elevation image with added slope, aspect, and TPI bands.

        Notes
        -----
        - TPI is computed as elevation minus focal mean (5-pixel square kernel).
        - Image is clipped to the ADM0 boundary of the selected country.
        """
        if dem_product not in self.list_of_products:
            raise ValueError(
                f"DEM product '{dem_product}' not recognized. "
                f"Choose from {list(self.list_of_products.keys())}."
            )
            
        self.query = ee.Image(self.list_of_products[dem_product]).select('elevation')

        self.query = self.query.clip(self.country_filter)
        slope = ee.Terrain.slope(self.query)
        aspect = ee.Terrain.aspect(self.query)
        tpi = self.query.select('elevation').subtract(self.query.select('elevation').focalMean(5, 'square'))
        tpi = tpi.rename('tpi')
        
        self.query = self.query.addBands(slope).addBands(aspect).addBands(tpi)
        
        return self.query

    def download_data(self,
        raster_data: ee.Image,
        output_fn: str,
        scale: int = 250
    ) -> None:
        """
        Download DEM raster data as a GeoTIFF.

        Parameters
        ----------
        raster_data : ee.Image
            The image to export.
        output_fn : str
            File path where the downloaded GeoTIFF will be saved.
        scale : int, optional
            Spatial resolution in meters. Default is 250.

        Raises
        ------
        ValueError
            If `_adm_filter` has not been initialized.
        """
        
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
        
