import ee
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any



class GEEDataDownloader(ABC):
  """
  Abstract base class for defining Google Earth Engine (GEE)
  data downloaders.

  Notes
  -----
  Any subclass must implement the following abstract methods:
  - `list_of_products`
  - `initialize_query`
  - `download_data`
  """
  @property  
  @abstractmethod
  def list_of_products(self):
    """
    Dict[str, str]: Dictionary mapping product identifiers to
    their corresponding GEE dataset names.
    """
    return

  @abstractmethod
  def initialize_query(self, starting_date: Optional[str] = None,
                        ending_date: Optional[str] = None) -> Any:
    """
    Initialize the query for the given date range.

    Parameters
    ----------
    starting_date : str or None, optional
        Start date in `YYYY-MM-DD` format.
    ending_date : str or None, optional
        End date in `YYYY-MM-DD` format.

    Returns
    -------
    object
        GEE ImageCollection object resulting from the query.
    """
    pass

  @abstractmethod
  def download_data(self, **kwargs) -> Any:
    """
    Download the processed data.

    Parameters
    ----------
    **kwargs : dict
        Custom arguments required by the specific implementation.

    Returns
    -------
    Any
        Result of the download procedure.
    """
    pass

class GEECropMask(GEEDataDownloader):
  """
  Class for downloading crop mask datasets from Google Earth Engine.

  Parameters
  ----------
  country : str
      Name of the country for filtering the administrative boundary.
  product : str
      Key identifying the dataset to use. Must be one of
      `list_of_products`.

  Attributes
  ----------
  country : str
      Lowercase country name.
  product : str
      Selected crop mask product.
  country_filter : ee.FeatureCollection
      Filtered administrative boundary used for spatial filtering.
  query : ee.ImageCollection
      Query object initialized via `initialize_query`.
  """
    
  @property
  def list_of_products(self):
    """
    Available crop mask products.

    Returns
    -------
    dict
        Mapping between product name and GEE dataset ID.
    """
    return {
        'ESA': 'ESA/WorldCover/v200',
        'DYNAMICWORLD': 'GOOGLE/DYNAMICWORLD/V1'
    }

  @property
  def crop_mask_value(self) -> Dict[str, int]:
    """
    Value representing cropland for each dataset.

    Returns
    -------
    dict
        Mapping of product name to integer class value representing crops.
    """
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


  def initialize_query(self, year: Optional[int] = None) -> ee.ImageCollection:
    """
    Initialize the crop mask query for the selected product.

    Parameters
    ----------
    year : int or None, optional
        Year to filter DynamicWorld data (required for this product).
        If None, the current year is used.

    Returns
    -------
    ee.ImageCollection
        GEE ImageCollection object filtered by the country boundary.

    Notes
    -----
    - DynamicWorld requires filtering by year.
    - ESA WorldCover does not require date filtering.
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

    """
    Placeholder method for downloading processed crop mask data.

    Parameters
    ----------
    **kwargs : dict
        Arguments specifying export parameters (scale, region, file formats, etc.)

    Raises
    ------
    NotImplementedError
        If the method is not yet implemented.
    """
        
    pass
