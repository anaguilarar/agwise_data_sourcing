# Ag-Wise Data Sourcing: MODIS/VIIRS Time Series, SoilGrids, and Digital Elevation Model (DEM) Downloader

## Overview

This repository provides a Python-based toolkit for downloading, processing, and visualizing vegetation index (VI) time series from MODIS and VIIRS satellite imagery, along with soil property data from SoilGrids and terrain attributes derived from a digital elevation model (DEM). All data acquisition is performed using the Google Earth Engine (GEE) platform.

The tools are intended for researchers who require analysis-ready geospatial datasets for applications such as crop monitoring and crop modeling.

### Repository Structure

```
.
├── GEEMODIS_data_download.ipynb
├── GEESoilGrids_data_download.ipynb
├── GEEElevation_data_download.ipynb
├── README.md
├── gee_datasets
│   ├── __init__.py
│   ├── gee_data.py
│   ├── dem.py
│   ├── soil.py
│   └── processing_funs.py
└── utils
    └── plots.py
```

*   `GEEMODIS_data_download.ipynb`: A Jupyter Notebook explaining how to use the tools for downloading MODIS/VIIRS data.
*   `GEESoilGrids_data_download.ipynb`: A Jupyter Notebook explaining how to use the tools for downloading SoilGrids data.
*   `GEEElevation_data_download.ipynb`: A Jupyter Notebook explaining how to use the tools for downloading digital elevation data.
*   `gee_datasets/gee_data.py`: Contains the core classes (`GEEDataDownloader`) for interacting with Google Earth Engine and downloading data.
*   `gee_datasets/dem.py`:  DEM-specific tools for extracting elevation, slope, aspect, and other terrain metrics.
*   `gee_datasets/soil.py`:  SoilGrids download tools and utilities.
*   `gee_datasets/processing_funs.py`: Includes functions for time series processing, such as gap filling and smoothing.
*   `utils/plots.py`: Provides helper functions for plotting the time series data.


## Features

*   **MODIS/VIIRS Data Download**: Download VI data (e.g., NDVI) from MODIS and VIIRS products for a specified country and time range.
*   *   **Data Processing**:
        *   **Gap Filling**: Linear interpolation to fill gaps in the time series caused by cloud cover or other issues.
        *   **Smoothing**: Savitzky-Golay filter to smooth the time series and reduce noise.
*   *   **Crop Masking**: Apply a crop mask to focus the analysis on agricultural areas.

*   **SoilGrids Data Download**: Download soil property data (e.g., sand content, pH) for a specified country, administrative level, or specific coordinate.
*   *   **Datacube Creation**: Create a NetCDF datacube from multiple downloaded soil properties.
*   *   **Convert to DSSAT format type file**: Create a file that can be read using DSSAT process base model.

*   **Digital Elevation Model (DEM) and Terrain Derivatives**: Download terrain variables derived from DEMs, such as:
    *   Elevation
    *   Slope
    *   Aspect

*   **Visualization**: Plot raw and processed time series data, and visualize masked data on an interactive map using `geemap`.


## Dependencies

This project requires the following Python libraries:
*   `earthengine-api`
*   `pandas`
*   `matplotlib`
*   `geemap`
*   `jupyter`
