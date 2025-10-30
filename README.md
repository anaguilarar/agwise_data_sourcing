# Ag-Wise Data Sourcing: MODIS/VIIRS Time Series and SoilGrids Downloader 

## Overview

This repository provides a Python-based toolkit for downloading, processing, and visualizing vegetation index (VI) time series data from MODIS and VIIRS satellite imagery using the Google Earth Engine (GEE) platform. It is designed to facilitate the acquisition of analysis-ready data for agricultural applications, such as crop monitoring and yield estimation.

* MODIS/VIIRS: The main functionalities are demonstrated in the `GEEMODIS_data_download.ipynb` Jupyter Notebook.
* SoilGrids: The main functionalities are demonstrated in the `GEESoilGrids_data_download.ipynb` Jupyter Notebook.

## Features

*   **Data Download**: Download VI data (e.g., NDVI) from MODIS and VIIRS products for a specified country and time range.
*   **Data Processing**:
    *   **Gap Filling**: Linear interpolation to fill gaps in the time series caused by cloud cover or other issues.
    *   **Smoothing**: Savitzky-Golay filter to smooth the time series and reduce noise.
*   **Crop Masking**: Apply a crop mask to focus the analysis on agricultural areas.
*   **Administrative Filtering**: Filter data based on administrative boundaries (e.g., country, province, district).
*   **Visualization**: Plot raw and processed time series data, and visualize masked data on an interactive map using `geemap`.

## Repository Structure

```
.
├── GEEMODIS_data_download.ipynb
├── README.md
├── gee_datasets
│   ├── __init__.py
│   ├── gee_data.py
│   └── processing_funs.py
└── utils
    └── plots.py
```

*   `GEEMODIS_data_download.ipynb`: A Jupyter Notebook explaining how to use the tools in this repository.
*   `gee_datasets/gee_data.py`: Contains the core classes (`GEEMODIS`, `GEECropMask`) for interacting with Google Earth Engine and downloading data.
*   `gee_datasets/processing_funs.py`: Includes functions for time series processing, such as gap filling and smoothing.
*   `utils/plots.py`: Provides helper functions for plotting the time series data.

## Usage

The primary way to use this repository is through the `GEEMODIS_data_download.ipynb` notebook.

1.  **Open the notebook**: Open `GEEMODIS_data_download.ipynb` in a Jupyter environment.
2.  **Configure the download**: Modify the `configuration` dictionary at the beginning of the notebook to specify your area of interest, the desired GEE product, and the date range.
3.  **Run the cells**: Execute the cells in the notebook to download, process, and visualize the data.

## Configuration

The `configuration` dictionary in the notebook has the following structure:

```python
configuration = {
    'GENERAL_SETTINGS':{
      'ee_project_name': 'your-ee-project-name'
      },
    'PREPROCESSING':{
        'crop_mask': True,
        'crop_mask_product': 'ESA' # or 'DYNAMICWORLD'
    },
    'DATA_DOWNLOAD':
     {
      'ADM0_NAME': 'Kenya',
      'ADM1_NAME': 'Coast',
      'ADM2_NAME': None,
      'product': 'MOD13Q1', # e.g., 'MYD13Q1', 'MOD13A2', 'VNP13A1'
      'starting_date': '2023-01-01',
      'ending_date': '2023-12-01',
    }
}
```

*   `ee_project_name`: Your Google Earth Engine project name.
*   `crop_mask`: Whether to apply a crop mask.
*   `crop_mask_product`: The crop mask product to use ('ESA' or 'DYNAMICWORLD').
*   `ADM0_NAME`: The country name.
*   `product`: The MODIS/VIIRS product to use.
*   `starting_date` / `ending_date`: The time period for the data query.

## Dependencies

This project requires the following Python libraries:
*   `earthengine-api`
*   `pandas`
*   `matplotlib`
*   `geemap`
*   `jupyter`
