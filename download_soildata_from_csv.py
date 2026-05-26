from gee_datasets.soil import GEESoilGrids

import ee
import os
import sys
import yaml
import pandas as pd
from tqdm import tqdm


def depth_thickness(depth_str):
    parts = depth_str.split('-')
    return int(parts[1]) - int(parts[0])


def weighted_average_0_30(df, soil_properties):
    """Thickness-weighted average for depth layers fully within 0-30 cm."""
    def within_30(depth_str):
        parts = depth_str.split('-')
        return int(parts[1]) <= 30

    df = df[df['depth'].apply(within_30)].copy()
    if df.empty:
        return None

    df['thickness'] = df['depth'].apply(depth_thickness)
    total = df['thickness'].sum()

    result = {'depth': '0-30'}
    for prop in soil_properties:
        if prop in df.columns:
            result[prop] = (df[prop] * df['thickness']).sum() / total
    return result


def download_coordinate(data_downloader, coordinate, soil_properties, depths, scale):
    try:
        df = data_downloader.soildata_using_point(soil_properties, coordinate, depths=depths, scale=scale)
        return df
    except Exception as e:
        print(f'  Error downloading [{coordinate}]: {e}')
        return None


def main(config_path):
    assert os.path.exists(config_path), f'Config not found: {config_path}'
    print(f'-------> Starting: {config_path}')

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    ee.Initialize(project=config_dict['GENERAL_SETTINGS']['ee_project_name'])

    csv_path = config_dict['DATA_DOWNLOAD']['file_with_coordinates']
    assert os.path.exists(csv_path), f'CSV not found: {csv_path}'

    lon_col = config_dict['DATA_DOWNLOAD'].get('lon_column', 'Longitude')
    lat_col = config_dict['DATA_DOWNLOAD'].get('lat_column', 'Latitude')
    soil_properties = config_dict['DATA_DOWNLOAD']['properties']
    depths = config_dict['DATA_DOWNLOAD']['depths']
    scale = config_dict['DATA_DOWNLOAD'].get('scale', 250)
    output_path = config_dict['GENERAL_SETTINGS']['output_path']

    df_input = pd.read_csv(csv_path)
    assert lon_col in df_input.columns, f"Column '{lon_col}' not found in CSV. Available: {df_input.columns.tolist()}"
    assert lat_col in df_input.columns, f"Column '{lat_col}' not found in CSV. Available: {df_input.columns.tolist()}"

    data_downloader = GEESoilGrids(config_dict['DATA_DOWNLOAD'].get('ADM0_NAME', None))

    unique_coords = df_input[[lon_col, lat_col]].drop_duplicates().dropna().reset_index(drop=True)
    print(f'-------> {len(unique_coords)} unique coordinates to process')

    soil_records = []
    for _, row in tqdm(unique_coords.iterrows(), total=len(unique_coords), desc='Soil download'):
        lon, lat = float(row[lon_col]), float(row[lat_col])
        df_soil = download_coordinate(data_downloader, [lon, lat], soil_properties, depths, scale)
        if df_soil is None or df_soil.empty:
            print(f'  Warning: no data for [{lon}, {lat}]')
            continue
        avg = weighted_average_0_30(df_soil, soil_properties)
        if avg is not None:
            avg[lon_col] = lon
            avg[lat_col] = lat
            soil_records.append(avg)

    if not soil_records:
        print('No soil data downloaded.')
        return

    df_soil_avg = pd.DataFrame(soil_records)

    df_output = df_input.merge(df_soil_avg.drop(columns=['depth'], errors='ignore'),
                               on=[lon_col, lat_col], how='left')

    os.makedirs(output_path, exist_ok=True)
    input_stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_path, f'{input_stem}_soil_0_30cm.csv')
    df_output.to_csv(output_file, index=False)
    print(f'-------> Saved to: {output_file}')
    print(df_soil_avg.to_string())


if __name__ == '__main__':
    print('''\

        ============================================
        |                                          |
        |         AGWISE DATA SOURCING             |
        |      SoilGrids from CSV Coordinates      |
        |         0-30 cm Depth Average            |
        ============================================
    ''')

    args = sys.argv[1:]
    config = args[args.index('-config') + 1] if '-config' in args and len(args) > args.index('-config') + 1 else None
    assert config is not None, 'Provide a config file: -config path/to/config.yaml'
    main(config)
