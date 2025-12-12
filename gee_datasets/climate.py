import ee
import requests
import zipfile
import time
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import xarray as xr
import rioxarray as rxr
import pandas as pd


class OptimizedClimateDownloader:
    """
    Optimized downloader for CHIRPS and AgERA5 with dynamic stacks.

    Parameters
    ----------
    output_dir : str
        Base directory where TIFFs, CSVs and logs will be stored.
    """

    # GEE limits
    MAX_SIZE_MB = 32  # Limit per request in GEE
    SAFETY_FACTOR = 0.85  # Use 85% of the limit for safety

    # Dataset configuration (edit AgERA5 IDs / band names to match your assets)
    DATASETS = {
        "chirps": {
            "collection": "UCSB-CHG/CHIRPS/DAILY",
            "bands": {
                "precip": "precipitation",
            },
            "prefix": "chirps",
            "bytes_per_pixel": 4,
            "compression": 0.3,
        },
        "agera5": {
            # Colección AgERA5 v2 diaria en GEE (Climate Engine / community catalog)
            # Ver: https://gee-community-catalog.org/projects/agera5_datasets/
            "collection": "projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",
            "bands": {
                # Temperaturas
                "tmin": "Temperature_Air_2m_Min_24h",
                "tmax": "Temperature_Air_2m_Max_24h",
                "tmean": "Temperature_Air_2m_Mean_24h",

                # Precipitación (por si la quieres usar también desde AgERA5)
                "precip": "Precipitation_Flux",

                # Radiación solar
                "solrad": "Solar_Radiation_Flux",

                # Humedad relativa (puedes elegir cuál usar)
                "rh06": "Relative_Humidity_2m_06h",
                "rh15": "Relative_Humidity_2m_15h",
            },
            "prefix": "agera5",
            "bytes_per_pixel": 4,
            "compression": 0.3,
        },
    }

    def __init__(self, output_dir: str = "./chirps_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Directory for final TIFFs
        self.tiff_dir = self.output_dir / "tiffs"
        self.tiff_dir.mkdir(exist_ok=True)

        # Temporary directory for ZIPs
        self.temp_dir = self.output_dir / "temp_zips"
        self.temp_dir.mkdir(exist_ok=True)

        # Operation log
        self.download_log: list[dict] = []

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _get_dataset_config(self, product: str) -> dict:
        if product not in self.DATASETS:
            raise ValueError(f"Unknown product '{product}'. Available: {list(self.DATASETS.keys())}")
        return self.DATASETS[product]

    def _get_collection_and_band(
        self,
        product: Literal["chirps", "agera5"],
        variable: str,
    ) -> tuple[ee.ImageCollection, str, str]:
        cfg = self._get_dataset_config(product)
        bands = cfg.get("bands", {})
        if variable not in bands:
            raise ValueError(
                f"Unknown variable '{variable}' for product '{product}'. "
                f"Available: {list(bands.keys())}"
            )
        collection_id = cfg["collection"]
        band_name = bands[variable]
        prefix = cfg["prefix"]
        coll = ee.ImageCollection(collection_id)
        return coll, band_name, prefix

    # ------------------------------------------------------------------
    # Geometry & time helpers
    # ------------------------------------------------------------------
    def get_country_geometry(self, country_name):
        """Gets geometry for a country (FAO GAUL level0)."""
        countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
        country = countries.filter(ee.Filter.eq("ADM0_NAME", country_name))
        return country.geometry()
    
    def _convert_units(
        self,
        product: str,
        variable: str,
        img: ee.Image,
    ) -> ee.Image:
        """
        Aplica conversiones de unidades a la imagen original.

        - Temperaturas de AgERA5: Kelvin → Celsius
        - El resto de variables se deja igual.
        """
        # Temperaturas AgERA5 en Kelvin
        if product == "agera5" and variable in ("tmin", "tmax", "tmean"):
            # Kelvin → Celsius
            img_c = img.subtract(273.15)
            # Conserva propiedades
            return img_c.copyProperties(img, img.propertyNames())

        # Otras variables: sin cambios
        return img


    def _build_points_fc(
        self,
        points: tuple[float, float] | tuple[tuple[float, float], ...],
    ) -> ee.FeatureCollection:
        """
        Creates a FeatureCollection of points with properties:
        - point_id
        - lon
        - lat
        """
        first = points[0]
        if isinstance(first, (int, float)):
            pts = [points]  # single point
        else:
            pts = list(points)  # multiple points

        features = []
        for idx, (lon, lat) in enumerate(pts):
            geom = ee.Geometry.Point([lon, lat])
            feat = ee.Feature(
                geom,
                {
                    "point_id": idx,
                    "lon": float(lon),
                    "lat": float(lat),
                },
            )
            features.append(feat)

        return ee.FeatureCollection(features)

    def parse_date_window(self, window_mmdd: List[str]) -> Tuple[str, str]:
        """Converts MM-DD window to a usable format"""
        start_md, end_md = window_mmdd
        return start_md, end_md
    
    def _generate_dates_between(
        self,
        start_date: datetime,
        end_date: datetime,
        temp_target: Literal["daily", "monthly"],
    ) -> list[str]:
        """
        Genera una lista de fechas (string) entre start_date y end_date
        según temp_target.

        - daily  → 'YYYY-MM-DD' para cada día
        - monthly → 'YYYY-MM' para cada mes (primer día de cada mes)
        """
        out: list[str] = []

        if temp_target == "monthly":
            current = start_date.replace(day=1)
            while current <= end_date:
                out.append(current.strftime("%Y-%m"))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:  # "daily"
            current = start_date
            while current <= end_date:
                out.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

        return out
    
    
    def generate_date_list(
        self,
        years: List[int],
        window_mmdd: List[str],
        temp_target: Literal["daily", "monthly"],
    ) -> List[str]:
        """
        Genera la lista de fechas según:
        - years = [start_year, end_year]
        - window_mmdd = ["MM-DD_inicio", "MM-DD_fin"]

        Si la ventana NO cruza de año (p.ej. 03-01 → 05-31):
            Para cada año en [start_year, end_year]:
                [year-MM-DD_inicio, year-MM-DD_fin]

        Si la ventana SÍ cruza de año (p.ej. 10-01 → 01-31):
            Para cada año en [start_year, end_year-1]:
                [year-MM-DD_inicio, (year+1)-MM-DD_fin]
        """
        dates: list[str] = []
        start_md, end_md = self.parse_date_window(window_mmdd)

        start_month, start_day = map(int, start_md.split("-"))
        end_month, end_day = map(int, end_md.split("-"))

        # ¿La ventana cruza de año? (ej.: 10-01 → 01-31)
        crosses_year = (end_month, end_day) < (start_month, start_day)

        if not crosses_year:
            # Ventana dentro del mismo año (ej. 03-01 → 05-31)
            for year in range(years[0], years[1] + 1):
                start_date = datetime(year, start_month, start_day)
                end_date = datetime(year, end_month, end_day)
                dates.extend(
                    self._generate_dates_between(start_date, end_date, temp_target)
                )
        else:
            # Ventana que cruza año (ej. 10-01 → 01-31)
            # Usamos años de inicio: start_year .. end_year-1
            for year in range(years[0], years[1]):
                start_date = datetime(year, start_month, start_day)
                end_date = datetime(year + 1, end_month, end_day)
                dates.extend(
                    self._generate_dates_between(start_date, end_date, temp_target)
                )

        return dates
    
    # ------------------------------------------------------------------
    # Stack helpers
    # ------------------------------------------------------------------
    def estimate_image_size_mb(
        self,
        region: ee.Geometry,
        scale: int = 5000,
        product: Literal["chirps", "agera5"] = "chirps",
        variable: str = "precip",
    ) -> float:
        """
        Estimates the size of a single image in MB for the given product/variable.

        Uses a very simple approximation:
        area [km²] -> pixels -> bytes (bytes_per_pixel × compression_factor).
        """
        # Compute area in km²
        area_km2 = region.area().divide(1e6).getInfo()

        # Compute number of pixels (scale in meters)
        pixel_area_km2 = (scale / 1000) ** 2
        num_pixels = area_km2 / pixel_area_km2

        cfg = self._get_dataset_config(product)
        bytes_per_pixel = cfg.get("bytes_per_pixel", 4)
        compression_factor = cfg.get("compression", 0.3)

        estimated_mb = (num_pixels * bytes_per_pixel * compression_factor) / (1024**2)

        return estimated_mb

    def calculate_optimal_stack_size(
        self,
        single_image_mb: float,
        total_images: int,
    ) -> int:
        """
        Calculates how many images can go into a single stack.

        Returns
        -------
        int
            Optimal number of images per stack.
        """
        max_allowed_mb = self.MAX_SIZE_MB * self.SAFETY_FACTOR

        # Calculate images per stack
        images_per_stack = int(max_allowed_mb / single_image_mb)

        # Minimum 1, maximum 50 (to avoid very large stacks in memory)
        images_per_stack = max(1, min(images_per_stack, 50))

        return images_per_stack

    def create_stack_groups(
        self,
        dates: List[str],
        images_per_stack: int,
    ) -> List[List[str]]:
        """
        Groups dates into stacks.

        Returns
        -------
        list[list[str]]
            List of date groups.
        """
        stacks: list[list[str]] = []
        for i in range(0, len(dates), images_per_stack):
            stack = dates[i : i + images_per_stack]
            stacks.append(stack)

        return stacks

    # ------------------------------------------------------------------
    # Image stack creation (generic: CHIRPS & AgERA5)
    # ------------------------------------------------------------------
    def create_image_stack(
        self,
        product: Literal["chirps", "agera5"],
        variable: str,
        dates: List[str],
        temp_target: Literal["daily", "monthly"],
        temp_agg: Literal["mean", "min", "max", "sum"],
        region: ee.Geometry,
    ) -> ee.Image:
        """
        Creates a multiband stack image for multiple dates and a specific product/variable.

        Returns
        -------
        ee.Image
            One band per date (e.g. chirps_precip_YYYYMMDD).
        """
        coll, band_name, prefix = self._get_collection_and_band(product, variable)

        bands: list[ee.Image] = []

        for date in dates:
            if temp_target == "daily":
                # Specific day
                start = ee.Date(date)
                end = start.advance(1, "day")
                img = coll.filterDate(start, end).first()
                if img is None:
                    raise ValueError(f"No image found for {product}/{variable} on {date}")
                img = img.select(band_name)
                img = self._convert_units(product, variable, img)

                band_id = f"{prefix}_{variable}_{date.replace('-', '')}"

            else:  # monthly
                # Aggregate whole month
                parts = date.split("-")
                year, month = int(parts[0]), int(parts[1])

                month_start = ee.Date.fromYMD(year, month, 1)
                month_end = month_start.advance(1, "month")

                monthly_data = coll.filterDate(month_start, month_end).select(band_name)

                if temp_agg == "mean":
                    img = monthly_data.mean()
                elif temp_agg == "sum":
                    img = monthly_data.sum()
                elif temp_agg == "min":
                    img = monthly_data.min()
                elif temp_agg == "max":
                    img = monthly_data.max()
                else:
                    raise ValueError(f"Unknown temp_agg: {temp_agg}")
                img = self._convert_units(product, variable, img)
                band_id = f"{prefix}_{variable}_{date.replace('-', '')}"

            bands.append(img.rename(band_id))

        if not bands:
            raise ValueError("No images found to build the stack.")

        # Combine into a multiband stack
        if len(bands) == 1:
            stack = bands[0]
        else:
            stack = bands[0]
            for band in bands[1:]:
                stack = stack.addBands(band)

        return stack.clip(region)

    # ------------------------------------------------------------------
    # Download helpers (stacks)
    # ------------------------------------------------------------------
    def download_stack(
        self,
        stack: ee.Image,
        stack_id: str,
        region: ee.Geometry,
        scale: int = 5000,
        max_retries: int = 3,
    ) -> Tuple[bool, str, float, str]:
        """
        Downloads a stack (can be ZIP or direct TIFF).

        Returns
        -------
        (success, message, elapsed_time, file_type)
        """
        start_time = time.time()
        temp_path = self.temp_dir / f"{stack_id}.tmp"

        for attempt in range(max_retries):
            try:
                # Generate download URL
                url = stack.getDownloadURL(
                    {
                        "region": region,
                        "scale": scale,
                        "format": "ZIPPED_GEO_TIFF",
                        "crs": "EPSG:4326",
                    }
                )

                # Download
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()

                # Save temporarily
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Detect file type
                with open(temp_path, "rb") as f:
                    magic = f.read(4)

                # ZIP: starts with 'PK\x03\x04'
                # TIFF: starts with 'II*\x00' (little-endian) or 'MM\x00*' (big-endian)
                if magic[:2] == b"PK":
                    file_type = "zip"
                    final_path = self.temp_dir / f"{stack_id}.zip"
                elif magic[:2] in (b"II", b"MM"):
                    file_type = "tiff"
                    final_path = self.temp_dir / f"{stack_id}.tif"
                else:
                    raise ValueError(f"Unknown format: {magic[:4]}")

                # Rename with correct extension
                temp_path.rename(final_path)

                elapsed = time.time() - start_time
                return True, f"✓ Stack {stack_id}", elapsed, file_type

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    elapsed = time.time() - start_time
                    return False, f"✗ Stack {stack_id}: {str(e)[:50]}", elapsed, "unknown"

        return False, f"✗ Stack {stack_id}: Max retries", time.time() - start_time, "unknown"

    def extract_stack(self, file_path: Path) -> List[Path]:
        """
        Extracts TIFF(s) from the ZIP or returns the TIFF as is.

        Does NOT split by bands.

        If it is a ZIP:
            - Extracts .tif/.tiff
            - Renames them using the ZIP name:
                zip_name.zip -> zip_name.tif      (if there is 1 tiff)
                zip_name.zip -> zip_name_1.tif,
                                zip_name_2.tif... (if there are several tiffs)
        """
        extracted_files: list[Path] = []

        try:
            # ZIP case
            if file_path.suffix.lower() == ".zip":
                zip_stem = file_path.stem  # base name of the zip, without extension

                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    tif_files = [
                        f
                        for f in zip_ref.namelist()
                        if f.lower().endswith((".tif", ".tiff"))
                    ]

                    for idx, tif_file in enumerate(tif_files, start=1):
                        # Extract to tiff_dir
                        zip_ref.extract(tif_file, self.tiff_dir)
                        extracted = self.tiff_dir / tif_file

                        # If it came inside a subfolder, move it to tiff_dir
                        if extracted.parent != self.tiff_dir:
                            flat_path = self.tiff_dir / extracted.name
                            extracted.rename(flat_path)
                            extracted = flat_path

                        # New name based on ZIP
                        if len(tif_files) == 1:
                            new_name = f"{zip_stem}.tif"
                        else:
                            new_name = f"{zip_stem}_{idx}.tif"

                        final_path = self.tiff_dir / new_name

                        # Overwrite if exists
                        if final_path.exists():
                            final_path.unlink()

                        extracted.rename(final_path)
                        extracted_files.append(final_path)

                # Delete original ZIP
                file_path.unlink()

            # Direct TIFF case
            elif file_path.suffix.lower() in (".tif", ".tiff"):
                extracted_files.append(file_path)

        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")

        return extracted_files

    # ------------------------------------------------------------------
    # Points mode (time series) – generic
    # ------------------------------------------------------------------
    def download_points_timeseries(
        self,
        product: Literal["chirps", "agera5"],
        variable: str,
        points: tuple[float, float] | tuple[tuple[float, float], ...],
        years: List[int],
        window_mmdd: List[str],
        temp_target: Literal["daily", "monthly"] = "daily",
        temp_agg: Literal["mean", "min", "max", "sum"] = "sum",
        scale: int = 5000,
        file_name: Optional[str] = None,
    ) -> Path:
        """
        Downloads a time series table for one or several points.

        Row = point × date (day or month).

        Columns:
            point_id, lon, lat, date, <variable>
        """
        # 1) Points FeatureCollection
        points_fc = self._build_points_fc(points)

        # 2) Date list
        dates = self.generate_date_list(years, window_mmdd, temp_target)

        coll, band_name, prefix = self._get_collection_and_band(product, variable)

        if file_name is None:
            file_name = f"{prefix}_{variable}_points_{temp_target}_{years[0]}_{years[1]}"

        # =======================
        # DAILY CASE (row = point × day)
        # =======================
        if temp_target == "daily":
            start_date = dates[0]
            end_date_dt = datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=1)
            end_date = end_date_dt.strftime("%Y-%m-%d")

            daily_coll = coll.filterDate(start_date, end_date)

            def sample_daily(img):
                img_var = img.select(band_name)
                img_var = self._convert_units(product, variable, img_var)
                img_var = ee.Image(img_var).rename(variable) 
                date_str = img.date().format("YYYY-MM-dd")
                samples = img_var.sampleRegions(
                    collection=points_fc,
                    scale=scale,
                    geometries=True,
                )
                return samples.map(lambda f: f.set("date", date_str))

            fc_per_image = daily_coll.map(sample_daily)
            all_samples = ee.FeatureCollection(fc_per_image).flatten()

        # =======================
        # MONTHLY CASE (row = point × month)
        # =======================
        else:  # "monthly"
            monthly_images: list[ee.Image] = []

            for date_str in dates:  # each element is "YYYY-MM"
                parts = date_str.split("-")
                year, month = int(parts[0]), int(parts[1])

                month_start = ee.Date.fromYMD(year, month, 1)
                month_end = month_start.advance(1, "month")

                monthly_data = coll.filterDate(month_start, month_end).select(band_name)

                if temp_agg == "mean":
                    img = monthly_data.mean()
                elif temp_agg == "sum":
                    img = monthly_data.sum()
                elif temp_agg == "min":
                    img = monthly_data.min()
                elif temp_agg == "max":
                    img = monthly_data.max()
                else:
                    raise ValueError(f"Unknown temp_agg: {temp_agg}")

                img = img.select(0).rename(variable).set("date_str", date_str)
                img = self._convert_units(product, variable, img)
                monthly_images.append(img)

            monthly_coll = ee.ImageCollection(monthly_images)

            def sample_monthly(img):
                date_str = ee.String(img.get("date_str"))
                samples = img.sampleRegions(
                    collection=points_fc,
                    scale=scale,
                    geometries=True,
                )
                return samples.map(lambda f: f.set("date", date_str))

            fc_per_image = monthly_coll.map(sample_monthly)
            all_samples = ee.FeatureCollection(fc_per_image).flatten()

        # =======================
        # EXPORT TABLE (CSV)
        # =======================
        selectors = ["point_id", "lon", "lat", "date", variable]

        url = all_samples.getDownloadURL(
            filetype="CSV",
            selectors=selectors,
            filename=file_name,
        )

        out_path = self.output_dir / f"{file_name}.csv"

        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Point time series ({product}/{variable}, {temp_target}) saved to: {out_path}")
        return out_path
    
    def _build_netcdf_from_tiffs(
        self,
        product: str,
        variable: str,
        temp_target: str,
        stacks: list[list[str]],
    ):
        """
        Lee todos los GeoTIFFs en self.tiff_dir (stacks) y construye
        un único NetCDF con dims (time, y, x) y una sola variable: `variable`.

        Devuelve:
            Path al NetCDF creado, o None si no se pudo construir.
        """
        tiff_files = sorted(self.tiff_dir.glob("*.tif"))

        if not tiff_files:
            print("⚠ No hay archivos TIFF para construir NetCDF; se omite.")
            return None

        if len(tiff_files) != len(stacks):
            print(
                f"⚠ No construyo NetCDF: nº de TIFFs ({len(tiff_files)}) "
                f"≠ nº de stacks ({len(stacks)})."
            )
            return None

        all_dataarrays = []
        all_times = []

        print("\n📦 Construyendo NetCDF a partir de los TIFF descargados...")

        for tif_path, stack_dates in zip(tiff_files, stacks):
            da = rxr.open_rasterio(tif_path)

            if "band" not in da.dims:
                da = da.expand_dims("band")

            n_bands = da.sizes["band"]
            if n_bands != len(stack_dates):
                print(
                    f"  ⚠ Mismatch en {tif_path.name}: "
                    f"{n_bands} bandas vs {len(stack_dates)} fechas. Lo salto."
                )
                continue

            da = da.rename({"band": "time"})
            da = da.assign_coords(time=np.array(stack_dates, dtype="datetime64[ns]"))
            da.name = variable

            all_dataarrays.append(da)
            all_times.extend(stack_dates)

        if not all_dataarrays:
            print("⚠ No se pudo construir NetCDF (ningún TIFF válido).")
            return None

        big_da = xr.concat(all_dataarrays, dim="time")
        big_da = big_da.sortby("time")
        ds = big_da.to_dataset(name=variable)

        nc_path = self.output_dir / f"{product}_{variable}_{temp_target}.nc"
        ds.to_netcdf(nc_path)

        print(f"✅ NetCDF guardado en: {nc_path}")
        return nc_path
    
    
    def _build_grid_table_from_netcdf(
    self,
    nc_path: Path,
    variable: str,
    temp_target: str,
) -> Path | None:
        """
        Convierte el NetCDF espacial (time, y, x) en una tabla larga estilo 'points':

        columnas: point_id, lon, lat, date, <variable>

        Guarda el CSV en la misma carpeta que el NetCDF y devuelve su Path.
        """
        if not nc_path.exists():
            print(f"⚠ NetCDF no encontrado: {nc_path}")
            return None

        ds = xr.open_dataset(nc_path)

        if variable not in ds.data_vars:
            print(f"⚠ Variable '{variable}' no está en el NetCDF.")
            return None

        da = ds[variable]

        # da.to_dataframe → índice con [time, y, x] y columna <variable>
        df = da.to_dataframe(name=variable).reset_index()

        # Renombrar coordenadas espaciales a lon/lat
        rename_map = {}
        if "x" in df.columns:
            rename_map["x"] = "lon"
        if "y" in df.columns:
            rename_map["y"] = "lat"
        df = df.rename(columns=rename_map)

        if "lon" not in df.columns or "lat" not in df.columns:
            print("⚠ No se encontraron columnas 'lon' y 'lat' en el NetCDF.")
            return None

        # Formatear fecha
        if temp_target == "monthly":
            df["date"] = df["time"].dt.strftime("%Y-%m")
        else:  # daily
            df["date"] = df["time"].dt.strftime("%Y-%m-%d")

        # Crear point_id único por lon/lat
        points_df = (
            df[["lon", "lat"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        points_df["point_id"] = np.arange(len(points_df), dtype=int)

        # Merge para añadir point_id
        df = df.merge(points_df, on=["lon", "lat"], how="left")

        # Orden y columnas para que coincida con tu estructura de points
        df = df[["point_id", "lon", "lat", "date", variable]].sort_values(
            ["point_id", "date"]
        ).reset_index(drop=True)

        # Guardar CSV: mismo nombre base que el NetCDF + sufijo _table
        csv_path = nc_path.with_name(nc_path.stem + "_table.csv")
        df.to_csv(csv_path, index=False)

        print(f"✅ Tabla gridded guardada en: {csv_path}")
        return csv_path




    # ------------------------------------------------------------------
    # MAIN GENERIC METHOD (CHIRPS & AgERA5)
    # ------------------------------------------------------------------
    def download_meteo_optimized(
        self,
        product: Literal["chirps", "agera5"],
        variable: str,
        points: tuple[float, float] | tuple[tuple[float, float], ...] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        shape: Optional[str] = None,
        years: List[int] | None = None,
        window_mmdd: List[str] | None = None,
        temp_target: Literal["daily", "monthly"] = "daily",
        temp_agg: Literal["mean", "min", "max", "sum"] = "sum",
        scale: int = 5000,
        max_workers: int = 5,
    ):
        """
        Optimized download with dynamic stacks and parallelism.

        Parameters
        ----------
        product : {"chirps", "agera5"}
            Dataset to download.
        variable : str
            Variable within the product (e.g. "precip", "tmin", "tmax", "rh", "solrad").
        points : tuple or list of tuples, optional
            If provided, runs in POINT MODE and returns a CSV time series.
        bbox : (min_lon, min_lat, max_lon, max_lat), optional
            Rectangle region for raster mode.
        shape : str, optional
            Country name (GAUL level0) to use as region in raster mode.
        years : [start_year, end_year]
        window_mmdd : ["MM-DD", "MM-DD"]
        temp_target : {"daily", "monthly"}
        temp_agg : {"mean", "sum", "min", "max"}
        scale : int
            Spatial resolution in meters.
        max_workers : int
            Parallel downloads (recommended: 2–4).
        """
        if years is None or window_mmdd is None:
            raise ValueError("You must provide 'years' and 'window_mmdd'.")

        # 1) POINT MODE → long table
        if points is not None:
            print("\n" + "=" * 70)
            print(f"📍 POINT MODE: {product}/{variable} – downloading point time series as long table")
            print("=" * 70)

            out_table = self.download_points_timeseries(
                product=product,
                variable=variable,
                points=points,
                years=years,
                window_mmdd=window_mmdd,
                temp_target=temp_target,
                temp_agg=temp_agg,
                scale=scale,
                file_name=None,
            )
            return out_table  # Exit here, no stacks for points

        # 2) RASTER MODE (bbox or shape)
        flags = [points is not None, bbox is not None, shape is not None]
        if sum(flags) == 0:
            raise ValueError("You must specify one of: points, bbox or shape.")
        if sum(flags) > 1:
            raise ValueError("You cannot combine points, bbox and shape at the same time.")

        # --- Build region (ee.Geometry) ---
        if bbox is not None:
            # Rectangle: (min_lon, min_lat, max_lon, max_lat)
            region = ee.Geometry.Rectangle(bbox)
        elif shape is not None:
            # Assume 'shape' is a country name; reuse helper
            region = self.get_country_geometry(shape)
        else:
            raise ValueError("Either bbox or shape must be provided for raster mode.")

        print("\n" + "=" * 70)
        print(f"🚀 OPTIMIZED DOWNLOAD WITH STACKS – {product}/{variable}")
        print("=" * 70)

        # 1. Generate date list
        dates = self.generate_date_list(years, window_mmdd, temp_target)
        total_images = len(dates)
        print(f"\n📅 Total images: {total_images}")
        print(f"   Range: {dates[0]} → {dates[-1]}")

        # 2. Estimate size per image
        print(f"\n📊 Estimating download capacity...")
        single_image_mb = self.estimate_image_size_mb(region, scale, product=product, variable=variable)
        print(f"   Estimated size per image: {single_image_mb:.2f} MB")

        # 3. Compute optimal stacks
        images_per_stack = self.calculate_optimal_stack_size(single_image_mb, total_images)
        stacks = self.create_stack_groups(dates, images_per_stack)

        print(f"\n📦 Stack configuration:")
        print(f"   Images per stack: {images_per_stack}")
        print(f"   Estimated size per stack: {single_image_mb * images_per_stack:.2f} MB")
        print(f"   Total stacks: {len(stacks)}")
        print(f"   Parallel workers: {max_workers}")

        # Temporal summary
        if images_per_stack == 1:
            periodo = "1 image"
        elif temp_target == "daily":
            dias = images_per_stack
            if dias >= 365:
                periodo = f"~{dias // 365} years"
            elif dias >= 30:
                periodo = f"~{dias // 30} months"
            else:
                periodo = f"{dias} days"
        else:  # monthly
            meses = images_per_stack
            if meses >= 12:
                periodo = f"~{meses // 12} years"
            else:
                periodo = f"{meses} months"

        print(f"   Period per stack: {periodo}")

        # Overall size
        total_size_gb = (single_image_mb * total_images) / 1024
        print(f"\n💾 Total estimated size: {total_size_gb:.2f} GB")

        input("\n⏸️  Press ENTER to start download...")

        # 4. Prepare download tasks
        download_tasks = []
        for i, stack_dates in enumerate(stacks):
            stack_id = (
                f"{product}_{variable}_"
                f"stack_{i + 1:04d}_"
                f"{stack_dates[0].replace('-', '')}_to_"
                f"{stack_dates[-1].replace('-', '')}"
            )

            download_tasks.append(
                {
                    "stack_id": stack_id,
                    "dates": stack_dates,
                    "index": i + 1,
                    "total": len(stacks),
                }
            )

        # 5. Parallel download
        print(f"\n{'=' * 70}")
        print("⬇️  STARTING DOWNLOAD")
        print("=" * 70)

        start_time = time.time()
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create stacks and download
            futures = {}

            for task in download_tasks:
                # Create stack
                stack = self.create_image_stack(
                    product,
                    variable,
                    task["dates"],
                    temp_target,
                    temp_agg,
                    region,
                )

                # Submit download
                future = executor.submit(
                    self.download_stack,
                    stack,
                    task["stack_id"],
                    region,
                    scale,
                )
                futures[future] = task

            # Process results
            for future in as_completed(futures):
                task = futures[future]
                success, message, elapsed, file_type = future.result()

                # Progress display
                if success:
                    successful += 1
                    print(f"{message} ({task['index']}/{task['total']}) - {elapsed:.1f}s")
                else:
                    failed += 1
                    print(f"{message} ({task['index']}/{task['total']})")

                # Log
                self.download_log.append(
                    {
                        "stack_id": task["stack_id"],
                        "product": product,
                        "variable": variable,
                        "success": success,
                        "elapsed": elapsed,
                        "num_images": len(task["dates"]),
                    }
                )

        total_download_time = time.time() - start_time

        # 6. Extract all ZIPs
        print(f"\n{'=' * 70}")
        print("📂 EXTRACTING FILES")
        print("=" * 70)

        zip_files = list(self.temp_dir.glob("*.zip"))
        total_extracted = 0

        for zip_file in zip_files:
            extracted = self.extract_stack(zip_file)
            total_extracted += len(extracted)
            print(f"  ✓ {zip_file.name} → {len(extracted)} TIFFs")

        # Clean up temp directory (if empty)
        try:
            if self.temp_dir.exists() and not list(self.temp_dir.glob("*")):
                self.temp_dir.rmdir()
        except Exception:
            pass

        # 7. Final summary
        print(f"\n{'=' * 70}")
        print("✅ DOWNLOAD COMPLETED")
        print("=" * 70)
        print(f"Product / variable: {product} / {variable}")
        print(f"Successful stacks:  {successful}/{len(stacks)}")
        print(f"Failed stacks:      {failed}/{len(stacks)}")
        print(f"Extracted TIFFs:    {total_extracted}")
        print(f"Total time:         {total_download_time / 60:.1f} minutes")
        print(f"Speed:              {total_images / total_download_time:.1f} imgs/min")
        print(f"\n📁 Files in: {self.tiff_dir}")
        print("=" * 70)
    
        # Save log
        log_path = self.output_dir / "download_log.json"
        
        with open(log_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "product": product,
                        "variable": variable,
                        "total_images": total_images,
                        "total_stacks": len(stacks),
                        "images_per_stack": images_per_stack,
                        "successful_stacks": successful,
                        "failed_stacks": failed,
                        "total_time_min": total_download_time / 60,
                        "images_per_min": total_images / total_download_time,
                    },
                    "stacks": self.download_log,
                },
                f,
                indent=2,
            )

        print(f"📊 Log saved to: {log_path}\n")
        
         # 8. Construir NetCDF (sólo en modo raster; en modo points ya hicimos return)
        nc_path = None
        try:
            nc_path = self._build_netcdf_from_tiffs(
                product=product,
                variable=variable,
                temp_target=temp_target,
                stacks=stacks,
            )
        except Exception as e:
            print(f"⚠ Error al construir NetCDF: {e}")

        # 9. Si se creó el NetCDF, construir tabla gridded estilo 'points'
        if nc_path is not None:
            try:
                self._build_grid_table_from_netcdf(
                    nc_path=nc_path,
                    variable=variable,
                    temp_target=temp_target,
                )
            except Exception as e:
                print(f"⚠ Error al construir tabla desde NetCDF: {e}")

    # ------------------------------------------------------------------
    # BACKWARD-COMPATIBLE WRAPPER FOR OLD API
    # ------------------------------------------------------------------
    def download_chirps_optimized(
        self,
        points: tuple[float, float] | tuple[tuple[float, float], ...] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        shape: Optional[str] = None,
        years: List[int] | None = None,
        window_mmdd: List[str] | None = None,
        temp_target: Literal["daily", "monthly"] = "daily",
        temp_agg: Literal["mean", "min", "max", "sum"] = "sum",
        scale: int = 5000,
        max_workers: int = 5,
    ):
        """
        Backward-compatible wrapper that preserves the previous CHIRPS-specific API.

        This simply calls `download_meteo_optimized` with:
            product = "chirps"
            variable = "precip"
        """
        return self.download_meteo_optimized(
            product="chirps",
            variable="precip",
            points=points,
            bbox=bbox,
            shape=shape,
            years=years,
            window_mmdd=window_mmdd,
            temp_target=temp_target,
            temp_agg=temp_agg,
            scale=scale,
            max_workers=max_workers,
        )
