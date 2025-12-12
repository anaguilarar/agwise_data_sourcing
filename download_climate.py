import os
import sys
import ee
import yaml

from gee_datasets.climate import OptimizedClimateDownloader


def initialize_ee(project_id: str):
    """
    Inicializa Google Earth Engine con el proyecto dado.
    """
    ee.Initialize(project=project_id)


def load_config(config_path: str) -> dict:
    """
    Carga el archivo YAML y lo devuelve como diccionario.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"La ruta al archivo de configuración no existe: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def parse_points(points_cfg):
    """
    Convierte la estructura de puntos del YAML a un tuple de tuples:

    YAML:
      points:
        - [lon1, lat1]
        - [lon2, lat2]

    → ((lon1, lat1), (lon2, lat2))
    """
    if points_cfg is None:
        return None

    # Esperamos una lista de listas [lon, lat]
    pts = []
    for lon, lat in points_cfg:
        pts.append((float(lon), float(lat)))

    return tuple(pts)


def main(config_path: str):
    # 1) Cargar configuración
    cfg = load_config(config_path)

    # 2) Inicializar GEE
    ee_project = cfg["GENERAL_SETTINGS"]["ee_project_name"]
    initialize_ee(ee_project)

    # 3) Extraer parámetros de DATA_DOWNLOAD
    data_cfg = cfg["DATA_DOWNLOAD"]

    product = data_cfg["product"]           # "chirps" o "agera5"
    variable = data_cfg["variable"]         # "precip", "tmin", "tmax", etc.
    years = data_cfg["years"]               # [start_year, end_year]
    window_mmdd = data_cfg["window_mmdd"]   # ["MM-DD", "MM-DD"]
    temp_target = data_cfg.get("temp_target", "daily")
    temp_agg = data_cfg.get("temp_agg", "sum")
    scale = data_cfg.get("scale", 5000)
    max_workers = data_cfg.get("max_workers", 5)

    points_cfg = data_cfg.get("points", None)
    points = parse_points(points_cfg)

    if points is None:
        raise ValueError("En modo points es obligatorio definir 'points' en DATA_DOWNLOAD.")

    # 4) Construir carpeta de salida a partir de output_path + product/variable
    output_base = cfg["GENERAL_SETTINGS"]["output_path"]
    # p.ej. runs/chirps_precip_points_daily
    dir_name = f"{product}_{variable}_points_{temp_target}"
    output_dir = os.path.join(output_base, dir_name)

    print("==========================================")
    print("   Parámetros leídos del YAML")
    print("==========================================")
    print(f"Proyecto EE:       {ee_project}")
    print(f"Output dir:        {output_dir}")
    print(f"Producto:          {product}")
    print(f"Variable:          {variable}")
    print(f"Años:              {years}")
    print(f"Ventana (MM-DD):   {window_mmdd}")
    print(f"temp_target:       {temp_target}")
    print(f"temp_agg:          {temp_agg}")
    print(f"scale:             {scale}")
    print(f"max_workers:       {max_workers}")
    print(f"Puntos:            {points}")
    print("==========================================\n")

    # 5) Crear downloader con ese directorio
    down = OptimizedClimateDownloader(output_dir=output_dir)

    # 6) Llamar a download_meteo_optimized usando los parámetros del YAML
    #    (modo POINTS → devuelve un CSV con las series de tiempo)
    out_table = down.download_meteo_optimized(
        product=product,
        variable=variable,
        points=points,
        bbox=None,
        shape=None,
        years=years,
        window_mmdd=window_mmdd,
        temp_target=temp_target,
        temp_agg=temp_agg,
        scale=scale,
        max_workers=max_workers,
    )

    print(f"\n✅ Descarga terminada. Archivo de salida: {out_table}")


if __name__ == "__main__":
    print(
        r"""
        ========================================
        |                                      |
        |     AGWISE DATA SOURCING - POINTS    |
        |         CLIMATE (CHIRPS/AgERA5)      |
        |                                      |
        ========================================
        """
    )

    args = sys.argv[1:]
    config = (
        args[args.index("-config") + 1]
        if "-config" in args and len(args) > args.index("-config") + 1
        else None
    )

    if config is None:
        raise SystemExit("Uso: python run_meteo_points.py -config ruta/al/config.yml")

    main(config)
