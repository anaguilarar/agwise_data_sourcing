import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series_from_ic(ic, band, summarize_fn, region_filter, title=None):
    """
    Plot a time series from an ee.ImageCollection.

    Parameters
    ----------
    ic : ee.ImageCollection
        Image collection (e.g. smoothed SG series).
    band : str
        Band to extract (e.g. 'NDVI').
    summarize_fn : callable
        Function (img, region_filter, band) -> ee.Feature
        Used to reduce each image to a single feature with 'date' + band value.
    region_filter : ee.Feature / ee.Geometry
        Region of interest passed to summarize_fn.
    title : str
        Title for the plot.
    """

    # Apply summarization to each image in the collection
    time_series_features = ic.map(
        lambda image: summarize_fn(image, region_filter, band)
    )

    # Convert to client-side dictionary
    features = time_series_features.getInfo()["features"]

    # Build DataFrame
    df = pd.DataFrame(
        [
            {
                "date": f["properties"]["date"],
                band: f["properties"].get(band)
            }
            for f in features
            if f["properties"].get(band) is not None
        ]
    )

    if df.empty:
        print("⚠️ No data found for band:", band)
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df[band], marker="o", linestyle="-", color="red")
    plt.xlabel("Date")
    plt.ylabel(band)
    plt.title(title or f"{band} Time Series")
    plt.grid(True)
    plt.show()

    return df
