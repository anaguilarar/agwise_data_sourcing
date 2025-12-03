import ee

def moving_average(collection, band, window=3):
    # Must be odd
    half = (window - 1) // 2

    def smooth(img):
        date = ee.Date(img.get('system:time_start'))
        start = date.advance(-16 * half, 'day')
        end = date.advance(16 * half, 'day')

        neigh = collection.filterDate(start, end).select(band)
        mean = neigh.mean().rename(band + '_smoothed')
        return img.addBands(mean, overwrite=True)

    return collection.map(smooth)

def summarize_collection_tots(image, roi, band, scale = 250):

    reduced_value = image.select(band).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=scale,
        maxPixels = 1e13
    )

    return ee.Feature(None, {
        'date': image.date().format(),
        band: reduced_value.get(band)
    })



def fill_gaps_linear(collection, band):
    # Sort collection by time
    collection = collection.sort('system:time_start')

    def interp(img):
        img_date = ee.Date(img.get('system:time_start'))

        prev = collection.filterDate(
            ee.Date(img.get('system:time_start')).advance(-32, 'day'),
            img_date
        ).limit(1, 'system:time_start', False)

        next_ = collection.filterDate(
            img_date,
            ee.Date(img.get('system:time_start')).advance(32, 'day')
        ).limit(1)

        prev_img = ee.Image(ee.Algorithms.If(prev.size().gt(0), prev.first(), img))
        next_img = ee.Image(ee.Algorithms.If(next_.size().gt(0), next_.first(), img))

        prev_val = prev_img.select(band)
        next_val = next_img.select(band)

        # Interpolation fraction
        t = img_date.millis()
        t0 = ee.Number(prev_img.get('system:time_start'))
        t1 = ee.Number(next_img.get('system:time_start'))

        # Avoid division by zero
        frac = ee.Number(
                ee.Algorithms.If(
                    t1.neq(t0),
                    t.subtract(t0).divide(t1.subtract(t0)),
                    0
                )
            )

        interp_val = prev_val.add(next_val.subtract(prev_val).multiply(frac))
        return img.addBands(interp_val.rename(band), overwrite=True)

    return collection.map(interp)

def assign_str_date(img):
    eedate = ee.Date(img.get('system:time_start'))
    strdate = eedate.format('YYYY_MM_dd')
    return img.set('system:id', strdate)

def smooth_ts_using_savitsky_golay_modis(
    img_collection,
    band = 'NDVI',
    window_size=9,
    order=2  # keep default quadratic, but we’ll only add t
):
    """
    Apply Savitzky-Golay smoothing to MODIS NDVI time series in Earth Engine.
    """

    # --- Prepare predictors (constant + time) ---
    first_date = ee.Date(img_collection.first().get("system:time_start"))

    def add_time_bands(image):
        date = ee.Date(image.get("system:time_start"))
        t = date.difference(first_date, "day")
        return (image.select([band])
                .toFloat()
                .addBands(ee.Image(1).toFloat().rename("constant"))
                .addBands(ee.Image.constant(t).toFloat().rename("t"))
                .set("system:time_start", date.millis()))

    s1res = img_collection.map(add_time_bands)

    # --- Parameters ---
    half_window = (window_size - 1) // 2
    n_images = s1res.size()
    valid_end = ee.Number(n_images).subtract(window_size).subtract(1)
    runLength = ee.List.sequence(0, valid_end.max(0))

    # --- Local fit using linear regression ---
    def applySG(i):
        start = ee.Number(i).int()
        end = start.add(window_size).int()

        # slice window
        window_ic = ee.ImageCollection(s1res.toList(n_images).slice(start, end))

        # combine predictors + response
        combined = window_ic.select(['constant','t', band])

        # run regression
        regress = combined.reduce(ee.Reducer.linearRegression(
            numX=2, numY=1
        ))

        coeffs = regress.select('coefficients')  # 2x1 array
        coeff_img = coeffs.arrayProject([0]).arrayFlatten([['constant','x']])

        # center image
        center = start.add(half_window)
        ref = ee.Image(s1res.toList(n_images).get(center))

        # predict smoothed NDVI
        smoothed = coeff_img.expression(
            "c + m*t", {
                'c': coeff_img.select('constant'),
                'm': coeff_img.select('x'),
                't': ref.select('t')
            }
        )

        return (smoothed.rename(band)
                .set('system:time_start', ref.get('system:time_start')))

    sg_series = ee.List(runLength.map(applySG))

    sgcollection = ee.ImageCollection(sg_series)
    return sgcollection.map(assign_str_date)