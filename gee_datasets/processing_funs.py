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

        return (smoothed.rename(band + '_smooth')
                .set('system:time_start', ref.get('system:time_start')))

    sg_series = ee.List(runLength.map(applySG))

    return ee.ImageCollection(sg_series)