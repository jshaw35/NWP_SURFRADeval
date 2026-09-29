"""
Identify the cloud regime from the forecasts themselves, rather than from
the observations, and report the skill with which each forecast product
reproduces the regime diagnosed from SURFRAD.

For each SURFRAD location and each individual forecast product, the same
day-level clear-sky-index discriminator that paper_figure4_5_6.py applies
to the observations is applied to the forecast's own clear-sky index. The
resulting (forecast regime, observed regime) pairs are tallied into a
contingency table and reduced to categorical skill scores, with overcast
detection as the headline event.

The same fixed thresholds are used on both sides of the comparison on
purpose: the contingency table then measures how well a forecast product
reproduces the observed classification, rather than how well a
re-tuned classifier fits itself.

The day-level regime assignments are written to a separate dataframe so
that later analysis can re-use them without re-reading the NWP files.
Both sides carry their clear-sky index mean and standard deviation
alongside the labels, so thresholds can be revisited in post-processing.

This code should be run after pre-processing data into timeseries
format using preprocess_to_timeseries.py

"""

# %%

import xarray as xr
import pandas as pd
import numpy as np
import os

# %%
# Cloud regime labels. The integer encoding is the contiguous set
# 1..len(REGIME_NAMES), so it can be used directly as a dimension
# coordinate in the contingency tables.
REGIME_NAMES = ["empty", "clear", "broken", "cloudy"]
REGIME_EMPTY = 0
REGIME_CLEAR = 1
REGIME_BROKEN = 2
REGIME_CLOUDY = 3


def mask_nwp_by_availability(
    ds: xr.Dataset,
    mask_vars: list,
):
    """
    Function to reconcile different data availability for
    NWP forecasts by masking out times when only some products
    have data. Does not assume all variables have the same
    availability (e.g. ensemble forecasts missing DNI data but
    not GHI data).

    Inputs:
    ds: xarray Dataset
        The NWP forecast data
    mask_vars: list
        list of variables to apply masking over.

    Outputs:
    ds: xarray Dataset
        Input Dataset masked consistently across NWP forecasts.
    xr.merge(masks): xarray dataset
        Dataset with a mask for each variable.
    """

    masks = []
    # If an nwp product is nan everywhere for a variable,
    # then ignore it when masking.
    num_nan = np.isnan(ds).all(dim=["valid_time", "time"]).sum(dim="nwp_source")
    for _var in mask_vars:
        _mask = np.isnan(ds[_var]).sum(dim="nwp_source") == num_nan[_var]

        ds[_var] = ds[_var].where(_mask)
        masks.append(_mask)

    # Apply the last mask to the step. Somewhat arbitrary.
    if "step" in ds.coords:
        ds = ds.assign_coords(step=ds["step"].where(_mask))

    return ds, xr.merge(masks)


def mask_surfrad_by_qc_flags(
    ds: xr.Dataset,
    mask_threshold: float = 0.1,
):
    """
    Use the surfrad QC flags to mask variables appropriately.

    Inputs:
    ds: xarray Dataset
        Dataset to mask over.
    mask_threshold: float
        float indicating the threshold for masking out surfrad data.
        e.g. 0.1 indicates that if more than 10% of the hourly
        period was flagged then that data will be masked out.

    Outputs:
    ds: xarray Dataset
        Data that has been masked.
    masks: xarray Dataset
        Masking array for each variable.
    """

    masks = []
    for _var in ds.data_vars:
        if (_var + "_flag") in ds.data_vars:
            _var_mask = ds[_var + "_flag"] <= mask_threshold
            ds[_var] = ds[_var].where(_var_mask)
            masks.append(_var_mask.rename(_var))

    return ds, xr.merge(masks)


def load_and_mask_surfrad(
    load_path: str,
    surfrad_sitename: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
):
    """
    Load Surfrad data and mask according to data availability.

    Inputs:
    load_path: string
        path to where data is stored
    surfrad_sitename: string
        string identifier corresponding the to surfrad site.
    datetime_start: pandas Timestamp
        Timestamp for the time period start
    datetime_end: pandas Timestamp
        Timestamp for the time period end

    Outputs:
    ds: xarray Dataset
        Surfrad data that has been loaded and masked.
    masks: xarray Dataset
        Masking array for each variable.
    """

    datestring = datetime_start.strftime("%Y%m%d_") + datetime_end.strftime("%Y%m%d")
    load_dir = os.path.join(
        load_path,
        surfrad_sitename,
        datestring,
    )
    filename = f"{datestring}_surfrad_{surfrad_sitename}.nc"
    ds = xr.open_dataset(os.path.join(load_dir, filename))
    ds, masks = mask_surfrad_by_qc_flags(ds)

    # Re-index along a continuous time dimension
    continuous_time = pd.date_range(
        start=datetime_start,
        end=datetime_end,
        freq="h",
    )
    ds = ds.reindex(valid_time=continuous_time)

    return ds, masks


def load_and_mask_nwp(
    load_path: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    datavars: list,
):
    """
    Load NWP data and mask according to data availability.

    Inputs:
    load_path: string
        path to where data is stored
    surfrad_sitename: string
        stirng corresponding the to surfrad site where the NWP
        was extracted to be nearest.
    datetime_start: pandas Timestamp
        Timestamp for the time period start
    datetime_end: pandas Timestamp
        Timestamp for the time period end
    datavars: list
        List of variables to mask over

    Outputs:
    ds: xarray DataArray
        NWP data that has been loaded and masked.
    """

    load_dir = os.path.join(
        load_path,
        "nwp",
    )

    nwp_timerange = pd.DatetimeIndex(
        np.arange(
            datetime_start + np.timedelta64(-2, "D"),
            datetime_end + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
    )
    filenames = [f"{os.path.join(load_dir,i.strftime('%Y%m%d'))}_nwp.nc" for i in nwp_timerange]
    filenames = [i for i in filenames if os.path.exists(i)]

    ds = xr.open_mfdataset(filenames, combine="nested", concat_dim="time")
    # Add "step" coordinate and mask for NWP source consistency.
    ds = ds.assign_coords(step=ds.valid_time - ds.time)

    # Select only forecast timesteps in the "day-ahead".
    # f020 corresponds to Z12 + 20 or ~12am CT of the day ahead
    # f044 corresponds to Z12 + 44 or ~12am CT of day after the day ahead.
    step_mask = np.bitwise_and(
        ds.step >= np.timedelta64(20, "h"),
        ds.step < np.timedelta64(44, "h"),
    )
    ds = ds.where(step_mask)

    ds, avail_masks = mask_nwp_by_availability(ds, datavars)

    # Re-index along a continuous time dimension
    continuous_time = pd.date_range(
        start=datetime_start,
        end=datetime_end,
        freq="h",
    )
    ds = ds.reindex(valid_time=continuous_time)

    return ds, avail_masks


def reindex_dataset(
    ds: xr.Dataset,
    var_name: str,
    multi_indices: list,
):
    """
    Modifying the indexes in xarray is difficult, so I cast back to xarray.
    Casting back requires differentiating between variables and coordinates
    manually (or they get extra duplicate dimensions), which requires
    referencing the coordinates of a coordinate in the original format.

    Inputs:
    ds: xarray Dataset
        Data to reindex
    var_name: string
        Identifier for variable to operate on
    multi_indices: list
        List of new indices for the output

    Outputs:
    ds_out: xarray Dataset
        Input data reindexed with original coordinates
    """
    ds_proc = ds.copy()

    # Name to avoid errors.
    ds_proc.name = var_name
    df = ds_proc.to_dataframe()

    # Re-index.
    df = df.reset_index().set_index(multi_indices)
    # Cast back to xarray, but exclude coordiantes or they become variables.
    ds_out = df[var_name].to_xarray()

    # Re-add the coordinates that are now re-indexed by the multi-index.
    for _coord in ds.coords:
        if _coord not in ds_out.coords:  # Check if a coordinate is missing
            _coord_da = df[_coord].to_xarray()  # If so, get it from the Dataframe
            for (
                _extra_coord
            ) in _coord_da.coords:  # Remove extra dimensions before re-adding
                if _extra_coord not in ds[_coord].coords:
                    # Unclear how best to reduce dimensions if masking is present
                    _coord_da = _coord_da.mean(dim=_extra_coord)
                    # _coord_da = _coord_da.isel({_extra_coord:0}).drop_vars(_extra_coord)

            ds_out = ds_out.assign_coords({_coord: _coord_da})
    return ds_out


def shift_and_reindex_time(
    data: xr.Dataset,
    data_var: str,
    time_var: str,
    utc_shift: int,
):
    """
    Shift a time variable by a constant offset and then reindex
    the time variable to a [day, hour] format.

    Inputs:
    data: xarray Dataset
        Data to operate on.
    data_var: string
        Identifier for variables in the data.
    time_var: string
        String identifying the time dimension to reindex.
    utc_shift: integer
        Number of hours to shift data by to get a local time.
        e.g. utc_shift = -6 sets the time coordinate to UTC - 6, or ~ET.
    """

    data = data.assign_coords(utc_shift=data[time_var] + np.timedelta64(utc_shift, "h"))

    # Extract day and hour components
    data["dayofyear"] = data["utc_shift"].dt.dayofyear
    data["hour"] = data["utc_shift"].dt.hour

    new_dims = [i for i in data.dims if i != "valid_time"]
    new_dims.extend(["dayofyear", "hour"])
    data_reindexed = reindex_dataset(
        ds=data,
        var_name=data_var,
        multi_indices=new_dims,
    )

    return data_reindexed


def build_product_list(
    nwp_dayahead_ds: xr.Dataset,
    requested_products: list,
):
    """
    Resolve the list of forecast products to analyze against the
    products that are actually present in the data. Any requested
    product that is missing is reported rather than silently dropped,
    since a hardcoded list would raise a KeyError later on if the
    analysis period were narrowed.

    Inputs:
    nwp_dayahead_ds: xarray Dataset
        The day-ahead NWP data, carrying the nwp_source dimension.
    requested_products: list
        The forecast products to analyze, in reporting order.

    Outputs:
    products: list
        The subset of requested_products that is present in the data.
    """

    available = [str(i) for i in nwp_dayahead_ds.nwp_source.values]

    missing = [i for i in requested_products if i not in available]
    if missing:
        print(
            "WARNING: requested forecast products are absent from the data "
            f"and will be skipped: {missing}"
        )

    return [i for i in requested_products if i in available]


def compute_clearsky_index(
    var_data: xr.DataArray,
    clearsky_data: xr.DataArray,
    sunlit_mask: xr.DataArray,
):
    """
    Compute a clear-sky index and restrict it to sunlit timesteps.

    The clear-sky reference is the pvlib clear-sky irradiance that
    preprocess.py adds to the SURFRAD record. It is a location-based
    model calculation rather than an observation, so it is a valid
    normalizer for the forecast as well as the observations.

    Both the forecast and the observation index are built with this
    function, and both are restricted by the same sunlit_mask, so that
    the two sides of the contingency table are always computed over an
    identical set of hours.

    Inputs:
    var_data: xarray DataArray
        GHI to normalize, e.g. SURFRAD ghi or the NWP dswrf.
    clearsky_data: xarray DataArray
        Clear-sky GHI on the same time dimension.
    sunlit_mask: xarray DataArray
        Boolean mask selecting the sunlit timesteps.

    Outputs:
    csi: xarray DataArray
        Clear-sky index, NaN outside the sunlit timesteps.
    """

    return (var_data / clearsky_data).where(sunlit_mask)


def classify_cloud_regime(
    csi: xr.DataArray,
    clear_thresh: float = 0.95,
    broken_thresh: float = 0.92,
    broken_stddev_weight: float = 2.0,
    min_sunlit_hours: int = 0,
):
    """
    Apply the day-level clear-sky-index cloud regime discriminator to a
    clear-sky index field. The thresholds are the same ones that
    paper_figure4_5_6.py uses to classify the observed days, and they are
    deliberately applied to the forecast with no re-tuning.

    A day is classified as:

    clear  if the daily mean clear-sky index exceeds clear_thresh;
    broken if it is not clear, but the daily mean plus
           broken_stddev_weight standard deviations exceeds broken_thresh,
           which admits variable days whose mean alone is low;
    cloudy if neither of the above;
    empty if the day has no sunlit hours with usable data.

    The standard deviation term is what separates a genuinely broken day
    from a uniformly hazy one; the classes remain mutually exclusive
    because the clear test is applied first and the standard deviation is
    never negative.

    The classification is purely categorical, which discards information
    about how confidently a forecast places a day on one side of a
    threshold. A probabilistic treatment is possible in post-processing:
    the daily clear-sky index mean recorded alongside the label is a
    continuous clearness signal, and could be used as a "clearness"
    probability and scored with a Brier score against the observed
    clear/cloudy indicator, which would be sensitive to the degree of
    forecast cloudiness rather than only to whether the label crossed
    the threshold.

    Inputs:
    csi: xarray DataArray
        Clear-sky index with a dayofyear dimension. An ensemble member
        of the dayofyear dimension is allowed.
    clear_thresh: float
        Daily mean clear-sky index above which a day is clear.
    broken_thresh: float
        Threshold applied to the daily mean plus the weighted standard
        deviation when testing for broken conditions.
    broken_stddev_weight: float
        Weight applied to the daily standard deviation in the broken test.
    min_sunlit_hours: int
        Minimum number of sunlit hours with usable data required to
        classify a day as anything other than empty. The default of 0
        matches paper_figure4_5_6.py, which classifies some days from as
        few as three sunlit hours. Raising it to around 6 would drop
        those marginal days. The sunlit hour counts are returned either
        way, so the choice can be revisited in post-processing.

    Outputs:
    regime: xarray DataArray
        Integer regime label over dayofyear, valued in REGIME_NAMES.
    csi_mean: xarray DataArray
        Daily mean clear-sky index.
    csi_stddev: xarray DataArray
        Daily standard deviation of the clear-sky index.
    n_hours: xarray DataArray
        Number of sunlit hours contributing to each day.
    """

    n_hours = csi.count(dim="hour")
    csi_mean = csi.mean(dim="hour")
    csi_stddev = csi.std(dim="hour")

    # Days with too little data to classify, and days with no data at all.
    n_hours_mask = n_hours >= min_sunlit_hours
    empty_mask = n_hours_mask & csi_mean.isnull()

    clear_mask = n_hours_mask & (csi_mean > clear_thresh)
    broken_mask = (
        n_hours_mask
        & ~clear_mask
        & (csi_mean + broken_stddev_weight * csi_stddev > broken_thresh)
    )
    cloudy_mask = n_hours_mask & ~(clear_mask | broken_mask | empty_mask)

    # Accumulate the labels. Later categories override earlier ones, so
    # the ordering also enforces that the categories stay exclusive.
    regime = xr.full_like(csi_mean, REGIME_EMPTY, dtype=int)
    regime = regime.where(~cloudy_mask, REGIME_CLOUDY)
    regime = regime.where(~broken_mask, REGIME_BROKEN)
    regime = regime.where(~clear_mask, REGIME_CLEAR)

    return regime, csi_mean, csi_stddev, n_hours


def contingency_table(
    obs_regime: xr.DataArray,
    fcast_regime: xr.DataArray,
    obs_empty: int = REGIME_EMPTY,
    fcast_empty: int = REGIME_EMPTY,
):
    """
    Tally the observed and forecast regime labels against each other.

    Days labelled empty on either side carry no information about the
    discriminator, so they are excluded and counted separately. The
    remaining days are intersected, so each forecast product is scored
    on the days where both labels exist.

    Inputs:
    obs_regime: xarray DataArray
        Observed regime labels over dayofyear.
    fcast_regime: xarray DataArray
        Forecast regime labels over dayofyear, optionally with a leading
        ensemble member dimension.
    obs_empty: int
        Observed label to exclude as uninformative.
    fcast_empty: int
        Forecast label to exclude as uninformative.

    Outputs:
    table: xarray DataArray
        Day counts with dimensions (nwp_source, obs_regime, fcast_regime).
    obs_counts: xarray DataArray
        Number of informative observed days per observed regime.
    fcast_empty_counts: xarray DataArray
        Number of days excluded because the forecast label is empty.
    obs_empty_counts: xarray DataArray
        Number of days excluded because the observed label is empty.
    """

    obs_useful = obs_regime != obs_empty
    fcast_useful = fcast_regime != fcast_empty

    # to_dataframe broadcasts the observed labels against the leading
    # product dimension of the forecast labels, so the joint tally can be
    # taken in one groupby.
    joint = xr.Dataset(
        {
            "obs_regime": obs_regime,
            "fcast_regime": fcast_regime,
        }
    ).to_dataframe()
    informative = (
        (joint["obs_regime"].to_numpy() != obs_empty)
        & (joint["fcast_regime"].to_numpy() != fcast_empty)
    )
    counts = (
        joint[informative]
        .groupby(["nwp_source", "obs_regime", "fcast_regime"])
        .size()
    )

    # Reindex onto the full grid of labels, so that every combination
    # exists and can be selected by label below, and so that unreachables
    # are explicit zeros rather than missing entries. The unstacked frame
    # is then rebuilt with explicit dimensions, because wrapping a
    # MultiIndexed Series directly would leave the labels as index levels
    # rather than as dimensions.
    full_index = pd.MultiIndex.from_product(
        [
            [str(i) for i in fcast_regime.nwp_source.values],
            list(range(len(REGIME_NAMES))),
            list(range(len(REGIME_NAMES))),
        ],
        names=["nwp_source", "obs_regime", "fcast_regime"],
    )
    grid = counts.reindex(full_index, fill_value=0).astype(int).unstack("fcast_regime")
    # The coordinates are rebuilt in the same nesting order that
    # from_product used above, which is the row and column order that
    # unstack leaves the frame in.
    nwp_coord = np.asarray([str(i) for i in fcast_regime.nwp_source.values])
    obs_coord = np.arange(len(REGIME_NAMES))
    fcast_coord = np.arange(len(REGIME_NAMES))
    table = xr.DataArray(
        grid.to_numpy().reshape(
            nwp_coord.size, obs_coord.size, fcast_coord.size
        ),
        dims=("nwp_source", "obs_regime", "fcast_regime"),
        coords={
            "nwp_source": nwp_coord,
            "obs_regime": obs_coord,
            "fcast_regime": fcast_coord,
        },
    )

    obs_counts = obs_useful.sum(dim="dayofyear")
    fcast_empty_counts = (fcast_regime == fcast_empty).sum(dim="dayofyear")
    obs_empty_counts = (obs_regime == obs_empty).sum(dim="dayofyear")

    return table, obs_counts, fcast_empty_counts, obs_empty_counts


def category_skill_scores(
    table: xr.Dataset,
    obs_category: int,
):
    """
    Reduce a contingency table to one-vs-rest categorical skill scores
    for a single observed category, treated as the event of interest.

    Inputs:
    table: xarray DataArray
        Day counts with dimensions (nwp_source, obs_regime, fcast_regime).
    obs_category: int
        The observed regime treated as the event.

    Outputs:
    scores: xarray Dataset
        Skill scores and the underlying counts, over nwp_source.
    """

    if obs_category not in table.obs_regime:
        raise ValueError(
            f"Observed category {obs_category} is absent from the contingency table."
        )

    # Each regime is scored as a one-versus-rest problem, so every label
    # other than the event is pooled into a single "not the event" class.
    # The empty label is already absent from the table, because days that
    # are uninformative on either side were excluded when it was built.
    other_obs = [int(i) for i in table.obs_regime.values if int(i) != obs_category]
    other_fcast = [int(i) for i in table.fcast_regime.values if int(i) != obs_category]

    hits = table.sel(obs_regime=obs_category, fcast_regime=obs_category)
    misses = table.sel(obs_regime=obs_category).sel(fcast_regime=other_fcast).sum(
        "fcast_regime"
    )
    false_alarms = table.sel(fcast_regime=obs_category).sel(obs_regime=other_obs).sum(
        "obs_regime"
    )
    correct_negatives = (
        table.sel(obs_regime=other_obs)
        .sel(fcast_regime=other_fcast)
        .sum(["obs_regime", "fcast_regime"])
    )

    # Selecting a single label leaves behind a zero-dimensional coordinate
    # recording which label was chosen. Those differ between the counts
    # above and would collide when the scores are gathered into a Dataset,
    # so they are dropped here.
    def bare(counts):
        scalars = [name for name, coord in counts.coords.items() if coord.ndim == 0]
        return counts.drop_vars(scalars) if scalars else counts

    hits = bare(hits)
    misses = bare(misses)
    false_alarms = bare(false_alarms)
    correct_negatives = bare(correct_negatives)

    # A zero denominator is mapped to NaN by masking it out, which yields
    # missing values rather than a divide-by-zero warning, so that
    # unreachables show up as gaps in the reported table.
    def ratio(num, den):
        return num / den.where(den != 0)

    pod = ratio(hits, hits + misses)
    far = ratio(false_alarms, hits + false_alarms)
    csi_threat = ratio(hits, hits + misses + false_alarms)
    bias_score = ratio(hits + false_alarms, hits + misses)

    total = hits + misses + false_alarms + correct_negatives
    accuracy = ratio(hits + correct_negatives, total)
    hss = ratio(
        2 * (hits * correct_negatives - false_alarms * misses),
        (hits + false_alarms) * (misses + correct_negatives)
        + hits * correct_negatives
        + false_alarms * misses,
    )

    return xr.Dataset(
        {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_negatives,
            "n_events": hits + misses,
            "n_non_events": false_alarms + correct_negatives,
            "pod": pod,
            "far": far,
            "csi_threat": csi_threat,
            "bias_score": bias_score,
            "accuracy": accuracy,
            "hss": hss,
        }
    )


def build_skill_table(
    assignments: pd.DataFrame,
    scoring_categories: list,
    obs_column: str = "obs_regime",
    fcast_column: str = "fcast_regime",
    empty_label: str = "empty",
):
    """
    Build the reported skill table from the day-level regime assignments.

    Inputs:
    assignments: pandas DataFrame
        Tidy day-level regime assignments, one row per
        (location, dayofyear, nwp_source).
    scoring_categories: list
        Observed categories to score, in reporting order.
    obs_column: string
        Column holding the observed regime label.
    fcast_column: string
        Column holding the forecast regime label.
    empty_label: string
        Label excluded from scoring.

    Outputs:
    skill: pandas DataFrame
        One row per (location, nwp_source, obs_regime).
    """

    records = []
    for (location, nwp_source), group in assignments.groupby(
        ["location", "nwp_source"], sort=False
    ):
        obs = group[obs_column]
        fcast = group[fcast_column]

        # Days labelled empty on either side carry no information about the
        # discriminator, so each product is scored on the days where both
        # labels exist. The excluded days are reported alongside the scores
        # so that the day accounting can be checked.
        informative = (obs != empty_label) & (fcast != empty_label)
        n_common = int(informative.sum())
        n_excluded = int((~informative).sum())
        obs_common = obs[informative]
        fcast_common = fcast[informative]

        for category in scoring_categories:
            in_obs = obs_common == category
            in_fcast = fcast_common == category
            hits = int((in_obs & in_fcast).sum())
            misses = int((in_obs & ~in_fcast).sum())
            false_alarms = int((~in_obs & in_fcast).sum())
            correct_negatives = int((~in_obs & ~in_fcast).sum())

            n_events = hits + misses
            n_non_events = false_alarms + correct_negatives
            total = n_events + n_non_events

            def ratio(num, den):
                return num / den if den != 0 else np.nan

            pod = ratio(hits, n_events)
            far = ratio(false_alarms, hits + false_alarms)
            csi_threat = ratio(hits, n_events + false_alarms)
            bias_score = ratio(hits + false_alarms, n_events)
            accuracy = ratio(hits + correct_negatives, total)
            hss = ratio(
                2 * (hits * correct_negatives - false_alarms * misses),
                (hits + false_alarms) * (misses + correct_negatives)
                + hits * correct_negatives
                + false_alarms * misses,
            )

            records.append(
                {
                    "location": location,
                    "nwp_source": nwp_source,
                    "obs_regime": category,
                    "n_days_common": n_common,
                    "n_days_excluded": n_excluded,
                    "n_obs": n_events,
                    "n_fcast": hits + false_alarms,
                    "n_non_events": n_non_events,
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "correct_negatives": correct_negatives,
                    "pod": pod,
                    "far": far,
                    "csi_threat": csi_threat,
                    "bias_score": bias_score,
                    "accuracy": accuracy,
                    "hss": hss,
                }
            )

    skill = pd.DataFrame.from_records(records)

    # Assert that the scored days and the excluded days account for every
    # day, so that no day is silently dropped between the two files. The
    # scored counts partition the common sample once per reported
    # category, so they are expected to sum to the common sample repeated
    # by the number of categories.
    n_categories = len(scoring_categories)
    totals = assignments.groupby(["location", "nwp_source"], sort=False).size()
    for (location, nwp_source), counts in (
        skill.groupby(["location", "nwp_source"], sort=False)[
            ["n_obs", "n_non_events"]
        ]
        .sum()
        .iterrows()
    ):
        subset = skill[
            (skill["location"] == location) & (skill["nwp_source"] == nwp_source)
        ]
        n_common = int(subset["n_days_common"].iloc[0])
        n_excluded = int(subset["n_days_excluded"].iloc[0])
        n_total = int(totals.loc[(location, nwp_source)])
        assert n_common + n_excluded == n_total, (
            f"Day accounting failed for {location}/{nwp_source}: "
            f"{n_common} common plus {n_excluded} excluded days out of {n_total}."
        )
        assert int(counts.sum()) == n_common * n_categories, (
            f"Scored counts failed for {location}/{nwp_source}: "
            f"{int(counts.sum())} scored out of {n_common * n_categories} expected."
        )

    return skill


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    data_save_path = "data/figure_outputs"
    save_data = True

    surfrad_var = "ghi"
    surfrad_clearsky_var = f"clearsky_{surfrad_var}"
    nwp_var = "dswrf"

    year_start = 2024
    month_start = 3
    day_start = 1

    year_end = 2024
    month_end = 7
    day_end = 8

    data_datetime_start = pd.Timestamp(year_start, month_start, day_start)
    data_datetime_end = pd.Timestamp(year_end, month_end, day_end)

    # Only the individual members are informative about the day
    # classification. The ensemble means are left out deliberately:
    # averaging blurs the diurnal cloud signal that the discriminator
    # relies on, so an ensemble mean is not a meaningful subject for it.
    nwp_products = [
        "hrrr",
        "rrfs_control",
        "rrfs_mem0001",
        "rrfs_mem0002",
        "rrfs_mem0003",
        "rrfs_mem0004",
        "rrfs_mem0005",
    ]

    # Cloud regime discriminator thresholds. These are the same values
    # that paper_figure4_5_6.py uses on the observations, and they are
    # applied to the forecasts unaltered so that the contingency table
    # measures the forecast rather than a re-tuned classifier.
    clear_thresh = 0.95
    broken_thresh = 0.92
    broken_stddev_weight = 2.0
    broken_thresh_hrrr_correction = 0.02
    sunlit_zenith = 80

    # The sunlit hours are selected by the SURFRAD solar zenith angle, so
    # that the observations and every forecast product are classified over
    # an identical set of hours. A classification error can therefore
    # never be an artifact of the two sides disagreeing about which hours
    # are sunlit. No minimum hour count is imposed, which matches
    # paper_figure4_5_6.py; see the note in classify_cloud_regime.
    min_sunlit_hours = 0

    # Overcast detection is the headline result, so the event category is
    # named explicitly. The other categories are scored too, because the
    # classes are imbalanced enough that the clear and broken numbers are
    # needed to interpret the overcast ones.
    scoring_categories = ["cloudy", "clear", "broken"]

    surfrad_sitenames = ["dra", "tbl", "fpk", "sxf", "bon", "gwn", "psu"]
    utc_shift_dict = {
        "dra": -8,
        "tbl": -7,
        "fpk": -7,
        "sxf": -6,
        "bon": -6,
        "gwn": -6,
        "psu": -5,
    }

    nwp_datavars = [
        "dswrf",
        "vbdsf",
        "vddsf",
    ]

    if save_data and (not os.path.exists(data_save_path)):
        os.makedirs(data_save_path)

    # The NWP files are the same for every site, so load them once and
    # subset to each site inside the loop.
    nwp_ds, nwp_masks = load_and_mask_nwp(
        load_path,
        data_datetime_start,
        data_datetime_end,
        nwp_datavars,
    )
    products = build_product_list(nwp_ds, nwp_products)

    assignments_list = []
    for surfrad_sitename in surfrad_sitenames:
        print(f"Processing {surfrad_sitename.upper()}")
        utc_shift = utc_shift_dict[surfrad_sitename]

        surfrad_ds, surfrad_masks = load_and_mask_surfrad(
            load_path,
            surfrad_sitename,
            data_datetime_start,
            data_datetime_end,
        )
        surfrad_ds = surfrad_ds.load()

        # Select the site. The "location" coordinate is dropped because
        # reindex_dataset reduces the coordinates it re-adds with a mean,
        # which cannot be applied to a string coordinate.
        site_nwp_ds = nwp_ds.sel(location=surfrad_sitename).drop_vars("location")
        site_nwp_masks = nwp_masks.sel(location=surfrad_sitename).drop_vars("location")

        # Collapse the forecast time dimension so the forecasts appear as
        # a timeseries. The f020-f044 window is 24 hours wide and the
        # record holds a single 12Z cycle per day, so each valid time
        # belongs to exactly one cycle and this picks that cycle's value.
        nwp_dayahead_ds = site_nwp_ds.sel(nwp_source=products)
        nwp_dayahead_ds = nwp_dayahead_ds.sum(dim="time", min_count=1).load()

        nwp_mask = site_nwp_masks[nwp_var].sum(dim="time")
        obs_mask = surfrad_masks[surfrad_var]

        surfrad_ds = surfrad_ds.where(nwp_mask > 0) # Mask identically to observations

        # The sunlit mask, in local time, shared by both sides.
        surfrad_sunlit = shift_and_reindex_time(
            surfrad_ds[surfrad_var].where(obs_mask).load(),
            # surfrad_ds[surfrad_var].where(obs_mask & (nwp_mask > 0)).load(),
            surfrad_var,
            "valid_time",
            utc_shift=utc_shift,
        )
        sunlit_mask = surfrad_sunlit.zenith < sunlit_zenith
        # The clear-sky reference is a coordinate of the observations
        # rather than a data variable, so it is passed with the
        # observations' own variable name. A coordinate selected on its
        # own is carried along as a coordinate of itself, which would put
        # two identically named columns into the frame that
        # reindex_dataset builds and cannot be converted back to xarray.
        clearsky_reindexed = shift_and_reindex_time(
            surfrad_ds[surfrad_clearsky_var].where(obs_mask).load(),
            surfrad_var,
            "valid_time",
            utc_shift=utc_shift,
        )

        # Observed cloud regime, the reference classification.
        obs_csi = compute_clearsky_index(
            surfrad_sunlit,
            clearsky_reindexed,
            sunlit_mask,
        )
        obs_regime, obs_csi_mean, obs_csi_stddev, obs_n_hours = classify_cloud_regime(
            obs_csi,
            clear_thresh=clear_thresh,
            broken_thresh=broken_thresh,
            broken_stddev_weight=broken_stddev_weight,
            min_sunlit_hours=min_sunlit_hours,
        )

        # Forecast cloud regime, classified independently per product.
        fcast_reindexed = shift_and_reindex_time(
            nwp_dayahead_ds[nwp_var].where(nwp_mask).load(),
            nwp_var,
            "valid_time",
            utc_shift=utc_shift,
        )
        fcast_csi = compute_clearsky_index(
            fcast_reindexed,
            clearsky_reindexed,
            sunlit_mask,
        )
        fcast_regime, fcast_csi_mean, fcast_csi_stddev, fcast_n_hours = classify_cloud_regime(
            fcast_csi,
            clear_thresh=clear_thresh,
            broken_thresh=broken_thresh + broken_thresh_hrrr_correction,
            broken_stddev_weight=broken_stddev_weight,
            min_sunlit_hours=min_sunlit_hours,
        )

        # Broadcast the per-day observed quantities across the products and
        # stack everything into one tidy table. The observed quantities
        # enter the table repeated for each product, so that the file can
        # be read or filtered on its own.
        n_products = fcast_regime.sizes["nwp_source"]
        obs_regime = obs_regime.expand_dims(nwp_source=fcast_regime.nwp_source)
        obs_csi_mean = obs_csi_mean.expand_dims(nwp_source=fcast_regime.nwp_source)
        obs_csi_stddev = obs_csi_stddev.expand_dims(nwp_source=fcast_regime.nwp_source)
        obs_n_hours = obs_n_hours.expand_dims(nwp_source=fcast_regime.nwp_source)

        site_assignments = xr.Dataset(
            {
                "obs_regime": obs_regime,
                "fcast_regime": fcast_regime,
                "obs_csi_mean": obs_csi_mean,
                "obs_csi_stddev": obs_csi_stddev,
                "fcast_csi_mean": fcast_csi_mean,
                "fcast_csi_stddev": fcast_csi_stddev,
                "obs_n_sunlit_hours": obs_n_hours,
                "fcast_n_sunlit_hours": fcast_n_hours,
            }
        )
        site_frame = site_assignments.to_dataframe().reset_index()

        # Replace the integer labels with their names, so that the saved
        # file is readable without a lookup table.
        for _column in ["obs_regime", "fcast_regime"]:
            site_frame[_column] = site_frame[_column].map(
                {i: REGIME_NAMES[i] for i in range(len(REGIME_NAMES))}
            )

        site_frame["location"] = surfrad_sitename
        # shift_and_reindex_time keys days by dayofyear, which is only
        # unique within a single year, so the calendar date is rebuilt
        # from the year of the analysis period.
        site_frame["date"] = [
            (
                pd.Timestamp(year=year_start, month=1, day=1)
                + pd.Timedelta(days=int(i) - 1)
            ).strftime("%Y-%m-%d")
            for i in site_frame["dayofyear"]
        ]

        assignments_list.append(
            site_frame[
                [
                    "location",
                    "dayofyear",
                    "date",
                    "nwp_source",
                    "obs_regime",
                    "fcast_regime",
                    "obs_csi_mean",
                    "obs_csi_stddev",
                    "fcast_csi_mean",
                    "fcast_csi_stddev",
                    "obs_n_sunlit_hours",
                    "fcast_n_sunlit_hours",
                ]
            ]
        )

    # %%
    assignments = pd.concat(assignments_list, ignore_index=True)
    skill = build_skill_table(assignments, scoring_categories)

    # Remove the desert locations and treat as a single location
    DESERT_SITES = ["dra", "tbl"]
    assignments_non_desert = assignments[
        ~assignments.location.isin(DESERT_SITES)
    ].copy()
    assignments_non_desert.loc[:, "location"] = "non_desert"
    skill_non_desert = build_skill_table(assignments_non_desert, scoring_categories)

    # %%
    # Cross-check the reported counts against a tally taken with a
    # separate code path, so that a mistake in either one is caught before
    # the numbers are written out. The headline overcast category is
    # checked, because it is the result the comparison is reported on.
    label_to_int = {name: i for i, name in enumerate(REGIME_NAMES)}
    for (_location, _nwp_source), _group in assignments.groupby(
        ["location", "nwp_source"], sort=False
    ):
        _dayofyear = _group["dayofyear"].to_numpy()
        _obs = xr.DataArray(
            _group["obs_regime"].map(label_to_int).to_numpy(),
            dims="dayofyear",
            coords={"dayofyear": _dayofyear},
        )
        _fcast = xr.DataArray(
            _group["fcast_regime"].map(label_to_int).to_numpy(),
            dims="dayofyear",
            coords={"dayofyear": _dayofyear},
        ).expand_dims(nwp_source=[_nwp_source])
        _tallied = category_skill_scores(
            contingency_table(_obs, _fcast)[0], obs_category=REGIME_CLOUDY
        )
        _reported = skill[
            (skill["location"] == _location)
            & (skill["nwp_source"] == _nwp_source)
            & (skill["obs_regime"] == REGIME_NAMES[REGIME_CLOUDY])
        ].iloc[0]
        for _key in ["hits", "misses", "false_alarms", "correct_negatives"]:
            assert int(_reported[_key]) == int(_tallied[_key]), (
                f"{_key} disagree for {_location}/{_nwp_source}: "
                f"{int(_reported[_key])} reported against "
                f"{int(_tallied[_key])} tallied."
            )

    # %%
    # Report the observed regime distribution, which is the reference the
    # skill scores are measured against.
    print("")
    print("Observed regime distribution (days):")
    print(assignments[assignments.nwp_source == products[0]].groupby("obs_regime").size())
    print("")
    print("Forecast regime distribution (days):")
    print(assignments.groupby(["nwp_source", "fcast_regime"]).size().unstack("fcast_regime"))

    print("")
    for _category in scoring_categories:
        print(f"{_category.upper()}")
        _cat_skill = skill[skill.obs_regime == _category]
        print(_cat_skill[["location", "nwp_source", "n_days_common", "n_obs", "pod", "far", "csi_threat", "bias_score", "hss"]].to_string(index=False))
        print("")

    # %%
    if save_data:
        assignments_path = os.path.join(
            data_save_path, "forecast_cloudregime_assignments.csv"
        )
        skill_path = os.path.join(data_save_path, "forecast_cloudregime_skill.csv")
        assignments.to_csv(assignments_path, index=False)
        skill.to_csv(skill_path, index=False)
        print(f"Saved {assignments_path}")
        print(f"Saved {skill_path}")
    # %%

    print("")
    print(f"Cross-checked the {REGIME_NAMES[REGIME_CLOUDY]} counts for "
          f"{assignments.groupby(['location', 'nwp_source']).ngroups} "
          "site and product pairs against an independent xarray tally.")
    # %%

    # Remove the desert locations and treat as a single location
    DESERT_SITES = ["dra", "tbl"]
    assignments_non_desert = assignments[
        ~assignments.location.isin(DESERT_SITES)
    ].copy()
    assignments_non_desert.loc[:, "location"] = "non_desert"
    skill_non_desert = build_skill_table(assignments_non_desert, scoring_categories)

    # %%
    if save_data:
        assignments_path = os.path.join(
            data_save_path, "forecast_cloudregime_assignments_nondesert.csv"
        )
        skill_path = os.path.join(data_save_path, "forecast_cloudregime_skill_nondesert.csv")
        assignments_non_desert.to_csv(assignments_path, index=False)
        skill_non_desert.to_csv(skill_path, index=False)
        print(f"Saved {assignments_path}")
        print(f"Saved {skill_path}")

    # %%
    # Select variables for the table
    variables = ["nwp_source", "obs_regime", "n_obs", "n_fcast", "pod", "far", "csi_threat", "bias_score", "hss"]
    skill_non_desert_table = skill_non_desert[variables].sort_values(by=["obs_regime", "nwp_source"])
    # Round POD, FAR, CSI, Bias, and HSS to 2 decimal places for better readability
    skill_non_desert_table[["pod", "far", "csi_threat", "bias_score", "hss"]] = skill_non_desert_table[["pod", "far", "csi_threat", "bias_score", "hss"]].round(2)
    skill_non_desert_table.to_latex()
    # %%