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
import matplotlib.pyplot as plt
import seaborn as sns
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


def undo_shift_and_reindex_time(
    data: xr.Dataset,
    time_var: str,
    utc_shift: int,
    year=2024,
):
    """
    Undo the shift_and_reindex_time operation to map hour dayofyear back to a continuous time dimension.
    This means reconstructing the valid_time dimension from the dayofyear and hour dimensions, and then shifting it back to UTC.
    """
    # Combine each day/hour pair into a timestamp, then undo the local-time
    # shift.  The explicit year avoids pandas defaulting to year 1900.
    year_repeat = np.repeat(year, len(data["dayofyear"]) * len(data["hour"]), axis=0)
    dayofyear = np.asarray(data["dayofyear"].values)
    # Repeat dayofyear for each hour to match the shape of the hour dimension
    dayofyear_repeat = pd.to_timedelta(np.repeat(dayofyear[:, np.newaxis], len(data["hour"]), axis=0).squeeze(), unit="D")
    hour = np.asarray(data["hour"].values)
    hour_repeat = pd.to_timedelta(np.repeat(hour[np.newaxis, :], len(data["dayofyear"]), axis=0).ravel(), unit="h")
    local_time = (
        pd.Timestamp(year=year, month=1, day=1)
        + dayofyear_repeat
        + hour_repeat
    )
    local_time = local_time - pd.to_timedelta(utc_shift, unit="h") - pd.to_timedelta(1, unit="D")  # Subtract 1 day to account for the fact that dayofyear starts at 1
    # time = (
        # local_time.ravel() - pd.to_timedelta(utc_shift, unit="h")
    # ).to_numpy()

    data = data.stack({time_var: ("dayofyear", "hour")})
    data = data.assign_coords({time_var: (time_var, local_time)})
    return data.sortby(time_var)


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


def sigmoid(x, A, alpha):
    return A/(1 + np.exp(-alpha * x))


def plot_diurnal_whiskerplot(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.DataArray,
    surfrad_masks: xr.DataArray,
    utc_shift: int,
    meteo_mask: xr.DataArray = None,
    meteo_name: str = None,
    nwp_sources_plot: list = ["hrrr", "rrfs_control"],
    nwp_sources_analyze: list = ["hrrr", "rrfs_control"],
    fontsize: float = 16,
    axs: np.ndarray = None,
    orientation: str = "horizontal",
    num_panels_dim: int = 2,
    whis: float or (float, float) = 1.5,
    showfliers: bool = False,
):
    """
    Visualize the error and show the spread for each hour 
    of the day using box and whisker plots.

    Inputs:
    surfrad_var: string
        variable identifier for SURFRAD observations.
    nwp_var: string
        variable identifier for NWP forecast data.
    surfrad_ds: xarray Dataset
        Dataset containing SURFRAD observations.
    nwp_ds: xarray Dataset
        Dataset containing NWP forecast data.
    nwp_masks: xarray Dataset
        Dataset containing masks for each NWP variable.
    surfrad_masks: xarray Dataset
        Dataset containing masks for each obs variable.
    utc_shift: integer
        number of hours to shift data by to get a local time.
        e.g. utc_shift = -6 sets the time coordinate to UTC - 6, or ~ET.
    meteo_mask: xarray Dataset
        Mask for meteorological conditions.
    meteo_name: string
        String appended to output file name to identify the mask.
    fontsize: float
        float for determining figure fontsizes
    orientation: string
        How to orient the figure panels: {"horizontal","vertical"}
    num_panels_dim: integer
        Number of panels to allow along the non-orientation dimension.
    whis: float or (float, float)
        The reach of the whiskers to the box length.
        See matplotlib boxplot documentation for details.
    show_fliers: boolean
        Whether to show outliers in the box and whisker plot.
    """
    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    surfrad_mask = surfrad_masks[surfrad_var]
    all_mask = nwp_mask & surfrad_mask
    if meteo_mask is not None:
        # Need to convert the mask to the valid_time dimension.
        meteo_days = meteo_mask.dayofyear.isel(dayofyear=meteo_mask)
        meteo_mask = [i.values in meteo_days for i in all_mask.valid_time.dt.dayofyear]
        all_mask = all_mask & meteo_mask
    else:
        meteo_name = None

    nwp_data = nwp_ds[nwp_var].where(all_mask).load()
    surfrad_data = surfrad_ds[surfrad_var].where(all_mask).load()

    nwp_reindexed = shift_and_reindex_time(
        nwp_data,
        nwp_var,
        "valid_time",
        utc_shift=utc_shift,
    )
    surfrad_reindexed = shift_and_reindex_time(
        surfrad_data,
        surfrad_var,
        "valid_time",
        utc_shift=utc_shift,
    )

    nwp_error_reindexed = nwp_reindexed - surfrad_reindexed

    # Create the figure and axes if not provided
    if axs is None:
        num_panels = len(nwp_error_reindexed.nwp_source)
        panel_size = 6
        if orientation == "horizontal":
            fig, axs = plt.subplots(
                num_panels_dim,
                int(np.ceil((num_panels) / num_panels_dim)),
                figsize=(
                    panel_size * np.ceil(num_panels / num_panels_dim),
                    panel_size * num_panels_dim,
                ),
            )
            fig.subplots_adjust(wspace=0.38, hspace=0.25)

        if orientation == "vertical":
            fig, axs = plt.subplots(
                int(np.ceil((num_panels) / num_panels_dim)),
                num_panels_dim,
                figsize=(
                    panel_size * num_panels_dim,
                    panel_size * np.ceil(num_panels / num_panels_dim),
                ),
            )
            fig.subplots_adjust(wspace=0.3, hspace=0.25)
        axs = [*axs.flat]

    stddevs = []
    rmses = []
    mbes = []
    colors = [sns.color_palette("colorblind")[0] for i in axs]
    for _nwp_source in nwp_sources_analyze:
    # for _nwp_source in nwp_error_reindexed.nwp_source:
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        error_stddev = error_data.where(surfrad_reindexed.zenith < 80).std()
        error_rmse = np.sqrt((error_data**2).where(surfrad_reindexed.zenith < 80).mean())
        error_mbe = error_data.where(surfrad_reindexed.zenith < 80).mean()
        stddevs.append(error_stddev)
        rmses.append(error_rmse)
        mbes.append(error_mbe)

    for _nwp_source, ax, _color in zip(nwp_sources_plot, axs, colors):
    # for _nwp_source, ax, _color in zip(nwp_error_reindexed.nwp_source, axs, colors):
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)

        # source_rmse = rmse.sel(nwp_source=_nwp_source)
        # # Only compute the RMSE over sunlit timesteps for consistency with
        # # other error metrics.
        error_stddev = error_data.where(surfrad_reindexed.zenith < 80).std()

        # Only plot if the source is in the list of sources to plot.
        # if _nwp_source in nwp_sources:

        # Convert error data to a format suitable for box plots
        error_values_by_hour = [error_data.sel(hour=h).values.flatten() for h in range(24)]
        # Filter out NaN values
        error_values_by_hour = [errors[~np.isnan(errors)] for errors in error_values_by_hour]

        # Only include hours that have data (typically daylight hours)
        valid_hours = [h for h in range(24) if len(error_values_by_hour[h]) > 0]
        valid_error_values = [error_values_by_hour[h] for h in valid_hours]

        # Create box plot
        box_parts = ax.boxplot(
            valid_error_values,
            positions=valid_hours,
            patch_artist=True,
            widths=0.7,
            whis=whis,
            showfliers=showfliers,  # Hide outliers for cleaner visualization
            medianprops={'color': 'black'},
            boxprops={'facecolor': _color, 'alpha': 0.5}
        )

        ax.set_ylim(-750, 750)
        ax.set_xlabel(f"Local Time", fontsize=fontsize)
        ax.set_ylabel(f"{surfrad_var.upper()} Error (Wm$^{-2}$)", fontsize=fontsize)
        ax.set_title(
            # f"{str(_nwp_source.values)}",
            _nwp_source,
            fontsize=fontsize,
        )
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels(range(0, 25, 6))
        ax.tick_params(axis="both", labelsize=fontsize - 2)

        stddevs_da = xr.DataArray(stddevs, dims="nwp_source", coords={"nwp_source":nwp_sources_analyze})
        rmse_da = xr.DataArray(rmses, dims="nwp_source", coords={"nwp_source":nwp_sources_analyze})
        mbe_da = xr.DataArray(mbes, dims="nwp_source", coords={"nwp_source":nwp_sources_analyze})
        stddevs_da = stddevs_da.assign_coords(variable="stddev").expand_dims("variable")
        rmse_da = rmse_da.assign_coords(variable="rmse").expand_dims("variable")
        mbe_da = mbe_da.assign_coords(variable="mbe").expand_dims("variable")

    return xr.combine_by_coords([stddevs_da, rmse_da, mbe_da])


def compute_nwp_ensemble_averages(
    ds: xr.Dataset,
    ens_mems: list,
    ens_name: str,
    count_threshold: int,
):
    """
    Create a new NWP product by averaging over specified products.
    Then mask out areas of the average where data is missing.

    Inputs:
    ds: xarray Dataset
        Data to operate on.
    ens_mems: list
        list of ensemble members identified by their "nwp_source" dimension.
    ens_name: string
        name for the new forecast product.
    count_threshold: integer
        integer specifying how many non-nans are needed to
        compute the new field. Otherwise it will be masked.

    Outputs:
    ds_out: xarray Dataset
        Data with the new mean value added along "nwp_source"
    """

    ens_subset = ds.sel(nwp_source=ens_mems)

    # Mask the ensemble means if there are less than "count_threshold" members in them.
    mems_empty = np.isnan(ens_subset).all(dim="valid_time").sum(dim="nwp_source")
    mems_mask = (len(ens_mems) - mems_empty) >= count_threshold

    ens_ds = ens_subset.mean(dim="nwp_source").where(mems_mask)

    ds_out = xr.merge(
        [
            ds,
            ens_ds.assign_coords(nwp_source=ens_name).expand_dims("nwp_source"),
        ]
    )
    # Modify the mask so it broadcasts against the correct nwp_source values.
    out_mask = np.bitwise_or(mems_mask, (ds_out.nwp_source != ens_name))

    return ds_out, out_mask


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
    Afactor = 8.0

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

    surfrad_sitenames_nondesert = ["fpk", "sxf", "bon", "gwn", "psu"] # "dra", "tbl",
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
    stddev_list = []
    for surfrad_sitename in surfrad_sitenames_nondesert:
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

        # Add an ensemble average for the RRFS forecasts.
        rrfs_mems = [i for i in list(nwp_dayahead_ds.nwp_source.values) if i != "hrrr"]
        nwp_dayahead_ds, rrfsmems_mask = compute_nwp_ensemble_averages(
            nwp_dayahead_ds,
            rrfs_mems,
            ens_name="rrfs_ensmean",
            count_threshold=6,
        )
        nwp_dayahead_ds, rrfsmems_mask = compute_nwp_ensemble_averages(
            nwp_dayahead_ds,
            list(nwp_dayahead_ds.nwp_source.values),
            ens_name="all_ensmean",
            count_threshold=6,
        )

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

        empty_mask = obs_regime == REGIME_EMPTY
        clear_mask = obs_regime == REGIME_CLEAR
        broken_mask = obs_regime == REGIME_BROKEN
        cloudy_mask = obs_regime == REGIME_CLOUDY

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
        # Use the HRRR forecast to predict the regime.
        hrrr_csi = fcast_csi.sel(nwp_source="hrrr")
        n_hours = hrrr_csi.count(dim="hour")
        csi_mean = hrrr_csi.mean(dim="hour")
        csi_stddev = hrrr_csi.std(dim="hour")
        P_cloudy = -1 * (csi_mean + broken_stddev_weight * csi_stddev - broken_thresh)

        hrrr_weight = sigmoid(P_cloudy, A=Afactor, alpha=8)

        fcast_regime, fcast_csi_mean, fcast_csi_stddev, fcast_n_hours = classify_cloud_regime(
            fcast_csi.sel(nwp_source="hrrr"),
            clear_thresh=clear_thresh,
            broken_thresh=broken_thresh + broken_thresh_hrrr_correction,
            broken_stddev_weight=broken_stddev_weight,
            min_sunlit_hours=min_sunlit_hours,
        )

        # Set the increased HRRR weight to 0 when the day classification is clear
        hrrr_weight_masked = hrrr_weight.where(~(fcast_regime==REGIME_CLEAR))
        # weights = xr.ones_like(fcast_reindexed.nwp_source)
        # Apply the HRRR weight to the HRRR product only, and leave the other products at 1.0
        # Remove the averages from the reindexed dataset so that the weighted average is computed only over the individual products.
        raw_nwp_reindexed = fcast_reindexed.sel(nwp_source=[i for i in fcast_reindexed.nwp_source.values if i not in ["rrfs_ensmean", "all_ensmean"]])
        weights = xr.where(raw_nwp_reindexed.nwp_source != "hrrr", 1, 1 + hrrr_weight_masked).fillna(1.0)

        # Map the weights back to valid_time time coordinates, so that the weighted average is computed over the same time dimension as the forecast regime.
        weights_hourly = weights.broadcast_like(raw_nwp_reindexed)
        weights_valid_time = undo_shift_and_reindex_time(weights_hourly, "valid_time", utc_shift=utc_shift, year=year_start).fillna(1.0)

        # Now compute the new weighted average and add it to the nwp_dayahead_ds dataset.
        hrrr_informed_weighted_mean = nwp_dayahead_ds.sel(nwp_source=weights.nwp_source).weighted(weights_valid_time).mean("nwp_source")
        hrrr_informed_weighted_mean = hrrr_informed_weighted_mean.expand_dims(nwp_source=["weighted_mean"])
        combine_nwp_ds = xr.concat([nwp_dayahead_ds, hrrr_informed_weighted_mean], dim="nwp_source")

        # # Compute the weighted average of the forecast regime across all products
        # weighted_mean = fcast_reindexed.weighted(weights).mean("nwp_source")
        # weighted_mean = weighted_mean.expand_dims(nwp_source=["weighted_mean"])
        # combine_nwp_ds = fcast_reindexed.copy()
        # combine_nwp_ds = xr.concat([combine_nwp_ds, weighted_mean], dim="nwp_source")

        nwp_sources_plot = ["all_ensmean", "weighted_mean"]
        showfliers = False
        # "nwp_sources_analyze" must match the order of the nwp_source dimension in the nwp_dayahead_ds
        nwp_sources_analyze = ["hrrr", "rrfs_control", "rrfs_ensmean", "all_ensmean", "rrfs_mem0001", "rrfs_mem0002", "rrfs_mem0003", "rrfs_mem0004", "rrfs_mem0005", "weighted_mean"]

        # Plot the same error whisker plots while masking for different cloud conditions.
        all_stddevs = plot_diurnal_whiskerplot(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=combine_nwp_ds,
            nwp_sources_plot=nwp_sources_plot,
            nwp_sources_analyze=nwp_sources_analyze,
            nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
            surfrad_masks=surfrad_masks,
            utc_shift=utc_shift,
            # meteo_mask=clear_mask,
            # orientation="horizontal",
            # num_panels_dim=1,
            whis=(2.5, 97.5),
            showfliers=showfliers,
            # axs=_axs1,
        )
        all_stddevs.name = "all"
        clear_stddevs = plot_diurnal_whiskerplot(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=combine_nwp_ds,
            nwp_sources_plot=nwp_sources_plot,
            nwp_sources_analyze=nwp_sources_analyze,
            nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
            surfrad_masks=surfrad_masks,
            utc_shift=utc_shift,
            meteo_mask=clear_mask,
            orientation="horizontal",
            num_panels_dim=1,
            whis=(2.5, 97.5),
            showfliers=showfliers,
            # axs=_axs1,
        )
        clear_stddevs.name = "clear"

        # Broken clouds (partly cloudy)
        broken_stddevs = plot_diurnal_whiskerplot(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=combine_nwp_ds,
            nwp_sources_plot=nwp_sources_plot,
            nwp_sources_analyze=nwp_sources_analyze,
            nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
            surfrad_masks=surfrad_masks,
            utc_shift=utc_shift,
            meteo_mask=broken_mask,
            orientation="horizontal",
            num_panels_dim=1,
            whis=(2.5, 97.5),
            showfliers=showfliers,
            # axs=_axs2,
        )
        broken_stddevs.name = "broken"

        # Overcast
        cloudy_stddevs = plot_diurnal_whiskerplot(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=combine_nwp_ds,
            nwp_sources_plot=nwp_sources_plot,
            nwp_sources_analyze=nwp_sources_analyze,
            nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
            surfrad_masks=surfrad_masks,
            utc_shift=utc_shift,
            meteo_mask=cloudy_mask,
            orientation="horizontal",
            num_panels_dim=1,
            whis=(2.5, 97.5),
            showfliers=showfliers,
            # axs=_axs3,
        )
        cloudy_stddevs.name = "cloudy"

        error_stddevs_conditions = xr.merge([all_stddevs, clear_stddevs, broken_stddevs, cloudy_stddevs]).assign_coords(location=surfrad_sitename).expand_dims("location")
        stddev_list.append(error_stddevs_conditions)
        # break

    error_stddevs_conditions = xr.combine_by_coords(stddev_list)
    error_stddevs_conditions.to_dataframe().to_csv(f"data/figure_outputs/error_all_cloud_conditions_HRRRweighted_Afactor_{Afactor}.csv")

    # %% 
    # Verify against the current error analysis data
    current_data = pd.read_csv("data/figure_outputs/error_all_cloud_conditions.csv")

    # %%
