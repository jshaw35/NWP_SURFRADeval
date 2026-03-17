"""
Compare NWP predictions with surfrad records. Specifically, create
figures for the summary presentation on RRFS performance.

This code should be run after pre-processing data into timeseries
format using preprocess_to_timeseries.py

"""

# %%

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import seaborn as sns
import glob


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


def nwp_timeseries_comparison(
    nwp_ds: xr.DataArray,
    surfrad_ds: xr.DataArray,
    colors: list,
    time_dim: str = "valid_time",
    ax: plt.axis = None,
):
    """
    Simple script to visualize observations and forecasts as
    time series.

    Inputs:
    nwp_ds: xarray DataArray
        DataArray containing multiple forecasts along the "nwp_source" dimension.
    surfrad_ds: xarray DataArray
        DataArray containing surfrad observations.
    colors: list
        list of strings corresponding to lineplot colors.
    time_dim: string
        identifier for the time dimension for both datasets.
    ax: matplotlib axis
        optional matplotlib axis to plot along.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if surfrad_ds is not None:
        ax.plot(
            surfrad_ds[time_dim],
            surfrad_ds,
            color="black",
            label="SURFRAD Obs.",
            linestyle="dashed",
        )

    for _nwp_source, _color in zip(nwp_ds.nwp_source, colors):

        nwp_data = nwp_ds.sel(nwp_source=_nwp_source)
        label_str = str(_nwp_source.values)
        ax.plot(
            nwp_data[time_dim],
            nwp_data,
            color=_color,
            label=label_str,
        )

    if "fig" in locals():
        return fig, ax
    else:
        return ax


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


def plot_bulk_metrics(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.Dataset,
    surfrad_masks: xr.Dataset,
    colors: list = sns.color_palette("colorblind"),
    save_figs: bool = False,
):
    """
    Produce a bar plot showing mean absolute error (MAE) and
    root-mean-square error (RMSE) for the clear-sky index (CSI)
    and the normal radiation field.

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
    colors: list
        list of objects that matplotlib can use as color input.
    save_figs: boolean
        boolean indicating whether the figure should be saved.

    Outputs:
    None. Figure is produced and optionally saved.
    """

    surfrad_clearsky_var = f"clearsky_{surfrad_var}"

    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask)
    drop_mask = ~np.isnan(nwp_ds[nwp_var]).all(dim="valid_time")
    nwp_data = (
        nwp_ds[nwp_var].isel(nwp_source=drop_mask).where(surfrad_masks[surfrad_var])
    )

    # Masking out solar elevation angles <10 degrees.
    surfrad_mask = surfrad_data.zenith < 80

    # Compute the clear-sky index error.
    error = (nwp_data - surfrad_data).load()
    error_csi = error / surfrad_data[surfrad_clearsky_var]
    error_csi = error_csi.where(surfrad_mask)

    # Compute error in the GHI forecast, excluding low insolation times.
    mae = np.abs(error).where(surfrad_mask).mean(dim="valid_time")
    rmse = np.sqrt(
        (error**2).where(surfrad_mask).mean(dim="valid_time")
    )

    # Compute error in the clear-sky index. No masking by insolation yet.
    mae_csi = np.abs(error_csi).mean(dim="valid_time")
    rmse_csi = np.sqrt(
        (error_csi**2).mean(dim="valid_time")
    )

    # Plot all error metrics together
    error_metrics = [
        mae_csi,
        rmse_csi,
        mae,
        rmse,
    ]

    error_labels = [
        f"Mean Absolute Error in Clear-sky {surfrad_var.upper()} Index",
        f"Root-Mean-Square Error in Clear-sky {surfrad_var.upper()} Index",
        f"Mean Absolute {surfrad_var.upper()} Error {surfrad_var.upper()} (Wm$^{-2}$)",
        f"Root-Mean-Square {surfrad_var.upper()} Error (Wm$^{-2}$)",
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flat

    for _error_data, _label, _ax in zip(error_metrics, error_labels, axs):
        _ax.bar(_error_data.nwp_source, _error_data, color=colors)
        _ax.set_ylabel(_label)
        _ax.tick_params(labelrotation=30)

    if save_figs:
        save_filename = f"NWP_error_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def plot_time_series(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.Dataset,
    surfrad_masks: xr.Dataset,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    colors: list = sns.color_palette("colorblind"),
    plot_error: bool = False,
    save_figs: bool = False,
):
    """
    Visualize the forecast and observations by comparing
    time series over several weeks.

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
    datetime_start: pandas Timestamp
        Identifier for time series start.
    datetime_end: pandas Timestamp
        Identifier for time series end.
    colors: list
        list of objects that matplotlib can use as color input.
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    orientation: string
        How to orient the figure panels: {"horizontal","vertical"}
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    plot_error: boolean
        boolean to indicate if the plot should show absolute values or error.

    Outputs:
    None. Figure is produced and optionally saved.
    """

    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    nwp_data = nwp_ds[nwp_var].where(surfrad_masks[surfrad_var])
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask)

    nrow = int(np.ceil((datetime_end - datetime_start).days / 14))
    fig, axs = plt.subplots(nrow, 1, figsize=(15, 3 * nrow))
    axs = axs.flat

    for i, _ax in enumerate(axs):

        row_tstart = datetime_start + i * timedelta(days=14)
        timeseries_slice = slice(
            row_tstart,
            row_tstart + timedelta(days=14),
        )
        nwp_timeseries_subset = nwp_data.sel(
            valid_time=timeseries_slice,
            nwp_source=["hrrr", "rrfs_control"]
        )
        surfrad_timeseries_subset = surfrad_data.sel(valid_time=timeseries_slice)

        if plot_error:
            error_timeseries_subset = nwp_timeseries_subset - surfrad_timeseries_subset
            ax = nwp_timeseries_comparison(
                nwp_ds=error_timeseries_subset,
                surfrad_ds=None,
                colors=colors,
                time_dim="valid_time",
                ax=_ax,
            )
        else:
            ax = nwp_timeseries_comparison(
                nwp_ds=nwp_timeseries_subset,
                surfrad_ds=surfrad_timeseries_subset,
                colors=colors,
                time_dim="valid_time",
                ax=_ax,
            )
        ax.tick_params(labelrotation=0)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{surfrad_var.upper()} (Wm$^{-2}$)")
        ax.set_ylim(-10, 1050)
        ax.legend()

    if save_figs:
        if plot_error:
            save_filename = f"TimeSeriesErrorComparison_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        else:
            save_filename = f"TimeSeriesComparison_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def plot_composites(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.Dataset,
    surfrad_masks: xr.Dataset,
    utc_shift: int,
    colors: list = sns.color_palette("colorblind"),
    fontsize: float = 16,
    save_figs: bool = False,
):
    """
    Visualize the forecast and observations by compositing
    time series into daily values and errors.

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
    colors: list
        list of objects that matplotlib can use as color input.
    orientation: string
        How to orient the figure panels: {"horizontal","vertical"}
    save_figs: boolean
        boolean indicating whether the figure should be saved.

    Outputs:
    None. Figure is produced and optionally saved.
    """

    nwp_mask = nwp_masks[nwp_var].sum(dim="time").load()
    nwp_data = nwp_ds[nwp_var].where(surfrad_masks[surfrad_var])
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask).load()

    nwp_data = nwp_data.assign_coords(
        utc_shift=nwp_data.valid_time + np.timedelta64(utc_shift, "h")
    )
    surfrad_data = surfrad_data.assign_coords(
        utc_shift=surfrad_data.valid_time + np.timedelta64(utc_shift, "h")
    )

    # Composite to show average days.
    nwp_daily_composite = nwp_data.groupby("utc_shift.hour").mean()
    surfrad_daily_composite = surfrad_data.groupby("utc_shift.hour").mean()
    error_daily_composite = (nwp_data - surfrad_data).groupby("utc_shift.hour").mean()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs = axs.flat

    # Plot the average daily fields.
    ax = axs[0]
    ax.plot(
        surfrad_daily_composite.hour,
        surfrad_daily_composite,
        label="SURFRAD obs.",
        color="black",
        alpha=1.0,
        linestyle="dashed",
    )
    for _nwp_source, _color in zip(nwp_daily_composite.nwp_source, colors):

        _data = nwp_daily_composite.sel(nwp_source=_nwp_source)
        if np.isnan(_data).all():
            continue
        ax.plot(
            _data.hour,
            _data,
            label=str(_nwp_source.values),
            color=_color,
            alpha=0.5,
            linestyle="solid",
        )
    ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
    ax.set_ylabel(f"{surfrad_var.upper()} (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Daily {surfrad_var.upper()} Composite", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.legend()

    # Plot the average daily error.
    ax = axs[1]
    for _nwp_source, _color in zip(error_daily_composite.nwp_source, colors):

        _data = error_daily_composite.sel(nwp_source=_nwp_source)
        if np.isnan(_data).all():
            continue
        ax.plot(
            _data.hour,
            _data,
            label=str(_nwp_source.values),
            color=_color,
            alpha=0.5,
            linestyle="solid",
        )
    ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
    ax.set_ylabel(f"{surfrad_var.upper()} Error (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Daily {surfrad_var.upper()} Error Composite", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.legend()

    if save_figs:
        save_filename = (
            f"CompositeError_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        )
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


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


def plot_daily_values(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.DataArray,
    surfrad_masks: xr.DataArray,
    utc_shift: int,
    meteo_mask: xr.DataArray = None,
    meteo_name: str = None,
    fontsize: float = 16,
    orientation: str = "horizontal",
    save_figs: bool = False,
):
    """
    Plot individual daily values over each other to visualize
    the distribution of values.

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
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    """

    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    nwp_data = nwp_ds[nwp_var].where(surfrad_masks[surfrad_var])
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask).load()

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

    if orientation == "horizontal":
        fig, axs = plt.subplots(
            2,
            int(np.ceil(len(nwp_error_reindexed.nwp_source) / 2)),
            figsize=(6 * np.ceil(len(nwp_error_reindexed.nwp_source) / 2), 12),
        )
        fig.subplots_adjust(wspace=0.38, hspace=0.25)

    if orientation == "vertical":
        fig, axs = plt.subplots(
            int(np.ceil(len(nwp_error_reindexed.nwp_source) / 2)),
            2,
            figsize=(15, 6 * np.ceil(len(nwp_error_reindexed.nwp_source) / 2)),
        )
        fig.subplots_adjust(wspace=0.3, hspace=0.25)
    axs = axs.flat

    # Plot the observations in the first panel
    ax = axs[0]
    obs_data = surfrad_reindexed
    if meteo_mask is not None:
        obs_data = obs_data.where(meteo_mask)
    colors = [sns.color_palette("colorblind")[0] for i in axs]
    for _DOY in obs_data.dayofyear:

        _obs = obs_data.sel(dayofyear=_DOY)
        ax.plot(
            _obs.hour,
            _obs,
            color="black",
            alpha=0.25,
            linestyle="solid",
            linewidth=1,
        )
    ax.set_ylim(-10, 1100)
    ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
    ax.set_ylabel(f"{surfrad_var.upper()} (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Observed {surfrad_var.upper()}", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    for _nwp_source, ax, _color in zip(nwp_error_reindexed.nwp_source, axs[1:], colors):

        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        if meteo_mask is not None:
            error_data = error_data.where(meteo_mask)
        rmse = np.sqrt((error_data**2).mean())

        for _DOY in error_data.dayofyear:

            _error = error_data.sel(dayofyear=_DOY)
            ax.plot(
                _error.hour,
                _error,
                color=_color,
                alpha=0.25,
                linestyle="solid",
                linewidth=1,
            )
        ax.set_ylim(-1000, 1000)
        ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
        ax.set_ylabel(f"{surfrad_var.upper()} Error (Wm$^{-2}$)", fontsize=fontsize)
        ax.set_title(
            f"{str(_nwp_source.values)} RMSE: {rmse:.2f}",
            fontsize=fontsize,
        )
        ax.tick_params(axis="both", labelsize=fontsize - 2)

    if save_figs:
        save_filename = f"DailyErrorPanels_{meteo_name}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def plot_composite_spread(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.DataArray,
    surfrad_masks: xr.DataArray,
    utc_shift: int,
    meteo_mask: xr.DataArray = None,
    meteo_name: str = None,
    fontsize: float = 16,
    orientation: str = "horizontal",
    num_panels_dim: int = 2,
    save_figs: bool = False,
):
    """
    Visualize the error and show the spread.

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
    save_figs: boolean
        boolean indicating whether the figure should be saved.
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
    colors = [sns.color_palette("colorblind")[0] for i in axs]
    for _nwp_source, ax, _color in zip(nwp_error_reindexed.nwp_source, axs, colors):
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        # Only compute the RMSE over sunlit timesteps for consistency with
        # other error metrics.
        rmse = np.sqrt((error_data.where(surfrad_reindexed.zenith < 80) ** 2).mean())
        mean_error = error_data.mean(dim="dayofyear")
        # min_error = mean_error - 1.645 * error_data.std(dim="dayofyear")
        # max_error = mean_error + 1.645 * error_data.std(dim="dayofyear")
        min_error = error_data.quantile(q=0.1, dim="dayofyear")
        max_error = error_data.quantile(q=0.9, dim="dayofyear")
        ax.plot(
            mean_error.hour,
            mean_error,
            color=_color,
            alpha=0.5,
            linestyle="solid",
            linewidth=0.75,
            label="Mean Error",
        )
        ax.fill_between(
            min_error.hour,
            min_error,
            max_error,
            alpha=0.2,
            label="90% Confidence Interval",
        )
        ax.set_ylim(-750, 750)
        ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
        ax.set_ylabel(f"{surfrad_var.upper()} Error (Wm$^{-2}$)", fontsize=fontsize)
        ax.set_title(
            f"{str(_nwp_source.values)} RMSE: {rmse:.0f}",
            fontsize=fontsize,
        )
        ax.tick_params(axis="both", labelsize=fontsize - 2)
        ax.legend()

    if save_figs:
        save_filename = f"DailyErrorPanels_{meteo_name}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def plot_rank_histogram(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.DataArray,
    surfrad_masks: xr.DataArray,
    ens_members: list,
    meteo_mask: xr.DataArray = None,
    meteo_name: str = None,
    fontsize: float = 16,
    save_figs: bool = False,
):
    """
    Quantify the spread of the RRFS members using a rank histogram.
    Optionally use a meteorological conditions masks as well.

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
    ens_members: list
        list of string identifiers for the "nwp_source" dimension.
    meteo_mask: xarray Dataset
        Mask for meteorological conditions.
    meteo_name: string
        String appended to output file name to identify the mask.
    fontsize: float
        float for determining figure fontsizes
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    """

    nwp_data = nwp_ds[nwp_var].sel(nwp_source=ens_members)
    surfrad_data = surfrad_ds[surfrad_var].load()

    # Apply masking by availability, sun position, and meteorology.
    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    surfrad_mask = surfrad_masks[surfrad_var]
    zenith_mask = surfrad_ds.zenith < 80
    all_mask = nwp_mask & surfrad_mask & zenith_mask
    if meteo_mask is not None:
        # Need to convert the mask to the valid_time dimension.
        meteo_days = meteo_mask.dayofyear.isel(dayofyear=meteo_mask)
        meteo_mask = [i.values in meteo_days for i in all_mask.valid_time.dt.dayofyear]
        all_mask = all_mask & meteo_mask
    else:
        meteo_name = None

    # Compute rank for each timestep.
    rank = (surfrad_data > nwp_data).sum(dim="nwp_source").where(all_mask)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bins = np.arange(-0.5, len(ens_members) + 0.51, 1)
    ax.hist(
        rank,
        bins=bins,
        color="blue",
        alpha=0.8,
    )
    ax.hlines(
        all_mask.sum() / (len(ens_members) + 1),
        bins[0],
        bins[-1],
        color="black",
        linestyle="dashed",
    )
    ax.set_xlabel(
        "Observations Rank",
        fontsize=fontsize,
    )
    ax.set_ylabel("Count",fontsize=fontsize)
    if meteo_mask:
        title_str = f"{surfrad_var.upper()} rank histogram at {surfrad_sitename} during {meteo_name}"
    else:
        title_str = f"{surfrad_var.upper()} rank histogram at {surfrad_sitename}"
    ax.set_title(title_str, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 4)

    if save_figs:
        save_filename = f"RankHistogram_{meteo_name}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def plot_rank_histogram_check(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.DataArray,
    surfrad_masks: xr.DataArray,
    ens_members: list,
    rank_height: float = 1030,
    save_figs: bool = False,
):
    """
    Plot the highest and lowest ranks along with a time series
    in order to interrogate the rank histogram results.

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
    ens_members: list
        list of string identifiers for the "nwp_source" dimension.
    rank_height: float
        float used to set the height of the rank indicators.
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    """

    nwp_data = nwp_ds[nwp_var].sel(nwp_source=ens_members)
    surfrad_data = surfrad_ds[surfrad_var].load()

    # Apply masking by availability, sun position, and meteorology.
    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    surfrad_mask = surfrad_masks[surfrad_var]
    zenith_mask = surfrad_ds.zenith < 80
    all_mask = nwp_mask & surfrad_mask & zenith_mask

    rank = (surfrad_data > nwp_data).sum(dim="nwp_source").where(all_mask)

    fig, ax = plt.subplots(1, 1, figsize=(60, 7))
    ax.plot(
        surfrad_data.valid_time,
        surfrad_data,
        linestyle="dashed",
        color="black",
    )

    for _nwp_source in nwp_data.nwp_source:
        _data = nwp_data.sel(nwp_source=_nwp_source)
        ax.plot(
            _data.valid_time,
            _data,
        )

    ax.plot(
        rank.valid_time,
        np.where(rank==6, rank_height*np.ones_like(rank), np.nan),
        color="pink",
    )
    ax.plot(
        rank.valid_time,
        np.where(rank==0, rank_height*np.ones_like(rank), np.nan),
        color="black",
    )

    if save_figs:
        save_filename = f"RankHistogramTimeseries_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


def toy_rank_histogram(
    mean: float = 50,
    stddev: float = 10,
    mean_bias: float = 0,
    stddev_bias: float = 3,
    ens_size: int = 6,
    draws: int = 1000,
    random_seed: int = None,
    file_str: str = None,
    ax: plt.axis = None,
    fontsize: float = 16,
    save_figs: bool = False,
):
    """
    Create example rank histograms a la Suarez-Gutierrez et al. (2021)
    Exploiting large ensembles for a better yet simpler climate model evaluation, Climate Dynamics
    Just uses an underlying gaussian distribution for simplicity.

    Inputs:
    mean: float
        The mean values of the observations.
    stddev: float
        The standard deviation of the observations.
    mean_bias: float
        The mean bias of the ensemble.
    stddev_bias: float
        The bias in the ensemble standard deviation.
    ens_size: integer
        The number of ensemble members in the ensemble.
    draws: integer
        The number of draw (samples) from the underlying PDFs.
    random_seed: integer
        If specified, use a random seed for reproducible figures.
    file_str: string
        String identifier to be used when saving a figure.
    ax: matplotlib.pyplot axis
        Optionally include an axis to plot on.
    fontsize: float
        Fontsize to use in the figure.
    save_figs: boolean
        Boolean indicating if a string should be saved.

    Outputs:
    None, the figure is optionally saved if requested.

    """

    # Set the random seed for repeatability:
    if random_seed is not None:
        np.random.seed(random_seed)

    # Compute rank for each timestep.
    obs = mean + stddev * np.random.standard_normal((draws, 1))
    ens = (
        mean
        + mean_bias
        + (stddev + stddev_bias) * np.random.standard_normal((draws, ens_size))
    )
    rank = (obs > ens).sum(axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    bin_centers = np.arange(0, ens_size + 0.01, 1)
    bins_edges = np.arange(-0.5, ens_size + 0.51, 1)
    ax.hist(
        rank,
        bins=bins_edges,
        color="blue",
        alpha=0.8,
    )
    ax.hlines(
        draws / (ens_size + 1) * np.ones(len(bins_edges)),
        bins_edges[0],
        bins_edges[-1],
        color="black",
        linestyle="dashed",
    )
    ax.set_xlabel(
        "Rank",
        fontsize=fontsize,
    )
    ax.set_ylabel("Count", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 4)
    ax.set_xticks(bin_centers)

    if save_figs:
        save_filename = f"RankHistogram_{file_str}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    save_figs = False
    surfrad_sitename = "gwn"

    surfrad_var = "ghi"
    surfrad_clearsky_var = f"clearsky_{surfrad_var}"
    nwp_var = "dswrf"
    utc_shift = -6

    save_dir = os.path.join(
        "figures",
        surfrad_sitename,
    )

    if save_figs and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    nwp_datavars = [
        "dswrf",
        "vbdsf",
        "vddsf",
    ]

    year_start = 2024
    month_start = 3
    day_start = 1

    year_end = 2024
    month_end = 7
    day_end = 8

    data_datetime_start = pd.Timestamp(year_start, month_start, day_start)
    data_datetime_end = pd.Timestamp(year_end, month_end, day_end)

    datestring = data_datetime_start.strftime("%Y%m%d_") + data_datetime_end.strftime(
        "%Y%m%d"
    )

    surfrad_ds, surfrad_masks = load_and_mask_surfrad(
        load_path,
        surfrad_sitename,
        data_datetime_start,
        data_datetime_end,
    )
    surfrad_ds = surfrad_ds.load()
    nwp_ds, nwp_masks = load_and_mask_nwp(
        load_path,
        data_datetime_start,
        data_datetime_end,
        nwp_datavars,
    )

    # Collapse the forecast time dimension so the forecasts
    # appear as a timeseries. Must select a <= 24 hour forecast
    # window so there are no forecast overlaps.
    nwp_dayahead_ds = nwp_ds.sum(dim="time", min_count=1).load()

    plot_bulk_metrics(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(location=surfrad_sitename),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename),
        surfrad_masks=surfrad_masks,
        save_figs=save_figs,
    )

    plot_composites(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=None,
        orientation="vertical",
        save_figs=save_figs,
    )
# %%
if __name__ == "__main__":
    plot_composite_spread(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=None,
        orientation="horizontal",
        num_panels_dim=1,
        save_figs=save_figs,
    )

    # A more involved example applying masks for different
    # meteorological conditions and creating separate plots.
    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    obs_mask = surfrad_masks[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].where(obs_mask)
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask).load()

    surfrad_reindexed = shift_and_reindex_time(
        surfrad_data,
        surfrad_var,
        "valid_time",
        utc_shift=utc_shift,
    )

    # Compute the clear-sky index
    clearsky_index_data = surfrad_reindexed / surfrad_reindexed[surfrad_clearsky_var]
    clearsky_index_data = clearsky_index_data.where(surfrad_reindexed.zenith < 80)

    daily_csi_mean = clearsky_index_data.mean(dim="hour")
    daily_csi_stddev = clearsky_index_data.std(dim="hour")

    # Somewhat adhoc classifications from looking at the data.
    empty_mask = daily_csi_mean.isnull()
    clear_mask = daily_csi_mean > 0.95
    broken_mask = np.bitwise_and(
        ~clear_mask,
        daily_csi_mean + 2 * daily_csi_stddev > 0.92
    )
    cloudy_mask = ~(clear_mask | broken_mask | empty_mask)

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=clear_mask,
        meteo_name="Clear",
        orientation="vertical",
        save_figs=save_figs,
    )

    plot_composite_spread(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=clear_mask.sel(location=surfrad_sitename).drop_vars("location"),
        meteo_name="Clear",
        orientation="horizontal",
        num_panels_dim=1,
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=broken_mask,
        meteo_name="Brokenclouds",
        orientation="vertical",
        save_figs=save_figs,
    )

    plot_composite_spread(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=broken_mask.sel(location=surfrad_sitename).drop_vars("location"),
        meteo_name="Brokenclouds",
        orientation="horizontal",
        num_panels_dim=1,
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=cloudy_mask,
        meteo_name="Cloudy",
        orientation="vertical",
        save_figs=save_figs,
    )

    plot_composite_spread(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(nwp_source=["hrrr", "rrfs_control"], location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=cloudy_mask.sel(location=surfrad_sitename).drop_vars("location"),
        meteo_name="Cloudy",
        orientation="horizontal",
        num_panels_dim=1,
        save_figs=save_figs,
    )

    rrfs_members = [
        "rrfs_control",
        "rrfs_mem0001",
        "rrfs_mem0002",
        "rrfs_mem0003",
        "rrfs_mem0004",
        "rrfs_mem0005",
    ]
    plot_rank_histogram(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds.sel(location=surfrad_sitename).drop_vars("location"),
        nwp_masks=nwp_masks.sel(location=surfrad_sitename).drop_vars("location"),
        surfrad_masks=surfrad_masks,
        ens_members=rrfs_members,
        save_figs=save_figs,
    )

    # Create example rank histograms a la Suarez-Gutierrez et al.
    # "Perfect" ensemble
    mean = 40
    stddev = 10
    ens_size = 6
    draws = 885

    mean_bias = 0
    stddev_bias = 0

    fig, axs = plt.subplots(
        1,
        4,
        figsize=(12, 2.5),
    )
    plt.subplots_adjust(wspace=0.27)
    axs = axs.flat

    ax = axs[0]
    toy_rank_histogram(
        mean=mean,
        stddev=stddev,
        mean_bias=mean_bias,
        stddev_bias=stddev_bias,
        ens_size=ens_size,
        draws=draws,
        ax=ax,
        random_seed=41,
        fontsize=10,
    )

    # Over-dispersed
    mean = 40
    stddev = 10

    mean_bias = 0
    stddev_bias = 3
    ax = axs[1]

    toy_rank_histogram(
        mean=mean,
        stddev=stddev,
        mean_bias=mean_bias,
        stddev_bias=stddev_bias,
        ens_size=ens_size,
        draws=draws,
        ax=ax,
        random_seed=42,
        fontsize=10,
    )

    # Under-dispersed
    mean = 40
    stddev = 10

    mean_bias = 0
    stddev_bias = -3
    ax = axs[2]

    toy_rank_histogram(
        mean=mean,
        stddev=stddev,
        mean_bias=mean_bias,
        stddev_bias=stddev_bias,
        ens_size=ens_size,
        draws=draws,
        ax=ax,
        random_seed=43,
        fontsize=10,
    )

    # Under-dispersed and biased high
    mean = 40
    stddev = 10

    mean_bias = 4
    stddev_bias = 0
    ax = axs[3]

    toy_rank_histogram(
        mean=mean,
        stddev=stddev,
        mean_bias=mean_bias,
        stddev_bias=stddev_bias,
        ens_size=ens_size,
        draws=draws,
        ax=ax,
        random_seed=43,
        fontsize=10,
    )

    # # Under-dispersed and biased high
    # mean = 40
    # stddev = 10

    # mean_bias = 3
    # stddev_bias = -5
    # ax = axs[3]

    # toy_rank_histogram(
    #     mean=mean,
    #     stddev=stddev,
    #     mean_bias=mean_bias,
    #     stddev_bias=stddev_bias,
    #     ens_size=ens_size,
    #     draws=draws,
    #     ax=ax,
    #     random_seed=43,
    #     fontsize=10,
    # )

    for _ax in axs:
        _ax.set_ylim(0, 350)
    # %%

    # Extra plots to visualize the RRFS ensemble.
    surfrad_var = "ghi"
    nwp_var = "dswrf"
    surfrad_data = surfrad_ds[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].sel(location=surfrad_sitename) #.drop_sel(nwp_source=["hrrr"])

    # Shift so the time is UTC - 6 hours so days contain all sunlit timesteps.
    nwp_data["valid_time"] = nwp_data["valid_time"] - np.timedelta64(6, "h")
    surfrad_data["valid_time"] = surfrad_data["valid_time"] - np.timedelta64(6, "h")

    # day = "2024-03-21"
    day = "2024-03-07"
    fontsize = 16

    # Plot just the RRFS control
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_title(day)
    _nwp_day = nwp_data.sel(valid_time=day)
    _surfrad_day = surfrad_data.sel(valid_time=day)

    ax.plot(
        _surfrad_day.valid_time.dt.hour,
        _surfrad_day,
        label="SURFRAD",
        color="black",
        alpha=0.8,
        linestyle="solid",
        linewidth=1,
    )

    for _nwp_source in _nwp_day.nwp_source:

        _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
        if _nwp_source != "rrfs_control":
            continue
        ax.plot(
            _forecast_data.valid_time.dt.hour,
            _forecast_data,
            label=str(_nwp_source.values),
            color="blue",
            alpha=0.8,
            linestyle="solid",
            linewidth=1,
        )

    ax.set_ylim(-10, 1000)
    ax.set_xlabel("Hour of the Day", fontsize=fontsize)
    ax.set_ylabel("GHI (Wm$^{-2}$)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:2],
        ["Observations", "RRFS Control"],
        loc="upper left",
    )

    # Plot all RRFS forecast products
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_title(day)
    _nwp_day = nwp_data.sel(valid_time=day)
    _surfrad_day = surfrad_data.sel(valid_time=day)

    ax.plot(
        _surfrad_day.valid_time.dt.hour,
        _surfrad_day,
        label="SURFRAD",
        color="black",
        alpha=0.8,
        linestyle="solid",
        linewidth=1,
    )

    for _nwp_source in _nwp_day.nwp_source:

        _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
        if _nwp_source == "rrfs_control":
            alpha = 0.8
            linestyle = "solid"
            color = "blue"
        elif _nwp_source == "hrrr":
            alpha = 0.8
            linestyle = "solid"
            color = "red"
        else:
            alpha = 0.5
            linestyle = "dashed"
            color = "blue"
        ax.plot(
            _forecast_data.valid_time.dt.hour,
            _forecast_data,
            label=str(_nwp_source.values),
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            linewidth=1,
        )

    ax.set_ylim(-10, 1000)
    ax.set_xlabel("Hour of the Day", fontsize=fontsize)
    ax.set_ylabel("GHI (Wm$^{-2}$)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:4],
        ["Observations", "HRRR", "RRFS Control", "RRFS ensemble (n=5)"],
        loc="upper left",
    )


    # Plot all RRFS forecast products
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax, day in zip(axs, ["2024-03-07", "2024-03-21"]):

        ax.set_title(day)
        _nwp_day = nwp_data.sel(valid_time=day)
        _surfrad_day = surfrad_data.sel(valid_time=day)

        ax.plot(
            _surfrad_day.valid_time.dt.hour,
            _surfrad_day,
            label="SURFRAD",
            color="black",
            alpha=0.8,
            linestyle="solid",
            linewidth=1,
        )

        for _nwp_source in _nwp_day.nwp_source:

            _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
            if _nwp_source == "rrfs_control":
                alpha = 0.8
                linestyle = "solid"
                color = "blue"
            elif _nwp_source == "hrrr":
                alpha = 0.8
                linestyle = "solid"
                color = "red"
            else:
                alpha = 0.5
                linestyle = "dashed"
                color = "blue"
            ax.plot(
                _forecast_data.valid_time.dt.hour,
                _forecast_data,
                label=str(_nwp_source.values),
                color=color,
                alpha=alpha,
                linestyle=linestyle,
                linewidth=1,
            )

        ax.set_ylim(-10, 1000)
        ax.set_xlabel("Hour of the Day", fontsize=fontsize)
        ax.set_ylabel("GHI (Wm$^{-2}$)", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        handles, labels = ax.get_legend_handles_labels()
    axs[0].legend(
        handles[:4],
        ["Observations", "HRRR", "RRFS Control", "RRFS Members 1-5"],
        loc="upper left",
    )

# %%