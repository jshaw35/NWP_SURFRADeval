"""
Compare NWP predictions with surfrad records.

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


def plot_binned_error(
    error_data: xr.DataArray,
    group: xr.DataArray,
    bins: list,
    ax: plt.axis = None,
    color: str = None,
):
    """
    Group error data by an index and supplied bins.
    Plot the error histogram to visualize error.

    Inputs:

    error_data: xarray DataArray
        data to be binned and plotted.
    group: xarray index
        index to group by using groupby_bins.
    bins: list
        list of bin centers for binning and plotting.
    ax: matplotlib axis
        optional matplotlib axis to plot on.
    color: string
        string for histogram color.
    """

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(8, 6),
        )

    # Reduce across all dimensions other than the clear-sky GHI
    binned_error = error_data.groupby_bins(
        group,
        bins=bins,
    )
    histo_mean = binned_error.mean(...)
    # histo_count = binned_error.count(...)

    # ^The single line above seems to work but I don't
    # under why since we want to index by insolation.
    binned_index = group.groupby_bins(
        group,
        bins=bins,
    )
    histo_count = binned_index.count(...)

    fontsize = 16
    count_color = "black"

    ax.stairs(
        histo_mean,
        bins,
        color=color,
        linestyle="solid",
        linewidth=3,
        label=str(error_data.nwp_source.values),
        fill=True,
    )

    axb = ax.twinx()
    axb.stairs(
        histo_count,
        bins,
        color=count_color,
        linestyle="dashed",
        linewidth=3,
        label="Bin Count",
    )
    axb.set_ylabel("Bin Counts", fontsize=fontsize, color=count_color)
    axb.tick_params(axis="y", colors=count_color)

    if "fig" in locals():
        return fig, ax
    else:
        return ax


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
    surfrad_sitename: str,
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

    datestring = datetime_start.strftime("%Y%m%d_") + datetime_end.strftime("%Y%m%d")
    load_dir = os.path.join(
        load_path,
        datestring,
    )

    filename = f"????????_????????_nwp_{surfrad_sitename}.nc"
    filenames = glob.glob(os.path.join(load_dir, filename))
    filenames.sort()
    ds = xr.open_mfdataset(filenames, combine="nested")
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
        f"Mean Absolute Error in Clear-sky {surfrad_var} Index",
        f"Root-Mean-Square Error in Clear-sky {surfrad_var} Index",
        f"Mean Absolute {surfrad_var} Error {surfrad_var} (Wm$^{-2}$)",
        f"Root-Mean-Square {surfrad_var} Error (Wm$^{-2}$)",
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


def plot_histogram_error(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.Dataset,
    surfrad_masks: xr.Dataset,
    colors: list = sns.color_palette("colorblind"),
    orientation: str = "horizontal",
    bins: list = [-1, 0, 20, 100, 500, 750, 1000],
    fontsize: float = 16,
    save_figs: bool = False,
):
    """
    Produce a histogram plot for each NWP forecast plotting the error
    in a radiation field against the equivalent clear-sky radiation field.

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
    orientation: string
        How to orient the figure panels: {"horizontal","vertical"}
    save_figs: boolean
        boolean indicating whether the figure should be saved.
    bins: list
        list of bin boundaries for the histogram.
    fontsize: float
        float for setting fonts within the figure.

    Outputs:
    None. Figure is produced and optionally saved.
    """

    nwp_mask = nwp_masks[nwp_var].sum(dim="time")
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask)
    drop_mask = ~np.isnan(nwp_ds[nwp_var]).all(dim="valid_time")
    nwp_data = (
        nwp_ds[nwp_var].isel(nwp_source=drop_mask).where(surfrad_masks[surfrad_var])
    )

    surfrad_clearsky_var = f"clearsky_{surfrad_var}"

    error = (nwp_data - surfrad_data).load()

    if orientation == "horizontal":
        fig, axs = plt.subplots(
            2,
            int(np.ceil(len(error.nwp_source) / 2)),
            figsize=(6 * np.ceil(len(error.nwp_source) / 2), 12),
        )
        fig.subplots_adjust(wspace=0.38, hspace=0.25)

    if orientation == "vertical":
        fig, axs = plt.subplots(
            int(np.ceil(len(error.nwp_source) / 2)),
            2,
            figsize=(15, 6 * np.ceil(len(error.nwp_source) / 2)),
        )
        fig.subplots_adjust(wspace=0.3, hspace=0.25)
    axs = axs.flat

    for _nwp_source, ax, _color in zip(error.nwp_source, axs, colors):

        _error = np.abs(error.sel(nwp_source=_nwp_source))

        ax = plot_binned_error(
            _error,
            _error[surfrad_clearsky_var],
            bins=bins,
            color=_color,
            ax=ax,
        )

        ax.set_xscale("log")
        ax.set_xlim((1, 1000))
        ax.set_ylim((0, 250))
        ax.set_xlabel(f"Clear-sky {surfrad_var}", fontsize=fontsize)
        ax.set_ylabel(f"{surfrad_var} Error (Wm$^{-2}$)", fontsize=fontsize)
        ax.set_title(str(_nwp_source.values), fontsize=fontsize)

    if save_figs:
        save_filename = (
            f"clearsky{surfrad_var}_histo_{datestring}_{surfrad_sitename}.png"
        )
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
        ax.set_ylabel(f"{surfrad_var} (Wm$^{-2}$)")
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
    ax.set_ylabel(f"{surfrad_var} (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Daily {surfrad_var} Composite", fontsize=fontsize)
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
    ax.set_ylabel(f"{surfrad_var} Error (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Daily {surfrad_var} Error Composite", fontsize=fontsize)
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
    ax.set_ylabel(f"{surfrad_var} (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_title(f"Observed {surfrad_var}", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    for _nwp_source, ax, _color in zip(nwp_error_reindexed.nwp_source, axs[1:], colors):

        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        if meteo_mask is not None:
            error_data = error_data.where(meteo_mask)
        # Only compute the RMSE over sunlit timesteps for consistency with
        # other error metrics.
        rmse = np.sqrt((error_data.where(surfrad_reindexed.zenith < 80)**2).mean())

        for _DOY in error_data.dayofyear:

            _error = error_data.sel(dayofyear=_DOY)
            ax.plot(
                _error.hour,
                _error,
                color=_color,
                alpha=0.5,
                linestyle="solid",
                linewidth=0.75,
            )
        ax.set_ylim(-1000, 1000)
        ax.set_xlabel(f"Hour of the Day (UTC + {utc_shift})", fontsize=fontsize)
        ax.set_ylabel(f"{surfrad_var} Error (Wm$^{-2}$)", fontsize=fontsize)
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
        title_str = f"{surfrad_var} rank histogram at {surfrad_sitename} during {meteo_name}"
    else:
        title_str = f"{surfrad_var} rank histogram at {surfrad_sitename}"
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


# %%

if __name__ == "__main__":
    # Perform comparison of surfrad obs. and HRRR GHI forecast

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    save_figs = True
    surfrad_sitename = "sxf"
    full_load_path = os.path.join(load_path, surfrad_sitename)

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
    month_end = 6
    day_end = 3

    data_datetime_start = pd.Timestamp(year_start, month_start, day_start)
    data_datetime_end = pd.Timestamp(year_end, month_end, day_end + 1)

    datestring = data_datetime_start.strftime("%Y%m%d_") + data_datetime_end.strftime(
        "%Y%m%d"
    )

    surfrad_ds, surfrad_masks = load_and_mask_surfrad(
        full_load_path,
        surfrad_sitename,
        data_datetime_start,
        data_datetime_end,
    )
    surfrad_ds = surfrad_ds.load()
    nwp_ds, nwp_masks = load_and_mask_nwp(
        full_load_path,
        surfrad_sitename,
        data_datetime_start,
        data_datetime_end,
        nwp_datavars,
    )

    # Collapse the forecast time dimension so the forecasts
    # appear as a timeseries. Must select a <= 24 hour forecast
    # window so there are no forecast overlaps.
    nwp_dayahead_ds = nwp_ds.sum(dim="time", min_count=1).load()

    # Add an ensemble average for the RRFS forecasts.
    rrfs_mems = [i for i in list(nwp_dayahead_ds.nwp_source.values) if i != "hrrr"]
    nwp_dayahead_ds, rrfsmems_mask = compute_nwp_ensemble_averages(
        nwp_dayahead_ds,
        rrfs_mems,
        ens_name="rrfs_mean",
        count_threshold=3,
    )

    # Repeat but exclude mem0001 and mem0004 to create a "good" ensemble mean.
    rrfs_goodmems = rrfs_mems.copy()
    rrfs_goodmems.remove("rrfs_mem0001")
    rrfs_goodmems.remove("rrfs_mem0004")
    nwp_dayahead_ds, goodmems_mask = compute_nwp_ensemble_averages(
        nwp_dayahead_ds,
        rrfs_goodmems,
        ens_name="rrfs_goodmean",
        count_threshold=3,
    )

    plot_bulk_metrics(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        save_figs=save_figs,
    )

    plot_histogram_error(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        save_figs=save_figs,
    )

    plot_time_series(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        datetime_start=data_datetime_start,
        datetime_end=data_datetime_end,
        plot_error=False,
        save_figs=save_figs,
    )

    plot_time_series(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        datetime_start=data_datetime_start,
        datetime_end=data_datetime_end,
        plot_error=True,
        save_figs=save_figs,
    )

    plot_composites(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=None,
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
    clear_mask = daily_csi_mean > 1.02
    broken_mask = np.bitwise_and(
        ~clear_mask,
        daily_csi_mean + 2 * daily_csi_stddev > 0.92
    )
    cloudy_mask = ~np.bitwise_or(clear_mask, broken_mask)

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=clear_mask,
        meteo_name="Clear",
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=broken_mask,
        meteo_name="Brokenclouds",
        save_figs=save_figs,
    )

    plot_daily_values(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        utc_shift=utc_shift,
        meteo_mask=cloudy_mask,
        meteo_name="Cloudy",
        save_figs=save_figs,
    )

# Test rank histograms:
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
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        ens_members=rrfs_members,
        save_figs=save_figs,
    )

    plot_rank_histogram(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        ens_members=rrfs_members,
        meteo_mask=cloudy_mask,
        meteo_name="Cloudy Conditions",
        save_figs=save_figs,
    )

    plot_rank_histogram_check(
        surfrad_var=surfrad_var,
        nwp_var=nwp_var,
        surfrad_ds=surfrad_ds,
        nwp_ds=nwp_dayahead_ds,
        nwp_masks=nwp_masks,
        surfrad_masks=surfrad_masks,
        ens_members=rrfs_members,
        save_figs=save_figs,
    )

# %%
