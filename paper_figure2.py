"""
Show diurnal biases in GHI for HRRR and RRFS relative to SURFRAD.

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
    fontsize: float = 16,
    axs: np.ndarray = None,
    orientation: str = "horizontal",
    num_panels_dim: int = 2,
    whis: float or (float, float) = 1.5,
    showfliers: bool = False,
    save_figs: bool = False,
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

    colors = [sns.color_palette("colorblind")[0] for i in axs]
    for _nwp_source, ax, _color in zip(nwp_error_reindexed.nwp_source, axs, colors):
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)

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
            f"{str(_nwp_source.values)}",
            fontsize=fontsize,
        )
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels(range(0, 25, 6))
        ax.tick_params(axis="both", labelsize=fontsize - 2)

    if save_figs:
        save_filename = f"DailyErrorPanels_{meteo_name}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )

    return axs


def plot_diurnal_whiskerplot2(
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
    axs: np.ndarray = None,
    orientation: str = "horizontal",
    num_panels_dim: int = 2,
    whis: float or (float, float) = 1.5,
    showfliers: bool = False,
    save_figs: bool = False,
):
    """
    Visualize the error and show the spread for each hour 
    of the day using box and whisker plots, with multiple NWP sources
    staggered side-by-side within the same panel.

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

    # Create the figure and axes if not provided
    if axs is None:
        num_panels = 1
        panel_size = 8
        if orientation == "horizontal":
            fig, axs = plt.subplots(
                num_panels_dim,
                num_panels,
                figsize=(
                    panel_size * num_panels,
                    panel_size * num_panels_dim,
                ),
            )
            if num_panels_dim == 1:
                axs = [axs]
            else:
                axs = [*axs.flat]
            fig.subplots_adjust(wspace=0.38, hspace=0.25)

        if orientation == "vertical":
            fig, axs = plt.subplots(
                num_panels,
                num_panels_dim,
                figsize=(
                    panel_size * num_panels_dim,
                    panel_size * num_panels,
                ),
            )
            if num_panels_dim == 1:
                axs = [axs]
            else:
                axs = [*axs.flat]
            fig.subplots_adjust(wspace=0.3, hspace=0.25)

    ax = axs[0] if isinstance(axs, list) else axs
    colors = sns.color_palette("colorblind")[:len(nwp_error_reindexed.nwp_source)]
    
    # Get valid hours (those with data)
    valid_hours = set()
    for _nwp_source in nwp_error_reindexed.nwp_source:
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        for h in range(24):
            if not np.isnan(error_data.sel(hour=h).values).all():
                valid_hours.add(h)
    valid_hours = sorted(list(valid_hours))
    
    # Stagger positions for side-by-side comparison
    num_sources = len(nwp_error_reindexed.nwp_source)
    offset = 0.40
    position_offset = np.linspace(-offset * (num_sources - 1) / 2, offset * (num_sources - 1) / 2, num_sources)
    
    # Store legend elements
    legend_elements = []
    
    for source_idx, (_nwp_source, _color) in enumerate(zip(nwp_error_reindexed.nwp_source, colors)):
        error_data = nwp_error_reindexed.sel(nwp_source=_nwp_source)
        
        # Convert error data to a format suitable for box plots
        error_values_by_hour = [error_data.sel(hour=h).values.flatten() for h in valid_hours]
        # Filter out NaN values
        error_values_by_hour = [errors[~np.isnan(errors)] for errors in error_values_by_hour]
        
        # Create staggered positions
        positions = [h + position_offset[source_idx] for h in valid_hours]
        
        # Create box plot
        box_parts = ax.boxplot(
            error_values_by_hour,
            positions=positions,
            patch_artist=True,
            widths=0.30,
            whis=whis,
            showfliers=showfliers,
            medianprops={'color': 'black'},
            boxprops={'facecolor': _color, 'alpha': 0.7}
        )

        # Add legend element for this source
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=_color, alpha=0.7, label=str(_nwp_source.values).upper()))
    
    ax.set_ylim(-750, 750)
    ax.set_xlabel("Local Time (hours)", fontsize=fontsize)
    ax.set_ylabel(f"{surfrad_var.upper()} Error (Wm$^{-2}$)", fontsize=fontsize)
    ax.set_xticks(valid_hours)
    ax.set_xticklabels([f"{h:02d}" for h in valid_hours])
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.legend(handles=legend_elements, fontsize=fontsize - 2)
    ax.grid(axis='y', alpha=0.3)

    if save_figs:
        save_filename = f"DailyErrorPanels_{meteo_name}_{surfrad_var}_{datestring}_{surfrad_sitename}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path,
            format="png",
            bbox_inches="tight",
        )

    return axs if isinstance(axs, list) else ax


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    save_figs = False

    surfrad_var = "ghi"
    surfrad_clearsky_var = f"clearsky_{surfrad_var}"
    nwp_var = "dswrf"

    save_dir = os.path.join(
        "figures",
        "paper_figures",
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

    # Create figure for all sites.
    fig, axs = plt.subplots(
        7,
        2,
        figsize=(10, 25),
    )

    fig2, axs2 = plt.subplots(
        7,
        1,
        figsize=(8, 25),
    )

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
    # localtime_offsets
    for surfrad_sitename, _axs, _axs2 in zip(surfrad_sitenames, axs, axs2):
        print(f"Processing {surfrad_sitename.upper()}")
        utc_shift = utc_shift_dict[surfrad_sitename]

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

        _axsb = plot_diurnal_whiskerplot(
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
            whis=(2.5, 97.5),
            showfliers=False,
            save_figs=save_figs,
            axs=_axs,
        )

        _axsc = plot_diurnal_whiskerplot2(
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
            whis=(2.5, 97.5),
            showfliers=False,
            save_figs=save_figs,
            axs=_axs2,
        )

        # %%
        # Correct the figure axes for readability.
        panel_letters = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'n.']
        panel_labels = []

        for i, let in enumerate(panel_letters):
            surfrad_sitename = surfrad_sitenames[i // 2]
            panel_labels.append(f"{let} {surfrad_sitename.upper()}")
        for ax, panel_label in zip(axs.flat, panel_labels):
            ax.set_title("")
            ax.annotate(panel_label, xy=(0.05, 0.9), xycoords="axes fraction", fontsize=16)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xlim(4.5, 19.5)
            ax.set_ylim(-750, 750)
            ax.xaxis.set_major_locator(plt.MultipleLocator(6))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
            ax.tick_params(which='major', length=8)
            ax.tick_params(which='minor', length=4)
            ax.hlines(0, 4.5, 19.5, color='red', linestyle='solid', linewidth=1, zorder=1)
        for ax in axs.flat[:-2]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        for ax in axs[:, 1]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        axs[0, 0].set_title("HRRR", fontsize=16)
        axs[0, 1].set_title("RRFS Control", fontsize=16)
        fig.subplots_adjust(wspace=0.1)

        # %%
        # Correct the figure axes for readability. New Fig.
        panel_letters = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.']
        panel_labels = []

        for i, let in enumerate(panel_letters):
            surfrad_sitename = surfrad_sitenames[i]
            panel_labels.append(f"{let} {surfrad_sitename.upper()}")
        for ax, panel_label in zip(axs2.flat, panel_labels):
            ax.set_title("")
            ax.annotate(panel_label, xy=(0.02, 0.9), xycoords="axes fraction", fontsize=16)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xlim(4.5, 19.5)
            ax.set_ylim(-750, 750)
            ax.xaxis.set_major_locator(plt.MultipleLocator(6))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
            ax.tick_params(which='major', length=8)
            ax.tick_params(which='minor', length=4)
            ax.hlines(0, 4.5, 19.5, color='red', linestyle='solid', linewidth=1, zorder=1)
        for ax in axs2.flat[:-1]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        for ax in axs2.flat[1:]:
            ax.get_legend().remove()
        fig.subplots_adjust(wspace=0.1)

    # %%
    fig.savefig(
        os.path.join(save_dir, f"DiurnalWhisker_{surfrad_var}_{datestring}_localtime.png"),
        format="png",
        bbox_inches="tight",
        dpi=200,
    )

    fig2.savefig(
        os.path.join(save_dir, f"DiurnalWhisker_{surfrad_var}_{datestring}_localtime_shared.png"),
        format="png",
        bbox_inches="tight",
        dpi=200,
    )

    # Figure caption:
    # Hourly box and whisker plots of surface downwelling shortwave radiation error
    # (Wm$^{-2}$) for HRRR and RRFS Control forecasts relative to SURFRAD observations at each SURFRAD site. Whiskers span a 95% confidence interval.
    # %%