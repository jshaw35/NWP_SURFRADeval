"""
Show diurnal biases in clearsky-normalized GHI for HRRR and RRFS relative to SURFRAD.

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
    nwp_masks: xr.DataArray,
    utc_shift: int,
    meteo_mask: xr.DataArray = None,
    meteo_name: str = None,
    fontsize: float = 16,
    ax: plt.axis = None,
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
    surfrad_data = surfrad_ds[surfrad_var].where(nwp_mask).load()

    surfrad_reindexed = shift_and_reindex_time(
        surfrad_data,
        surfrad_var,
        "valid_time",
        utc_shift=utc_shift,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    obs_data = surfrad_reindexed
    if meteo_mask is not None:
        obs_data = obs_data.where(meteo_mask)
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
    ax.set_title(f"{meteo_name}", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)

    return ax


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    save_figs = False
    showfliers = False

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
    fig1, axs1 = plt.subplots(
        7,
        3,
        figsize=(12, 25),
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
    stddev_list = []
    Ndays_list = []
    Nhours_list = []
    for surfrad_sitename, _axs1 in zip(surfrad_sitenames, axs1):
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
        # nwp_ds = nwp_ds.sel(nwp_source=["hrrr", "rrfs_control", "rrfs_mem0002"], location=surfrad_sitename).drop_vars("location")
        nwp_ds = nwp_ds.sel(location=surfrad_sitename).drop_vars("location")
        nwp_masks = nwp_masks.sel(location=surfrad_sitename).drop_vars("location")

        # A more involved example applying masks for different
        # meteorological conditions and creating separate plots.
        nwp_mask = nwp_masks[nwp_var].sum(dim="time")
        obs_mask = surfrad_masks[surfrad_var]
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

        # classifications from looking at the data.
        empty_mask = daily_csi_mean.isnull()
        clear_mask = daily_csi_mean > 0.95
        broken_mask = np.bitwise_and(
            ~clear_mask,
            daily_csi_mean + 2 * daily_csi_stddev > 0.92
        )
        cloudy_mask = ~(clear_mask | broken_mask | empty_mask)

        clear_Ndays = clear_mask.sum()
        broken_Ndays = broken_mask.sum()
        cloudy_Ndays = cloudy_mask.sum()

        # Spaghetti plot of the clearsky index for each scene type.
        plot_daily_values(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_masks=nwp_masks,
            utc_shift=utc_shift,
            meteo_mask=clear_mask,
            meteo_name="Clear",
            ax=_axs1[0],
        )
        plot_daily_values(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_masks=nwp_masks,
            utc_shift=utc_shift,
            meteo_mask=broken_mask,
            meteo_name="Partly Cloudy",
            ax=_axs1[1],
        )
        plot_daily_values(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_masks=nwp_masks,
            utc_shift=utc_shift,
            meteo_mask=cloudy_mask,
            meteo_name="Overcast",
            ax=_axs1[2],
        )
        _axs1[0].text(0.88, 0.93, clear_Ndays.values, transform=_axs1[0].transAxes, ha="center", va="center", color="red", fontsize=16)
        _axs1[1].text(0.88, 0.93, broken_Ndays.values, transform=_axs1[1].transAxes, ha="center", va="center", color="red", fontsize=16)
        _axs1[2].text(0.88, 0.93, cloudy_Ndays.values, transform=_axs1[2].transAxes, ha="center", va="center", color="red", fontsize=16)

    # Correct the figure axes for readability.
    panel_letters = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'n.']
    panel_labels = []

    # Remove y-labels and xtick labels for inner plots.
    for _ax in axs1[:,1:].flat:
        _ax.set_ylabel("")
        _ax.set_yticklabels([])
    for _ax in axs1[:-1,:].flat:
        _ax.set_xlabel("")
        _ax.set_xticklabels([])
    # Add the site name to the first column of plots.
    for _ax, surfrad_sitename in zip(axs1[:,0].flat, surfrad_sitenames):
        _ax.text(0.11, 0.93, surfrad_sitename.upper(), transform=_ax.transAxes, ha="center", va="center", color="red", fontsize=16)
    # Correct the x-axis labels for the bottom row of plots.
    for _ax in axs1[-1,:].flat:
        _ax.set_xlabel("Hour of the Day (local time)", fontsize=14)
    # Remove the titles outside of the top row of plots.
    for _ax in axs1[1:,:].flat:
        _ax.set_title("")
    # %%
    fig1.savefig(
        os.path.join(save_dir, f"DiurnalSpaghetti_SURFRADonly_cloudregimes.png"),
        format="png",
        bbox_inches="tight",
        dpi=100,
    )
    # %%