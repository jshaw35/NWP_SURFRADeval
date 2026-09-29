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
    ax=None,
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
    ax: matplotlib axis
        axis to plot on. If None, then a new figure and axis is created.
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
    bins = np.arange(-0.5, len(ens_members) + 0.51, 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
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
        color="red",
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


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    save_figs = False
    showfliers = False

    surfrad_var = "ghi"
    surfrad_clearsky_var = f"clearsky_{surfrad_var}"
    nwp_var = "dswrf"

    # morning_hours = [5, 11]
    # afternoon_hours = [13, 19]
    # Results appear insensitive to how morning/afternoon are defined.
    morning_hours = [7, 11]
    afternoon_hours = [13, 17]

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
    for surfrad_sitename, _ax in zip(surfrad_sitenames, axs):
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
        nwp_ds = nwp_ds.sel(location=surfrad_sitename).drop_vars("location")
        nwp_masks = nwp_masks.sel(location=surfrad_sitename).drop_vars("location")

        # Create morning and afternoon masks. 
        # Neither of these plays nice with implied bitwise comparisons
        surfrad_morning_mask = ((surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 >= morning_hours[0]).values & \
        ((surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 <= morning_hours[1]).values

        surfrad_afternoon_mask = ((surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 >= afternoon_hours[0]).values & \
        ((surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 <= afternoon_hours[1]).values

        surfrad_morning_mask = np.bitwise_and(
            (surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 >= morning_hours[0],
            (surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 <= morning_hours[1]
        )
        surfrad_afternoon_mask = np.bitwise_and(
            (surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 >= afternoon_hours[0],
            (surfrad_masks[surfrad_var].valid_time.dt.hour + utc_shift) % 24 <= afternoon_hours[1]
        )

        # Collapse the forecast time dimension so the forecasts
        # appear as a timeseries. Must select a <= 24 hour forecast
        # window so there are no forecast overlaps.
        nwp_dayahead_ds = nwp_ds.sum(dim="time", min_count=1).load()

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

        # classifications from looking at the data.
        empty_mask = daily_csi_mean.isnull()
        clear_mask = daily_csi_mean > 0.95
        broken_mask = np.bitwise_and(
            ~clear_mask,
            daily_csi_mean + 2 * daily_csi_stddev > 0.92
        )
        cloudy_mask = ~(clear_mask | broken_mask | empty_mask)

        # Collapse the forecast time dimension so the forecasts
        # appear as a timeseries. Must select a <= 24 hour forecast
        # window so there are no forecast overlaps.
        nwp_dayahead_ds = nwp_ds.sum(dim="time", min_count=1).load()

        rrfs_members = [
            "hrrr",
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
            surfrad_masks=surfrad_masks & surfrad_morning_mask,
            ens_members=rrfs_members,
            ax=_ax[0],
            save_figs=save_figs,
        )
        plot_rank_histogram(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=nwp_dayahead_ds,
            nwp_masks=nwp_masks,
            surfrad_masks=surfrad_masks & surfrad_afternoon_mask,
            ens_members=rrfs_members,
            ax=_ax[1],
            save_figs=save_figs,
        )
        # break

        # %%
    # Correct the figure axes for readability.
    panel_letters = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'k.', 'l.', 'm.', 'n.']
    panel_labels = []
    for i, let in enumerate(panel_letters):
        surfrad_sitename = surfrad_sitenames[i // 2]
        panel_labels.append(f"{let} {surfrad_sitename.upper()}")
    for ax, panel_label in zip(axs.flat, panel_labels):
        ax.set_title("")
        ax.annotate(panel_label, xy=(0.02, 0.90), xycoords="axes fraction", fontsize=12)
        ax.set_ylim(0, 300)
    for ax in axs[0,:]:
        ax.set_ylim(0, 500) # Use separate limits for the top row
    for ax in axs.flat[:-2]:
        ax.set_xlabel("")
    for ax in axs[:, 1]:
        ax.set_ylabel("")
    axs[0, 0].set_title(f"Morning ({morning_hours[0]} - {morning_hours[1]} Local Time)", fontsize=16)
    axs[0, 1].set_title(f"Afternoon ({afternoon_hours[0]} - {afternoon_hours[1]} Local Time)", fontsize=16)

    # %%

    fig.savefig(
        os.path.join(save_dir, f"RankHistogram_{surfrad_var}_{datestring}_{morning_hours[0]}_{morning_hours[1]}_{afternoon_hours[0]}_{afternoon_hours[1]}_hrrr.png"),
        format="png",
        bbox_inches="tight",
        dpi=200,
    )

    # Figure caption:
    # Rank histograms for morning (left column) and afternoon (right column) hours at each SURFRAD site. Observation rank is computed relative to a 6-member RRFS ensemble forecast (RRFS Control and 5 ensemble members). Dashed horizontal lines indicate the rank histogram of an ideally dispersed ensemble. Morning and afternoon hours are calculated with each site's local time without daylight savings time adjustments.
    # %%