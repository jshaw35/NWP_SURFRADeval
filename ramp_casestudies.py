"""
Compare NWP predictions with surfrad records for specific days
when there was a clear ramp in GHI (preferably in the evening).

This code should be run after pre-processing data into timeseries
format using preprocess_to_tseries.py

"""

# %%

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import glob


def MaskNWPByAvailability(
    nwp_ds: xr.Dataset,
    mask_vars: list,
):
    """
    Function to reconcile different data availability for
    NWP forecasts by masking out times when only some products
    have data. Does not assume all variables have the same
    availability (e.g. ensemble forecasts missing DNI data but
    not GHI data).

    Inputs:
    nwp_ds: xarray Dataset containing the forecast data
    mask_vars: list of variables to apply masking over.

    Outputs:
    nwp_ds: Input Dataset masked consistently across NWP forecasts.
    xr.merge(masks): xarray dataset with a mask for each variable.
    """

    masks = []
    # If an nwp product is nan everywhere for a variable,
    # then ignore it when masking.
    num_nan = np.isnan(nwp_ds).all(dim=["valid_time", "time"]).sum(dim="nwp_source")
    for _var in mask_vars:
        _mask = np.isnan(nwp_ds[_var]).sum(dim="nwp_source") == num_nan[_var]

        nwp_ds[_var] = nwp_ds[_var].where(_mask)
        masks.append(_mask)

    # Apply the last mask to the step. Somewhat arbitrary.
    if "step" in nwp_ds.coords:
        nwp_ds = nwp_ds.assign_coords(step=nwp_ds["step"].where(_mask))

    return nwp_ds, xr.merge(masks)


def MaskSurfradByQCFlags(
    surfrad_ds: xr.Dataset,
    mask_threshold: float = 0.1,
):
    """
    Use the surfrad QC flags to mask variables appropriately.

    Inputs:
    surfrad_ds: xarray Dataset to mask over.
    mask_threshold: float indicating the threshold for masking out \
    surfrad data. e.g. 0.1 indicates that if more than 10% of the \
    hourly period was flagged then that data will be masked out.

    Outputs:
    surfrad_ds: xarray Dataset that has been masked.
    """

    for _var in surfrad_ds.data_vars:
        if (_var + "_flag") in surfrad_ds.data_vars:
            _var_mask = surfrad_ds[_var + "_flag"] <= mask_threshold
            surfrad_ds[_var] = surfrad_ds[_var].where(_var_mask)

    return surfrad_ds


def LoadAndMaskSurfrad(
    load_path: str,
    surfrad_sitename: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
):
    """
    Load Surfrad data and mask according to data availability.

    Inputs:
    load_path: string to where data is stored
    surfrad_sitename: string corresponding the to surfrad site.
    datetime_start: pandas Timestamp for the time period start
    datetime_end: pandas Timestamp for the time period end

    Outputs:
    ds: Surfrad data that has been loaded and masked.
    """

    datestring = datetime_start.strftime("%Y%m%d_") + datetime_end.strftime("%Y%m%d")
    load_dir = os.path.join(
        load_path,
        datestring,
    )
    filename = f"{datestring}_surfrad_{surfrad_sitename}.nc"
    ds = xr.open_dataset(os.path.join(load_dir, filename))
    ds = MaskSurfradByQCFlags(ds)

    # Re-index along a continuous time dimension
    continuous_time = pd.date_range(
        start=datetime_start,
        end=datetime_end,
        freq="h",
    )
    ds = ds.reindex(valid_time=continuous_time)

    return ds


def LoadAndMaskNWP(
    load_path: str,
    surfrad_sitename: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    datavars: list,
):
    """
    Load NWP data and mask according to data availability.

    Inputs:
    load_path: string to where data is stored
    surfrad_sitename: string corresponding the to surfrad site where the NWP \
    was extracted to be nearest.
    datetime_start: pandas Timestamp for the time period start
    datetime_end: pandas Timestamp for the time period end
    datavars: list of variables to mask over

    Outputs:
    ds: NWP data that has been loaded and masked.
    """

    datestring = datetime_start.strftime("%Y%m%d_") + datetime_end.strftime("%Y%m%d")
    load_dir = os.path.join(
        load_path,
        datestring,
    )

    filename_wc = f"????????_????????_nwp_{surfrad_sitename}.nc"
    filenames = glob.glob(os.path.join(load_dir, filename_wc))
    filenames.sort()
    ds = xr.open_mfdataset(filenames, combine="nested")[datavars]
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

    ds, avail_masks = MaskNWPByAvailability(ds, datavars)

    # Re-index along a continuous time dimension
    continuous_time = pd.date_range(
        start=datetime_start,
        end=datetime_end,
        freq="h",
    )

    ds = ds.reindex(valid_time=continuous_time)

    return ds, avail_masks


if __name__ == "__main__":

    # Specify input and output fields and naming.
    surfrad_sitename = "gwn"
    load_path = f"data/processed_timeseries/{surfrad_sitename}"
    save_figs = False

    nwp_datavars = [
        "dswrf",
    ]

    year_start = 2024
    month_start = 3
    day_start = 1

    year_end = 2024
    month_end = 6
    day_end = 3

    data_datetime_start = pd.Timestamp(year_start, month_start, day_start)
    data_datetime_end = pd.Timestamp(year_end, month_end, day_end + 1)

    datestring = data_datetime_start.strftime("%Y%m%d_") + data_datetime_start.strftime(
        "%Y%m%d"
    )

    surfrad_ds = LoadAndMaskSurfrad(
        load_path,
        surfrad_sitename,
        data_datetime_start,
        data_datetime_end,
    ).load()
    nwp_ds, avail_masks = LoadAndMaskNWP(
        load_path,
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
    rrfs_mean = nwp_dayahead_ds.sel(nwp_source=rrfs_mems).mean(dim="nwp_source")

    nwp_dayahead_ds = xr.merge(
        [
            nwp_dayahead_ds,
            rrfs_mean.assign_coords(nwp_source="rrfs_mean").expand_dims("nwp_source"),
        ]
    )

    # Repeat but exclude mem0001 and mem0004 to create a "good" ensemble mean.
    rrfs_goodmems = rrfs_mems.copy()
    rrfs_goodmems.remove("rrfs_mem0001")
    rrfs_goodmems.remove("rrfs_mem0004")
    rrfs_goodmean = nwp_dayahead_ds.sel(nwp_source=rrfs_goodmems).mean(dim="nwp_source")
    nwp_dayahead_ds = xr.merge(
        [
            nwp_dayahead_ds,
            rrfs_goodmean.assign_coords(nwp_source="rrfs_goodmean").expand_dims(
                "nwp_source"
            ),
        ]
    )

    # Mask the ensemble means if there are less than 3 members in them.
    # Create masks
    rrfsmems_subset = nwp_dayahead_ds.sel(nwp_source=rrfs_mems)
    rrfsmems_empty = np.isnan(rrfsmems_subset).all(dim="valid_time").sum(dim="nwp_source")
    rrfsmems_mask = (len(rrfs_goodmems) - rrfsmems_empty) >= 3
    rrfsmems_mask = np.bitwise_or(rrfsmems_mask, (nwp_dayahead_ds.nwp_source != "rrfs_goodmean"))

    goodmems_subset = nwp_dayahead_ds.sel(nwp_source=rrfs_goodmems)
    goodmems_empty = np.isnan(goodmems_subset).all(dim="valid_time").sum(dim="nwp_source")
    goodmems_mask = (len(rrfs_goodmems) - goodmems_empty) >= 3
    goodmems_mask = np.bitwise_or(goodmems_mask, (nwp_dayahead_ds.nwp_source != "rrfs_mean"))
    ens_mask = np.bitwise_and(rrfsmems_mask, goodmems_mask)
    # Apply masks
    nwp_dayahead_ds = nwp_dayahead_ds.where(ens_mask)
    nwp_dayahead_ds = nwp_dayahead_ds.assign_attrs(ens_mask=ens_mask)

    # Create a plot for specific days with broken or partly cloudy conditions.
    surfrad_var = "ghi"
    nwp_var = "dswrf"
    surfrad_data = surfrad_ds[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].drop_sel(nwp_source=["rrfs_mean", "rrfs_goodmean"])

    # Shift so the time is UTC - 6 hours so days contain all sunlit timesteps.
    nwp_data["valid_time"] = nwp_data["valid_time"] - np.timedelta64(6, "h")
    surfrad_data["valid_time"] = surfrad_data["valid_time"] - np.timedelta64(6, "h")

    colors = sns.color_palette("colorblind")
    example_dates = [
        "2024-03-07",
        # "2024-03-13", # No NWP
        "2024-03-21",
        "2024-04-07",
        "2024-04-08",
        "2024-04-15",
        "2024-04-16",
        "2024-04-27",
        "2024-04-30",
        "2024-05-02",
        # "2024-05-05", # No NWP
        "2024-05-14",
        "2024-05-16",
        "2024-05-20",
        "2024-05-21",
        "2024-05-22",
        # "2024-05-28", # No NWP
        "2024-05-29",
        # "2024-06-02", # No NWP
    ]

    nrow = int(np.ceil(len(example_dates) / 4))
    fig, axs = plt.subplots(nrow, 4, figsize=(15, 3*nrow))
    fig.subplots_adjust(hspace=0.3)
    axs = axs.flat

    for _ax, _day in zip(axs, example_dates):
        _ax.set_title(_day)
        _nwp_day = nwp_data.sel(valid_time=_day)
        _surfrad_day = surfrad_data.sel(valid_time=_day)

        _ax.plot(
            _surfrad_day.valid_time.dt.hour,
            _surfrad_day,
            label="SURFRAD",
            color="black",
            alpha=0.5,
            linestyle="dashed",
        )

        for _nwp_source, _color in zip(_nwp_day.nwp_source, colors):

            _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
            if np.isnan(_forecast_data).all(): continue
            _ax.plot(
                _forecast_data.valid_time.dt.hour,
                _forecast_data,
                label=str(_nwp_source.values),
                color=_color,
                alpha=0.5,
                linestyle="solid",
            )

    handles, labels = _ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.8,0.06),framealpha=0.9)

    if save_figs:
        save_filename = f"GHIcomparison_partlycloudy_{surfrad_sitename}.png"
        fig.savefig(
            save_filename,
            format="png",
            bbox_inches="tight",
        )

    fig, axs = plt.subplots(nrow, 4, figsize=(15, 3*nrow))
    fig.subplots_adjust(hspace=0.3)
    axs = axs.flat

    for _ax, _day in zip(axs, example_dates):
        _ax.set_title(_day)
        _nwp_day = nwp_data.sel(valid_time=_day)
        _surfrad_day = surfrad_data.sel(valid_time=_day)

        for _nwp_source, _color in zip(_nwp_day.nwp_source, colors):

            _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
            if np.isnan(_forecast_data).all(): continue
            _ax.plot(
                _forecast_data.valid_time.dt.hour,
                _forecast_data - _surfrad_day,
                label=str(_nwp_source.values),
                color=_color,
                alpha=0.5,
                linestyle="solid",
            )

    handles, labels = _ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.8,0.06),framealpha=0.9)

    if save_figs:
        save_filename = f"GHIerror_partlycloudy_{surfrad_sitename}.png"
        fig.savefig(
            save_filename,
            format="png",
            bbox_inches="tight",
        )

    # Compare the HRRR with just the RRFS ensemble mean.
    surfrad_var = "ghi"
    nwp_var = "dswrf"
    surfrad_data = surfrad_ds[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].sel(nwp_source=["hrrr", "rrfs_goodmean"])

    # Shift so the time is UTC - 6 hours so days contain all sunlit timesteps.
    nwp_data["valid_time"] = nwp_data["valid_time"] - np.timedelta64(6, "h")
    surfrad_data["valid_time"] = surfrad_data["valid_time"] - np.timedelta64(6, "h")

    nrow = int(np.ceil(len(example_dates) / 4))
    fig, axs = plt.subplots(nrow, 4, figsize=(15, 3*nrow))
    fig.subplots_adjust(hspace=0.3)
    axs = axs.flat

    for _ax, _day in zip(axs, example_dates):
        _ax.set_title(_day)
        _nwp_day = nwp_data.sel(valid_time=_day)
        _surfrad_day = surfrad_data.sel(valid_time=_day)

        _ax.plot(
            _surfrad_day.valid_time.dt.hour,
            _surfrad_day,
            label="SURFRAD",
            color="black",
            alpha=0.5,
            linestyle="dashed",
        )

        for _nwp_source, _color in zip(_nwp_day.nwp_source, colors):

            _forecast_data = _nwp_day.sel(nwp_source=_nwp_source)
            if np.isnan(_forecast_data).all(): continue
            _ax.plot(
                _forecast_data.valid_time.dt.hour,
                _forecast_data,
                label=str(_nwp_source.values),
                color=_color,
                alpha=0.5,
                linestyle="solid",
            )

    handles, labels = _ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.81,0.08),framealpha=0.9)

    if save_figs:
        save_filename = f"GHIcomparison_RRFSens_partlycloudy_{surfrad_sitename}.png"
        fig.savefig(
            save_filename,
            format="png",
            bbox_inches="tight",
        )
# %%
