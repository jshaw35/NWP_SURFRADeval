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

    # %%

    # Extra plots to visualize the RRFS ensemble.
    surfrad_var = "ghi"
    nwp_var = "dswrf"
    surfrad_data = surfrad_ds[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].sel(location=surfrad_sitename)

    # Shift so the time is UTC - 6 hours so days contain all sunlit timesteps.
    nwp_data["valid_time"] = nwp_data["valid_time"] - np.timedelta64(6, "h")
    surfrad_data["valid_time"] = surfrad_data["valid_time"] - np.timedelta64(6, "h")

    fontsize = 16

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

        # ax.set_ylim(-10, 1000)
        ax.set_ylim(0, 1000)
        ax.set_xlabel("Hour of the Day", fontsize=fontsize)
        ax.set_ylabel("GHI (Wm$^{-2}$)", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        handles, labels = ax.get_legend_handles_labels()
    axs[0].legend(
        handles[:4],
        ["Observations", "HRRR", "RRFS Control", "RRFS Members 1-5"],
        loc="upper right",
    )
    for ax,label in zip(axs, ["a.", "b."]):
        # ax.grid(visible=True, which="both", linestyle="--", alpha=0.5)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, ha="center", fontsize=fontsize)

    # Annotate members 4 and 5
    # axs[0].text(0.3, 0.9, "RRFS mem4, mem5", transform=axs[0].transAxes, ha="center", fontsize=fontsize-2)
    # axs[1].text(0.75, 0.94, "RRFS mem4, mem5", transform=axs[1].transAxes, ha="center", fontsize=fontsize-2)
    axs[0].annotate(
        "RRFS mem1, RRFS mem4",
        xy=(0.48, 0.8),
        xytext=(0.25, 0.87),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1},
    )
    axs[1].annotate(
        "RRFS mem1, RRFS mem4",
        xy=(0.48, 0.87),
        xytext=(0.25, 0.94),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1},
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"fig_supp1_anecdotal.png"),
        dpi=300,
    )
    # %%

    # Extra plots to visualize the RRFS ensemble.
    surfrad_var = "ghi"
    nwp_var = "dswrf"
    surfrad_data = surfrad_ds[surfrad_var]
    nwp_data = nwp_dayahead_ds[nwp_var].sel(location=surfrad_sitename)

    # Shift so the time is UTC - 6 hours so days contain all sunlit timesteps.
    nwp_data["valid_time"] = nwp_data["valid_time"] - np.timedelta64(6, "h")
    surfrad_data["valid_time"] = surfrad_data["valid_time"] - np.timedelta64(6, "h")

    fontsize = 16
    day = "2024-03-07"
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

        # ax.set_ylim(-10, 1000)
        ax.set_ylim(0, 1000)
        ax.set_xlabel("Hour of the Day", fontsize=fontsize)
        ax.set_ylabel("GHI (Wm$^{-2}$)", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize - 4)
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:4],
        ["Observations", "HRRR", "RRFS Control", "RRFS Members 1-5"],
        loc="upper right",
    )
# %%
