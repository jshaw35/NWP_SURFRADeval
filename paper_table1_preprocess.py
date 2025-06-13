"""
Compare NWP predictions with surfrad records. Specifically, generate the data needed in Table 1.

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


def compute_bulk_metrics(
    surfrad_var: str,
    nwp_var: str,
    surfrad_ds: xr.Dataset,
    nwp_ds: xr.Dataset,
    nwp_masks: xr.Dataset,
    surfrad_masks: xr.Dataset,
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

    return error_metrics


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


# %%
if __name__ == "__main__":

    # Specify input and output fields and naming.
    load_path = "data/processed_timeseries"
    figure_save_path = "figures"
    data_save_path = "data/figure_outputs"
    save_figs = False

    surfrad_var = "ghi"
    surfrad_clearsky_var = f"clearsky_{surfrad_var}"
    nwp_var = "dswrf"
    utc_shift = -6


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

    error_dict = {}
    for surfrad_sitename in ["dra", "tbl", "fpk", "sxf", "bon", "gwn", "psu"]:
        print(f"Processing {surfrad_sitename.upper()}")

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

        error_metrics = compute_bulk_metrics(
            surfrad_var=surfrad_var,
            nwp_var=nwp_var,
            surfrad_ds=surfrad_ds,
            nwp_ds=nwp_dayahead_ds.sel(location=surfrad_sitename),
            nwp_masks=nwp_masks.sel(location=surfrad_sitename),
            surfrad_masks=surfrad_masks,
        )
        error_dict[surfrad_sitename] = error_metrics[-1].assign_coords(site=surfrad_sitename).expand_dims("site")
    ghi_rmse_da = xr.combine_by_coords(list(error_dict.values()))
    ghi_rmse_da.name = "GHI_RMSE"

    # Create a markdown table from the RMSE values

    # Get the site names from the code
    sites = ['dra', 'tbl', 'fpk', 'sxf', 'bon', 'gwn', 'psu']

    # Get the NWP model names from the dataset
    models = [str(src) for src in ghi_rmse_da.nwp_source.values]
    sites = [str(site) for site in ghi_rmse_da.site.values]

    # Create a pandas DataFrame with the RMSE values
    rmse_df = pd.DataFrame(
        np.array(ghi_rmse_da.round(1)),
        index=[site.upper() for site in sites],
        columns=models,
    )
    rmse_df.to_csv(f"{data_save_path}/table1_ghi_rmse.csv")

    # # Format the table as markdown
    # markdown_table = rmse_df.round(1).to_markdown()
    # print(markdown_table)

    # # For LaTeX output
    # latex_table = rmse_df.round(1).to_latex()
    # print("\nLaTeX Table:")
    # print(latex_table)
# %%

# RMSE Table

# hrrr	rrfs_control	rrfs_mem0001	rrfs_mem0002	rrfs_mem0003	rrfs_mem0004	rrfs_mem0005
# BON	158.7	176.1	198.1	171.9	184.6	192.2	177.3
# DRA	92.7	95.0	101.0	90.7	91.5	109.1	90.1
# FPK	164.3	168.5	181.8	174.0	174.6	190.2	173.8
# GWN	162.8	177.4	198.6	179.7	179.8	207.7	168.1
# PSU	156.9	174.0	200.7	169.4	165.3	201.4	167.0
# SXF	158.1	165.8	186.1	162.8	170.8	184.0	167.6
# TBL	187.9	177.8	197.1	176.6	185.3	199.7	191.0

# Dave wants me to compute the error variance (which I think is the variance after bias correction), but I am not sure how he wants me to remove the bias since it will depend on the time of day.
