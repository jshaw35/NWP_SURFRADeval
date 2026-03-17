"""
Pre-process surfrad observations and NWP data into
subsetted timeseries for analysis.

This requires:
- Applying appropriate dimensions so data can be concatenated.
    - This is handled by "PreprocForConcatAlongValidTime".
- Subsetting by location before concatenating datasets.
    - Subsetting requires identifying the closest location, which can
    be slow due to computing the distance between each gridcell and a
    desired point. Because NWP files share a spatial grid, determining
    the closest point only needs to be done once (rather than being repeated
    as each file is preprocessed). So when loading many files using
    preprocessing, we compute the closest point on a test file using
    "get_closest" and pass that location into the preprocessing call.

"""

# %%

import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from functools import partial
import itertools
import pvlib
from pvlib.location import Location


def haversine(
    lon1: np.array,
    lat1: np.array,
    lon2: np.array,
    lat2: np.array,
):
    """
    Function for the haversine function. Used to calculate the
    "great circle" distance between any two points on the earth's
    surface.

    Inputs:

    lon1: numpy array
        longitude in degrees of the first location
    lat1: numpy array
        latitude in degrees of the first location

    lon2: numpy array
        longitude in degrees of the second location
    lat2: numpy array
        latitude in degrees of the second location

    Outputs:

    distance_km: float
        Great circle distance between the locations in km.
    """

    R = 6371  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = np.deg2rad(np.array([lon1, lat1, lon2, lat2]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = R * c
    return distance_km


def get_closest(
    lat: float,
    lon: float,
    data: xr.DataArray,
):
    """
    Select the gridcell closest to a latitude-longitude coordinate.
    Use the haversine function to calculate great circle distances
    and index the nearest point.

    Inputs:

    lat: float
        Latitude in degrees of the desired point.

    lon: float
        Longitude in degrees of the desired point.

    data: xarray DataArray 
        (could be Dataset as well?) to compute distances with..

    Outputs:

    loc: tuple
        coordinates for indexing the closest location.
    """

    # Handle both 'lat','lon' and 'latitude','longitude' coordinates.
    if ("latitude" in data.coords) and ("longitude" in data.coords):
        distances = haversine(
            lon * np.ones(data.longitude.shape),
            lat * np.ones(data.latitude.shape),
            data.longitude,
            data.latitude,
        )
    elif ("lat" in data.coords) and ("lon" in data.coords):
        distances = haversine(
            lon * np.ones(data.lon.shape),
            lat * np.ones(data.lat.shape),
            data.lon,
            data.lat,
        )
    else:
        print('"data" is missing lat-lon coordinates.')
        return None

    loc = np.argwhere(distances == np.min(distances))

    return loc, np.min(distances)


def select_location(
    data: xr.DataArray,
    loc: tuple,
):
    """
    Select a horizontal location from data. Use with get_closest to \
    create a partial function to feed into preprocess in open_mfdataset.

    Inputs:
    data: xarray.DataArray or xarray.Dataset
        data to subset by horizontal location.

    loc: tuple 
        indices used to index horizontal location provided by get_closest.

    """
    # Assume there are only two horizontal dimensions.
    _hor_dims = data.longitude.dims
    sel_dict = dict(zip(_hor_dims, loc[0]))
    data_at_site = data.sel(sel_dict)

    return data_at_site


def preprocess_nwp(
    data: xr.DataArray,
    loc: tuple,
):
    """
    Wrapper to combine spatial subsetting (select_location) and dimension
    expanding for NWP file concatenation.

    Inputs:
    data: xarray DataArray 
        data to operate on
    loc: tuple
        input for select_location

    Outputs:
    data_out: xarray DataArray
        preprocessed data
    """

    # Select location
    data_subset = select_location(data, loc)

    # Add dimensions for concatenation
    data_out = data_subset.expand_dims(["valid_time", "time"]).drop_vars("step")

    return data_out


def load_and_subset_nwp(
    filelist: list,
    lat: float,
    lon: float,
):
    """
    Load a list of NWP files, concatentating appropriately and subsetting
    by a lat-lon location.

    Inputs:
    filelist: list
        list of filepaths to load and preprocess.
    lat: float
        latitude in degrees of the desired location.
    lon: float
        longitude in degrees of the desired location.
    """

    # Use a test file to speed up spatial subsetting.
    test_file = filelist[0]
    test_ds = xr.open_dataset(test_file)

    preproc_loc, dist = get_closest(lat, lon, test_ds)
    preproc_partial_func = partial(preprocess_nwp, loc=preproc_loc)

    preproc_ds = xr.open_mfdataset(
        filelist,
        chunks=None,
        preprocess=preproc_partial_func,
    )
    preproc_ds = preproc_ds.load()

    return preproc_ds


def load_surfrad_for_datetime_range(
    datapath: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    data_vars: list = None,
):
    """
    Try to load and concatenate any available surfrad data
    for a given timerange.

    datapath: string
        Directory where the "year" directories are for surfrad data.

    datetime_start: pandas Timestamp
        Start date.

    datetime_end: pandas Timestamp
        End date.

    data_vars: list
        list of strings corresponding to variables in the surfrad data.
    """

    years = np.arange(datetime_start.year, datetime_end.year + 1)
    dflist = []

    # Iterate over surfrad data and load each file, then concatenate.
    # But need to select the desire time period only.
    for _year in years:
        surfrad_files = glob.glob(
            f"{os.path.join(datapath, str(_year), '')}*.dat"
        )
        for _file in surfrad_files:

            # Parse the filenames to get the date.
            _file_timestr = _file.split(".")[-2][-5:]
            _file_datetime_start = pd.to_datetime(f"20{_file_timestr}", format="%Y%j")
            _file_datetime_end = _file_datetime_start + timedelta(1)

            # Only load if the file falls within the time range.
            # Check both beginning and end.
            if (
                (
                    (_file_datetime_start >= datetime_start)
                    and (_file_datetime_start <= datetime_end)
                )
                or
                (
                    (_file_datetime_end >= datetime_start)
                    and (_file_datetime_end <= datetime_end)
                )
            ):

                _data, _metadata = pvlib.iotools.read_surfrad(_file, map_variables=True)

                if data_vars is not None:
                    _data = _data[data_vars]
                # Mask out flagged values if not already masked.
                for _var in data_vars:
                    if (_var + "_flag") in data_vars:
                        _data[_var] = _data[_var].where(_data[_var + "_flag"] == 0)                    
                # Coarsen the data to hourly means for compression
                # and comparison with NWP. Could remove.
                _data = _data.resample("1h", offset=None).mean()
                dflist.append(_data)
    try:
        df = pd.concat(dflist)
    except ValueError:
        print("ERROR: No files found for this time range. Recheck your paths.")
        return None
    return df, _metadata


def instantaneous_to_average(
    data: xr.DataArray,
    dim: str,
    high_res: str = "5min",
    final_res: str = "1h",
    interp_method: str = "linear",
    keep_coords: bool = True,
):
    """
    Function for simply casting instantaneous data to time averages.
    Interpolate to a higher resolution and then compute an average.

    Inputs:

    data: xarray.DataArray
        Input data
    dim: string
        Dimension along which to resample data.
    high_res: string
        resolution to interpolate to.
    final_res: string
        Final resolution of "averaged" data.
    interp_method: string
        Interpolation method: {"linear", "slinear", "quadratic", "cubic", "nearest"}

    Outputs:

    averaged: xarray.DataArray
        Appropriately resampled array.
    """

    upscaled = data.resample({dim: high_res}).interpolate(interp_method)
    averaged = upscaled.resample({dim: final_res}).mean()

    # Keep coordinates.
    if keep_coords:
        for _coord in data.coords:
            if _coord not in averaged.coords:
                averaged = averaged.assign_coords({_coord: data[_coord]})

    return averaged


def add_clearsky_radiation(
    data: xr.Dataset,
    metadata: dict,
    time_dim: str,
):
    """
    Function to add clear-sky radiation fields as coordinates \
    to an xarray DataArray or Dateset object containing surfrad \
    observations using pvlib.

    Inputs:

    data: xarray Dataset
        object for a single lat-lon location
    metadata: dictionary
        Dictionary of metadata from the raw surfrad observations.
    time_dim: string
        identifier for the "true" time dimension (e.g. valid time).

    Outputs:

    data_out: xarray Dataset
        Input "data" object with "ghi", "dni", and "dhi" coordinates.
    """

    loc = Location(
        latitude=metadata["latitude"],
        longitude=metadata["longitude"],
        tz="UTC",
        altitude=metadata["elevation"],
        name=metadata["name"],
    )

    # Get insolation at data resolution (1-hour averages)
    # time_range = pd.DatetimeIndex(data[time_dim])

    # Get insolation at 5-minute resolution
    time_range = pd.DatetimeIndex(
        np.arange(
            data[time_dim][0].values,
            data[time_dim][-1].values + np.timedelta64(1, "h"),
            np.timedelta64(5, "m"),
        )
    )
    cs = loc.get_clearsky(time_range)
    cs = cs.resample("1h").mean()
    cs_ds = cs.to_xarray().rename(index=time_dim)
    cs_ds[time_dim] = pd.to_datetime(
        cs_ds[time_dim]
    )  # to_xarray screws up the time array

    # Add clear-sky radiation variables as coordinates to the NWP Dataset.
    data_out = data.assign_coords(
        {
            "clearsky_ghi": cs_ds["ghi"],
            "clearsky_dni": cs_ds["dni"],
            "clearsky_dhi": cs_ds["dhi"],
        },
    )

    return data_out


def add_solar_position(
    data: xr.DataArray,
    metadata: dict,
    time_dim: str,
):
    """
    Function to add solar position fields as coordinates \
    to an xarray DataArray or Dateset object containing surfrad \
    observations using pvlib.

    Inputs:

    data: xarray DataArray
        xarray object for a single lat-lon location
    metadata: dictionary
        Dictionary of metadata from the raw surfrad observations.
    time_dim: string
        indentifier for the "true" time dimension (e.g. valid time).

    Outputs:

    data_out: xarray DataArray
        Input data with solar zenith angle coordinate added.
    """

    loc = Location(
        latitude=metadata["latitude"],
        longitude=metadata["longitude"],
        tz="UTC",
        altitude=metadata["elevation"],
        name=metadata["name"],
    )

    # Get solar position at data resolution (1-hour averages)
    solpos = loc.get_solarposition(
        data[time_dim],
        pressure=data["pressure"],
    )
    solpos = solpos.resample("1h").mean()
    solpos_ds = solpos.to_xarray().rename(index=time_dim)
    solpos_ds[time_dim] = pd.to_datetime(
        solpos_ds[time_dim]
    )  # to_xarray screws up the time array

    # Add solar position variables as coordinates to the NWP Dataset.
    data_out = data.assign_coords(
        {
            "zenith": solpos_ds["zenith"],
        },
    )

    return data_out


def load_surfrad(
    datapath: str,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    time_dim: str,
    data_vars: list = None,
):
    """
    Wrapper function to load surfrad data, cast it to xarray, \
    and add clear-sky radiation fields using pvlib.

    Inputs:

    datapath: string
        Directory where the "year" directories are for surfrad data.
    datetime_start: pandas Timestamp
        Start date.
    datetime_end: pandas Timestamp
        End date.
    time_dim: string
        identifier that the time index is renamed to (e.g. "valid_time")
    data_vars: list
        list of strings corresponding to variables in the surfrad data.

    Outputs:

    ds: xarray Dataset
        Loaded surfrad data with clear-sky radiation fields and \
        solar position appended as coordinates.
    metadata: dictionary
        Dictionary with metadata from the raw surfrad observations.
    """

    df, metadata = load_surfrad_for_datetime_range(
        datapath=datapath,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        data_vars=data_vars,
    )

    surfrad_ds = df.to_xarray().rename(index=time_dim)
    surfrad_ds[time_dim] = pd.to_datetime(surfrad_ds[time_dim])

    surfrad_ds = add_clearsky_radiation(
        data=surfrad_ds,
        metadata=metadata,
        time_dim=time_dim,
    )

    surfrad_ds = add_solar_position(
        data=surfrad_ds,
        metadata=metadata,
        time_dim=time_dim,
    )

    return surfrad_ds, metadata


def nwp_load_save(
    chunk_timerange: np.array,
    surfrad_sitename: str,
    nwp_datavars: list,
    save_dir: str,
    nwp_datapath: str,
    surfrad_metadata: dict,
    datetime_start: pd.Timestamp,
    datetime_end: pd.Timestamp,
    nwp_notes: str,
):
    """
    This function wraps up different aspects of the data processing so that
    it can be called using dask.delayed to leverage parallel computing.

    Inputs:
    chunk_timerange: numpy array
        array of datetime-line objects for each day of data.
    surfrad sitename: string
        string identifier for reading in and writing out the preprocessed data.
    nwp_datavars: list
        list of string identifiers for NWP variables to extract.
    save_dir: string
        path to directory where data should be saved.
    nwp_datapath: string
        path to directory where NWP data is stored. \
        NWP data should be downloaded using download_fromEPRI.py
    surfrad_metadata: dictionary
        dictionary produced when preprocessing surfrad observations \
        used for subsetting NWP data.
    datetime_start: pandas Timestamp 
        timestamp for the overall period start
    datetime_end: pandas Timestamp
        timestamp for the overall period end
    nwp_notes: string
        string to be added to file attributes for documentation.
    """

    chunk_datestring = chunk_timerange[0].strftime('%Y%m%d_') + \
        chunk_timerange[-1].strftime('%Y%m%d')
    outfile_name = f"{chunk_datestring}_nwp_{surfrad_sitename}.nc"
    if os.path.exists(os.path.join(save_dir, outfile_name)):
        print("File already exists: ", os.path.join(save_dir, outfile_name))
        return

    ds_list = []

    # Load the single HRRR ensemble member
    iterlist = list(itertools.product(nwp_datavars, chunk_timerange))
    # Get all files to load for all variables and dates
    hrrr_filelist = []
    for _varname, _timestamp in iterlist:
        _hrrr_pathstr = os.path.join(
            nwp_datapath,
            "hrrr",
            "conus",
            _varname,
            f"{_timestamp.year:04d}",
            f"{_timestamp.month:02d}",
            f"{_timestamp.day:02d}",
            "",
        )
        hrrr_filelist.extend(glob.glob(f"{_hrrr_pathstr}*.grib2"))

    if len(hrrr_filelist) > 0:
        hrrr_ds = load_and_subset_nwp(
            filelist=hrrr_filelist,
            lat=surfrad_metadata["latitude"],
            lon=surfrad_metadata["longitude"],
        )

        # Perform averaging before saving.
        hrrr_ds = instantaneous_to_average(
            hrrr_ds,
            dim="valid_time",
        )

        hrrr_ds = hrrr_ds.sel(valid_time=slice(
            datetime_start,
            datetime_end,
        ))

        hrrr_ds = hrrr_ds.assign_coords({"nwp_source": "hrrr"}).expand_dims("nwp_source")
        ds_list.append(hrrr_ds)

    # Load RRFS forecasts (1 control and 5 ensemble members)
    iterlist = list(itertools.product(nwp_datavars, chunk_timerange))

    for _member in nwp_members:
        rrfs_filelist = []
        for _varname, _timestamp in iterlist:
            _rrfs_pathstr = os.path.join(
                nwp_datapath,
                "rrfs",
                "conus",
                _member,
                _varname,
                f"{_timestamp.year:04d}",
                f"{_timestamp.month:02d}",
                f"{_timestamp.day:02d}",
                "",
            )
            rrfs_filelist.extend(glob.glob(f"{_rrfs_pathstr}*.grib2"))

        if len(rrfs_filelist) > 0:
            rrfs_ds = load_and_subset_nwp(
                filelist=rrfs_filelist,
                lat=surfrad_metadata["latitude"],
                lon=surfrad_metadata["longitude"],
            )

            # Only take forecast data in the date range.
            rrfs_ds = rrfs_ds.sel(valid_time=slice(
                datetime_start,
                datetime_end,
            ))

            # Perform averaging before saving.
            rrfs_ds = instantaneous_to_average(
                rrfs_ds,
                dim="valid_time",
            )
            rrfs_ds = rrfs_ds.assign_coords({"nwp_source": f"rrfs_{_member}"}).expand_dims("nwp_source")
            ds_list.append(rrfs_ds)

    # Join the NWP data and add and appropriate note on the averages.
    nwp_ds = xr.merge(ds_list)
    nwp_ds = nwp_ds.assign_attrs({"note": nwp_notes})

    if not os.path.exists(os.path.join(save_dir, outfile_name)):
        nwp_ds.to_netcdf(os.path.join(save_dir, outfile_name))
        print(f"Created: {os.path.join(save_dir, outfile_name)}")
    else:
        print("File already exists: ", os.path.join(save_dir, outfile_name))
    del ds_list, nwp_ds
    try:
        del rrfs_ds
    except Exception:
        pass
    try:
        del hrrr_ds
    except Exception:
        pass


# %%

if __name__ == "__main__":

    # Specify input and output fields and naming.
    save_path = "data/processed_timeseries"

    surfrad_datapath = "data/surfrad"
    surfrad_sitename = "sxf"
    surfrad_datavars = [
        "ghi",
        "dhi",
        "dni",
        "ghi_flag",
        "dhi_flag",
        "dni_flag",
        "pressure",
    ]

    nwp_datavars = [
        "dswrf",
        "vbdsf",
        "vddsf",
    ]
    nwp_datapath = "data/nwp"
    nwp_notes = "These 'average-like' data were computed from hourly-resolution \
instantaneous values by interpolating to 5-minute resolution and computing an \
hourly mean value."
    nwp_members = ["control", "mem0001", "mem0002", "mem0003", "mem0004", "mem0005"]

    # Handle large time ranges differently here to reduce memory load.
    nwp_chunksize = 5  # number of days to process at once

    year_start = 2024
    month_start = 3
    day_start = 1

    year_end = 2024
    month_end = 6
    day_end = 3

    datetime_start = pd.Timestamp(
        year=year_start,
        month=month_start,
        day=day_start,
        hour=0,
        minute=0,
    )
    datetime_end = pd.Timestamp(
        year=year_end,
        month=month_end,
        day=day_end,
        hour=0,
        minute=0,
    ) + timedelta(days=1)

    datestring = datetime_start.strftime('%Y%m%d_') + datetime_end.strftime('%Y%m%d')
    save_dir = os.path.join(
        save_path,
        surfrad_sitename,
        datestring,
    )
    outfile_name = f"{datestring}_surfrad_{surfrad_sitename}.nc"
    if os.path.exists(os.path.join(save_dir, outfile_name)):
        print("File already exists: ", os.path.join(save_dir, outfile_name))
        surfrad_ds = xr.open_dataset(os.path.join(save_dir, outfile_name))
        surfrad_metadata = surfrad_ds.attrs
        del surfrad_ds
    else:
        # Load surfrad data corresponding to HRRR time period.
        surfrad_ds, surfrad_metadata = load_surfrad(
            datapath=f"{surfrad_datapath}/{surfrad_sitename}/",
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            time_dim="valid_time",
            data_vars=surfrad_datavars,
        )
        surfrad_ds = surfrad_ds.sel(valid_time=slice(
            datetime_start,
            datetime_end,
        ))
        surfrad_ds = surfrad_ds.assign_attrs(surfrad_metadata)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created: {save_dir}")
        surfrad_ds.to_netcdf(os.path.join(save_dir, outfile_name))
        del surfrad_ds

    # Load and concatenate all NWP data. Go 2 days ahead for forecast overlap.
    nwp_timerange = pd.DatetimeIndex(
        np.arange(
            datetime_start + np.timedelta64(-2, "D"),
            datetime_end,
            np.timedelta64(1, "D"),
        )
    )

    i = 0
    while i <= len(nwp_timerange):
        chunk_timerange = nwp_timerange[i:i + nwp_chunksize]
        i += nwp_chunksize

        nwp_load_save(
            chunk_timerange,
            surfrad_sitename,
            nwp_datavars,
            save_dir,
            nwp_datapath,
            surfrad_metadata,
            datetime_start,
            datetime_end,
            nwp_notes,
        )
