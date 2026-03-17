"""
Functions used to download single variable grib2 files from
the EPRI server where they have been downloaded using curl.

Contains:
- download_nwp_variable
- create_save_dir
- get_filepath
- get_filename

Files are NOAA NWP forecasts (HRRR, RRFS) for a specific
domain (e.g Continental US (CONUS)).

The files are located in this base directory:
http://top-pa-tower.epri.com/nwp/
Access requires being on-campus or connected via the VPN.

The directory structure is:
- {nwp_model}/
    - {domain}/
        - {member}/
            - {variable name}/
                - {year}/
                    - {month}/
                        - {dayofmonth}/

The data directories contain forecast files initialized on the date
given by the directory structure.

The file structure is:
{model}.t{forecast time}z.{product}.f{forecast horizon}.{domain}.grib2

So a full URL could look like:
http://top-pa-tower.epri.com/nwp/rrfs/conus/control/dswrf/2024/05/29/rrfs.t12z.prslev.f011.conus.grib2

"""

import os
import pandas as pd
import numpy as np
from ftplib import FTP
from datetime import timedelta
import itertools


def download_nwp_variable(
    filename: str,
    filepath: str,
    save_dir: str,
    overwrite: bool = False,
):
    """
    Download a single variable from the EPRI server.

    Parameters
    ----------
    filename : str
        Name of the grib2 file, but on the server and to be named.
    filepath : str
        URL path to the directory of the file on the EPRI server.
    save_dir : str
        Folder where to save the grib2 files.
    overwrite : bool
        If the file already exists locally, should you overwrite the file. True
        means re-download from AWS and overwrite; False means don't
        re-download.

    Returns
    -------
    error_message: 0 if the download was successful. An error code if not.
        Curl error codes described at https://curl.se/libcurl/c/libcurl-errors.html")

    Notes
    -----
    This function downloads the grib2 data via curl and saves the output to a
    local grib2 file. Therefore, the function does not return any values.

    """

    url = f"{filepath}/{filename}"
    out_file = os.path.join(save_dir, filename)
    if os.path.exists(out_file):
        print(f"{out_file} already exists.")
        return 0

    curl = f"curl -s -f {url} -o {out_file}"
    if overwrite:
        error_message = os.system(curl)
    else:
        if not os.path.exists(out_file):
            error_message = os.system(curl)
        else:
            print("File already exists:", out_file)
    return error_message


def create_save_dir(
    save_dir: str,
    init_time: pd.Timestamp,
    model: str,
    domain: str,
    member: str,
    varname: str,
):
    """
    Create save directory if it doesn't already exist.

    Parameters
    ----------
    save_dir: string
        directory where data should be saved
    init_time : pd.Timestamp
        Timestamp (UTC) of the initialization time of the NWP run
        e.g., 2024-02-01 06:00.
    model: string
        E.g. {'rrfs', 'hrrr'}
    domain: string
        {'conus', 'pr', }
    member: string
        {"control", "mem001", "mem002", "mem003", "mem004", "mem005"}
        Deterministic forecast (control) or ensemble member.
    varname: string
        Variable name. e.g. dswrf

    Returns
    -------
    save_dir : str
        Save directory filepath (e.g., nwp/rrfs/conus/control/dswrf/2024/05/29/)

    Notes
    -----
    Creates save directories following the structure of:
    The directory structure is:
    - {model}/
        - {domain}/
            - {member}/
                - {varname}/
                    - {year}/
                        - {month}/
                            - {dayofmonth}/

    """

    if member is not None:
        save_dir = os.path.join(
            save_dir,
            model,
            domain,
            member,
            varname,
            f"{init_time.year:04d}",
            f"{init_time.month:02d}",
            f"{init_time.day:02d}",
        )
    else:
        save_dir = os.path.join(
            save_dir,
            model,
            domain,
            varname,
            f"{init_time.year:04d}",
            f"{init_time.month:02d}",
            f"{init_time.day:02d}",
        )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created: {save_dir}")

    return save_dir


def get_filepath(
    init_time: pd.Timestamp,
    model: str,
    domain: str,
    member: str,
    varname: str,
    base_url: str = "http://top-pa-tower.epri.com/nwp/",
):
    """
    Create save directory if it doesn't already exist.

    Parameters
    ----------
    init_time : pd.Timestamp
        Timestamp (UTC) of the initialization time of the NWP run
        e.g., 2024-02-01 06:00.
    model: string
        {'rrfs', 'hrrr'}
    domain: string)
        {'conus', 'pr', }
    member: string
        {"control", "mem001", "mem002", "mem003", "mem004", "mem005"}
        Deterministic forecast (control) or ensemble member.
    varname: string
        Variable name. e.g. dswrf
    base_url: string
        URL where the data is hosted.

    Returns
    -------
    urlpath : str
        URL to the directory containing desired NWP files

    Notes
    -----
    Creates save directories following the structure of:
    The directory structure is:
    - {model}/
        - {domain}/
            - {member}/
                - {varname}/
                    - {year}/
                        - {month}/
                            - {dayofmonth}/

    """

    if member is not None:
        urlpath = f"{base_url}/{model}/{domain}/{member}/{varname}/{init_time.year:04d}/{init_time.month:02d}/{init_time.day:02d}"
    else:
        urlpath = f"{base_url}/{model}/{domain}/{varname}/{init_time.year:04d}/{init_time.month:02d}/{init_time.day:02d}"

    return urlpath


def get_filename(
    init_time: pd.Timestamp,
    model: str,
    product: str,
    horizon: int,
    domain: str,
    member: str = None,
):
    """
    Create the filename as on the EPRI tower server.

    Parameters
    ----------
    init_time : pd.Timestamp
        Timestamp (UTC) of the initialization time of the NWP run
        e.g., 2024-02-01 06:00.
    model: string
        {'rrfs', 'hrrr'}
    product: string
        e.g. 'psrlev'
    horizon: int
        Duration after the initialization when the forecast is valid.
    domain: string
        {'conus', 'pr', }
    member: string
        Optional string for RRFS ensemble data.
        e.g. {'control','mem0001','mem0002','mem0003','mem0004','mem0005'}

    Returns
    -------
    filename : str
        The filename on the EPRI tower server.

    Notes:

    The file structure is inconsistent both between control and ensemble members
    and with respect to time.

    For control members and ensemble members prior to 2024/04/20:
    {model}.t{forecast time}z.{product}.f{forecast horizon}.{domain}.grib2

    For ensemble members after 2024/04/20:
    {model}.t{forecast}z.m{member}.{product}.f{horizon:03d}.{domain}.grib2
    """

    if model == "rrfs":
        if (member is None) or (member == "control") or (init_time < pd.Timestamp("2024-04-20")):
            filepath = f"{model}.t{init_time.hour:02d}z.{product}.f{horizon:03d}.{domain}.grib2"
        else:
            filepath = f"{model}.t{init_time.hour:02d}z.m{member[-2:]}.{product}.f{horizon:03d}.{domain}.grib2"

    if model == "hrrr":
        filepath = f"{model}.t{init_time.hour:02d}z.{product}f{horizon:02d}.grib2"

    return filepath


def download_surfrad_data(
    outdir: str,
    station: str,
    startdate: pd.Timestamp,
    enddate: pd.Timestamp,
    saveover: bool = True,
):
    """
    Save surfrad data to a local machine using ftp.

    Inputs:

    outdir: string
        Directory path where surfrad data should be saved.
        If none then the current working directory will be used.

    station: string
        Name of the station's directory on the gml ftp server.
        Make sure that files are also named by this convention!
        E.g. Use "gwn" instead of "Goodwin_Creek_MS".
        See here: https://gml.noaa.gov/aftp/data/radiation/surfrad/

    startdate: pd.Timestamp
        Date when data should first be selected.

    enddate: pd.Timestamp
        Date when data should first be selected through.

    saveover: boolean
        Whether to overwrite data if a file already exists.

    Note: surfrad data is saved in daily .dat files. Cannot select data with
    more precision than daily.
    """

    surfrad_ftp_path = "data/radiation/surfrad"

    ftp = FTP("ftp.gml.noaa.gov")
    ftp.login()
    ftp.cwd(f"{surfrad_ftp_path}/{station}")

    os.makedirs(os.path.join(outdir, station), exist_ok=True)

    dates = pd.DatetimeIndex(np.arange(startdate,
                                       enddate+timedelta(days=1),
                                       timedelta(days=1)
                                       )
                             )
    # Determine files for each year separately due to directory structure.
    years = list(set(dates.year))
    years.sort()

    # Iterate over years:
    for _year in years:
        ftp.cwd(f"{_year}")
        ftp_files = ftp.nlst()
        os.makedirs(os.path.join(outdir, station, str(_year)), exist_ok=True)
        _ydates = dates[dates.slice_indexer(f'{_year}-01-01', f'{_year}-12-31')]

        for _date in _ydates:
            _file_str = f"{station}{_date.strftime('%y%j')}.dat"

            if (_file_str not in ftp_files):
                print(f"File \"{_file_str}\" not found for \"{_date.strftime('%Y%m%d')}\"")
                continue

            _save_path = os.path.join(
                outdir,
                station,
                str(_year),
                _file_str,
            )

            if not os.path.exists(_save_path):
                with open(_save_path, "wb") as local_file:
                    ftp.retrbinary(f"RETR {_file_str}", local_file.write)
            elif saveover:
                with open(_save_path, "wb") as local_file:
                    ftp.retrbinary(f"RETR {_file_str}", local_file.write)

        # Step back before going to the next years
        ftp.cwd("../")


if __name__ == "__main__":

    # Set up downloads by specifying a time range and download directories.
    year_start = 2024
    month_start = 3
    day_start = 1
    hour_start = 12

    year_end = 2024
    month_end = 6
    day_end = 3
    hour_end = 12

    surfrad_savedir = os.path.join("data", "surfrad")
    nwp_savedir = os.path.join("data", "nwp")

    data_datetime_start = pd.Timestamp(year_start, month_start, day_start, hour_start)
    data_datetime_end = pd.Timestamp(year_end, month_end, day_end + 1, hour_end)

    surfrad_sites = [
        "bon",
        "tbl",
        "dra",
        "fpk",
        "gwn",
        "psu",
        "sxf",
    ]
    for _site in surfrad_sites:
        download_surfrad_data(
            outdir=surfrad_savedir,
            station=_site,
            startdate=data_datetime_start,
            enddate=data_datetime_end,
            saveover=False,
        )

    # download HRRR forecasts initialized at 12Z for radiation fields.
    model = "hrrr"
    product = "wrfsfc"
    domain = "conus"
    members = [None]
    varnames = [
        "dswrf",
        "vbdsf",
        "vddsf",
    ]
    init_times = pd.date_range(
        start=data_datetime_start, end=data_datetime_end, freq="24h"
    )
    iterlist = list(itertools.product(varnames,members,init_times))
    for varname, member, init_time in iterlist:
        print("==>", member, init_time)
        save_dir = create_save_dir(
            nwp_savedir,
            init_time,
            model,
            domain,
            member,
            varname,
        )
        filepath = get_filepath(
            init_time,
            model,
            domain,
            member,
            varname,
        )
        min_fxx = 16
        max_fxx = 48
        for fxx in range(min_fxx, max_fxx + 1):
            try:
                filename = get_filename(
                    init_time,
                    model,
                    product,
                    fxx,
                    domain,
                    member,
                )
                err = download_nwp_variable(
                    filename,
                    filepath,
                    save_dir,
                )
            except Exception as e:  # This doesn't seem to be getting triggered...
                print("Error for:", member, init_time, fxx)
                print(e)
            if err != 0:
                print(f"Download failed for {filepath}/{filename} with error code {err}")

    # download RRFS deterministic forecasts initialized at 12Z
    model = "rrfs"
    product = "prslev"
    domain = "conus"
    members = [
        "control",
        "mem0001",
        "mem0002",
        "mem0003",
        "mem0004",
        "mem0005",
    ]
    varnames = [
        "dswrf",
        "vbdsf",
        "vddsf",
    ]
    init_times = pd.date_range(
        start=data_datetime_start, end=data_datetime_end, freq="24h"
    )
    # Use itertools to avoid nested for loops.
    # varname, member, and init_time are independently needed
    # to identify the directory.
    iterlist = list(itertools.product(varnames, members, init_times))
    for varname, member, init_time in iterlist:
        print("==>", member, init_time)
        save_dir = create_save_dir(
            nwp_savedir,
            init_time,
            model,
            domain,
            member,
            varname,
        )
        filepath = get_filepath(
            init_time,
            model,
            domain,
            member,
            varname,
        )
        min_ffx = 16
        max_fxx = 48
        for fxx in range(min_ffx, max_fxx + 1):
            try:
                filename = get_filename(
                    init_time,
                    model,
                    product,
                    fxx,
                    domain,
                    member,
                )
                err = download_nwp_variable(
                    filename,
                    filepath,
                    save_dir,
                )
            except Exception as e:
                print("Error for:", member, init_time, fxx)
                print(e)
            if err != 0:
                print(f"Download failed for {filepath}/{filename} with error code {err}")
