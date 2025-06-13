"""
This script generates a map of the continental US with 7 random locations plotted on it.
BND* 	Bondville, Illinois 	40.05192° N 	88.37309° W 	230 m 	6 hours from UTC 	April 1994
TBL 	Table Mountain, Boulder, Colorado 	40.12498° N 	105.23680° W 	1689 m 	7 hours from UTC 	July 1995
DRA 	Desert Rock, Nevada 	36.62373° N 	116.01947° W 	1007 m 	8 hours from UTC 	March 1998
FPK 	Fort Peck, Montana 	48.30783° N 	105.10170° W 	634 m 	7 hours from UTC 	November 1994
GWN 	Goodwin Creek, Mississippi 	34.2547° N 	89.8729° W 	98 m 	6 hours from UTC 	December 1994
PSU 	Penn. State Univ., Pennsylvania 	40.72012° N 	77.93085° W 	376 m 	5 hours from UTC 	June 1998
SXF 	Sioux Falls, South Dakota 	43.73403° N 	96.62328° W 	473 m 	6 hours from UTC 	June 2003

SGP 	ARM Southern Great Plains Facility, Oklahoma 	36.60406° N 	97.48525° W 	314 m 	6 hours from UTC (not included)

Figure Caption:
Locations of the 7 SURFRAD sites used in this study. Station acronyms correspond the following sites:
Bondville, Illinois (BND), Table Mountain, Boulder, Colorado (TBL), Desert Rock, Nevada (DRA), Fort Peck, Montana (FPK), Goodwin Creek, Mississippi (GWN), Penn State University, Pennsylvania (PSU), and Sioux Falls, South Dakota (SXF). The Southern Great Plains Facility in Oklahoma (SGP) was not used in this study and is thus absent from this figure.

Cite as: NOAA Earth System Research Laboratory, 1995: Surface Radiation Budget (SURFRAD) Network Observations. [indicate subset used]. NOAA National Centers for Environmental Information. [access date].

"""

# %%

import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

figure_dir = "figures"

if __name__ == "__main__":

    # Create a figure with a map of the continental US
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-95, central_latitude=35))

    # Set the extent to cover the continental US
    ax.set_extent([-120, -70, 24, 50], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, edgecolor='gray')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)

    # Location coordinates for the 7 sites
    locations_dict = {
        'BND': (88.37, 40.05),  # Bondville, Illinois
        'TBL': (105.24, 40.12),  # Table Mountain, Boulder, Colorado
        'DRA': (116.02, 36.62),  # Desert Rock, Nevada
        'FPK': (105.10, 48.31),  # Fort Peck, Montana
        'GWN': (89.87, 34.25),   # Goodwin Creek, Mississippi
        'PSU': (77.93, 40.72),   # Penn State University, Pennsylvania
        'SXF': (96.62, 43.73)    # Sioux Falls, South Dakota
    }
    latlon_array = np.array(list(locations_dict.values()))

    # Plot the random locations
    ax.scatter(360 - latlon_array[:,0], latlon_array[:,1], color='red', s=80, marker='o', 
            transform=ccrs.PlateCarree(), zorder=10, label='SURFRAD Site Locations')

    # Annotate the random locations with their coordinates
    for label in locations_dict:
        _lon = locations_dict[label][0]
        _lat = locations_dict[label][1]
        ax.text(360 - _lon + 0.5, _lat + 0.5, f'{label}', 
                transform=ccrs.PlateCarree(), fontsize=14, color='red')

    # Add title and legend
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f'{figure_dir}/paper_figures/fig1_SURFRAD_locations.pdf',
        dpi=150,
        bbox_inches='tight'
    )
    # plt.show()

# %%