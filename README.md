# RRFS
This repo contains code for evaluating the Rapid Refresh Forecast System (RRFS) and High Resolution Rapid Refresh (HRRR) models for energy applications.

Instructions for running:

1\. Set up a python virtual environment using requirements.txt

> *Using python virtual environments and pip:*

> python -m venv ./rrfs_venv  
> ./rrfs_venv/Scripts/activate  
> pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install pip-system-certs  
> pip install -r requirements.txt

> *Using conda:*

> conda create --name rrfs_venv --file requirements.txt  
> conda activate rrfs_venv

Or:
> conda env create -f environment.yaml

2\. Download Surfrad observations and NOAA NWP forecasts.

* Download single-variable NWP forecast data from an EPRI server.
* Download daily observations from different SURFRAD sites for forecast evaluation.

> *Run:*  
> python download.py

3\. Pre-process data subsettting to Surfrad locations.

* Process observation and model data into comparable timeseries.
* NWP data is subset at the nearest model gridcell to the observation site.
* Interpolation and averaging is performed to produce "comparable" hourly averages.

> *Run:*  
> python preprocess.py

4\. Produce analysis figures.  
* Produce a suite of analysis figures from the preprocessed time series data.

> *Run:*  
> python analyze.py

5\. Optional analysis of partly cloudy days at the Goodwin Creek, MS (gwn) surfrad site.  

* Look at how the different NWP products forecast GHI in a subset of partly cloudy days.

> *Run:*  
> python ramp_casestudies.py