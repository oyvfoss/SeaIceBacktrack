# Sea ice back-trajectory calculations
## Overview

Working repository for calculation of sea ice back-trajectories using the NSIDC [Polar Pathfinder 
Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4 
(NSIDC-0116)](https://nsidc.org/data/nsidc-0116/versions/4) product (*"NSIDC PP"*).

The code in this repository was written in order to calculate back-trajectories from the northwestern Barents Sea, and the code is tailored to that application.
### *Scripts*

#### **Define particle tracking class.ipynb**

Defines the *ice_particle* class used to create individual back-trajectories. 

The class functions contain the code for initialization, interpolation, back-propagation, and various utility functions.  

#### **Load and prepare data.ipynb**

Load sea ice drift data and land mask. 

- Loads the NSIDC PP data to an xarray Dataset. Multiple files are coincatenated to a single file, and interpolation is done across gaps.  
- Loads a land mask (optional) to be used to end trajectories that intersect coastlines. 

Should be modified depending on system configuration/file location/desired land mask...

### *Data*

#### *Sea ice drift data*

NSIDC PP Data were downloaded from [NSIDC](https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/north/daily/) with Earthdata access.

Files containing daily drift vectors are on the form:

``icemotion_daily_nh_25km_20120101_20121231_v4.1.nc``

  In the code examples, there are references to the ``ddir`` directory, which is set to my local directory containing these files.  ``ddir`` should be changed 
  to point to the location of the icemotion files.

#### *Coastline data*

The code takes a land mask as an (optional) input. The land mask, *MASK*, is used to terminate the trajectory calculation if the particle hits land.

In the applications here, the land mask *MASK*,  was made from an Eastern Arctic Ocean subset of the IBCAO-v4 bathymetry, downscaled to 2 km resolution. The Svalbard archipelago was removed because we did not want to terminate trajectories intersecting land there (because we are computing backtrajectories from moorings close to land in northeastern Svalbard). 

*MASK* can be replaced with any land mask containing *x, y, is_land(x, y)* and thr projection *proj* of *x, y* (see details in *Load and prepare data.ipynb*).


### *Application specifics*

- Performs daily back-propagation from an initial coordinate point.
- Sea ice drift velocity at the sea ice particle coordinate is obtained by interpolating 
  the drift components onto the coordinate

### *Some key dependencies*

- [pyproj](https://pyproj4.github.io/pyproj/stable/) is used for transforms between coordinate systems (e.g. WGS84 lat/lon and NSIDC PP EASE-grid x/y).
- [xarray](https://docs.xarray.dev/en/stable/) is used for handling NetCDF data.
- [scipy.interpolate.griddata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata) 
  is used for linear interpolation between unstructured points.
