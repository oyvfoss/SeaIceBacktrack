#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import datetime, sys, netCDF4, os, csv, pickle, glob
import calendar, subprocess, itertools
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from copy import deepcopy
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature
import trajectory_util
from joblib import Parallel, delayed

register_matplotlib_converters()
plt.style.use('seaborn-white')


def main():
    """
    Calculate backward trajectories starting from Fram Strait,
    using class method of trajectory_util.Particle.
    The code is tested by "ice_trajectory_test6.py".

    Output
    ------
    FS.YYYY.MM.backward_traj.DDD_days.n_NN.pkl
            The file contains list of DDD days backward trajectories 
            of NN particles.
            Each particle is a trajectory_util.Particle object.

    by Hiroshi Sumata / 2020.07.02
    """
    # Initialization ----

    verbose = True                             # output verbose message
    save_dir = './output/'                     # directory for data saving
    
    # Define experiment parameters ---

    #years = list(reversed(range(2000, 2019)))  # years for experiments (2000-2018)
    years = list(reversed(range(1990, 2000)))  # years for experiments (1990-1999)
    #months = [i for i in range(1, 13)]         # months for experiments
    months = [9, 3, 6, 12, 7, 8, 1, 2, 10, 11, 4, 5]
    tracking_days = -731                       # number of tracking days
    #tracking_days = -100
    
    fs_lat = 78.8     # latitude of FS, starting latitude of backward trajectories
    #fs_east = 10.25   # eastern edge of FS, longitude [deg]
    #fs_west = -17.5   # western edge of FS, longitude [deg]
    #lon_div = 0.5     # division of initial particle positions in longitude [deg]
    
    #init_lons = [lon for lon in np.arange(fs_west, fs_east, lon_div)]

    init_lons = [-10.0, -8, -6.5, -5, -4, -3, -1.5, 0]
    
    num_particles = len(init_lons)
    print('num_particles =', num_particles)

    step = 1 if tracking_days >= 0 else -1
    
    # experiment loop ----

    for month in months:    
        for year in years:

            # define output file names ----
            #
            # format of filename: 'FS.YYYY.MM.backward_traj.DDD_days.n_NN.pkl'
            #
            #                     YYYY: year   
            #                     MM  : month
            #                     DDD : number of trajectory calculated days
            #                     NN  : number of particles

            cyear = str(year)
            cmonth = str(month) if month > 9 else '0' + str(month)
            cdays = str(-tracking_days)
            fname = 'FS.' + cyear + '.' + cmonth \
                    + '.backward_traj.' + cdays + '_days.n_' \
                    + str(num_particles) + '.pkl'

            print('calculating: ', fname)

            date_init = datetime.date(year, month, 15)
            
            trajectories = Parallel(n_jobs = -1, verbose = 3)(
                [delayed(trajectory_util.ice_trajectory)
                 ('lon_' + str(-lon) + '_degW', date_init, lon, fs_lat, tracking_days)
                 for (n, lon) in enumerate(init_lons)])
            
            #print('backward trajectory calculation ended:',
            #      ' year =', year, ' month =', month)
            with open(save_dir + fname, 'wb') as f:
                pickle.dump(trajectories, f)
        
    print('end of program')
    
if __name__ == '__main__':
    main()
