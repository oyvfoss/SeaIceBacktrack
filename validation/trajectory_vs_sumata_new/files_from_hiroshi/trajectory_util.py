#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime, calendar, sys, netCDF4, os, csv, pickle, math
import pyproj
import pandas as pd


#from geopy.distance import geodesic
from numpy import sin, cos, arcsin, arccos, arctan, sign, mod
import trajectory_util

# ------------------------------------------------------------------------------------
#    Python utility program to calculate sea ice trajectory
#
#    by Hiroshi Sumata / 2020.06.12

#    NOTE:  This utility code requires an environment consructed by
#           >> conda create -n py3.6c python=3.6 pyproj numpy scipy matplotlib netCDF4
#              pandas seaborn notebook cartopy"
#           The above environment is currently saved as "py3.6c"
#
#           *basemap should not be installed, otherwise it causes problem when it runs 
#            proj.CRS
#
#    History: 2020.05.12 First version
#
# ------------------------------------------------------------------------------------
#
# LIST of classes and functions defined in this utility code
#
# classes:
#
#    Parcicle

class Particle:
    """
    particles representing sea ice floe
                                           by Hiroshi Sumata / 2020.05.12

    Parameters
    ----------
    date_init   : datetime.date object, initial date of the particle
    lon_init    : float [deg], initial longitude of the particle
    lat_init    : float [deg], initial latitude of the particle

    Returns
    -------
    particle object

    Attributes
    ----------
    name        : character string, name of particle, e.g. to simulate buoy trajectory
    stamps      : list of datetime.date object, describing history of the particle
    pos_xys     : list of position [x, y] defined on EASE grid, list indices corresponds to stamps
    lonlats     : list of [lon, lat] pairs corresponding to 'pos_yxs'
    conc        : list of sea ice concentration correspoinding to stamps, [%]
    terminate   : termination flag. True if once ice conc. < 15% or encounter land mass
    termination_flag: character string describing reason for termination
    u           : list of eastward drift speed corresponding to stmaps, [m/s]
    v           : list fo northward drift speed correspoinding to stmaps, [m/s]
    heatings    : to be added later on
    divergences : to be added later on
    ...

    methods
    -------
    advect(date1, date2, daily_drift) : advect the particle from date1 to date2 by daily_drift


    """
    def __init__(self,date_init, lon_init, lat_init):
        """
        define initial date and position of a particle

        by Hiroshi Sumata / 2020.05.12
        """
        # check data consistency --

        if date_init <= datetime.date(1980, 1, 1):
            print('ERROR: trajectory_util.Particle, initialization error')
            print('       date_init out of range')
            print('       date_init =', date_init()) 
            sys.exit()
        if not (-180.0 <= lon_init <= 180.0):
            print('ERROR: trajectory_util.Particle, initialization error')
            print('       lon_init out of range')
            print('       lon_init =', lon_init)
            sys.exit()
        if not (-90.0 <= lat_init <= 90.0):
            print('ERROR: trajectory_util.Particle, initialization error')            
            print('       lat_init out of range')
            print('       lat_init =', gat_init)
            sys.exit()
        
        self.stamps = [date_init]
        self.lonlats = [[lon_init, lat_init]]

        # ATTENTION: pyproj.CRS requires pyproj version 1.9.6 or later,  
        #            which requires python 3.6 environment               
        #            See Physical Oceanography Note (114) p90 for description
        #
        # NOTE: Result of the following transformation (lon, lat) ==> (X, Y) is tested
        #       and confirmed to be consistent with (X, Y) <==> (lon, lat) correspondence
        #       described in NSIDCv4.1 netcdf file.
        #       See Physical Oceanography Note (114) p90 for detail
        #
        ease = pyproj.CRS("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 " \
                          + "+a=6371228 +b=6371228 +units=m +no_defs")  
        wgs84 = pyproj.CRS("EPSG:4326") # latlon with WGS84 datum
        x, y = pyproj.transform(wgs84, ease, lat_init, lon_init)

        self.pos_xys = [[x, y]]  # unit [m]
        self.u = [np.nan]        # set np.nan for starting point of trajectory
        self.v = [np.nan]        # set np.nan for starting point of trajectory
        self.conc = [np.nan]     # set np.nan for starting point of trajectory
        self.div = [np.nan]
        
        self.terminate = False
        self.termination_flag = ''

        print('initialization of particle -----------')
        print('    self.stamps :', self.stamps[-1])
        print('    self.lonlats:', self.lonlats[-1])
        print('    self.pos_xys:', self.pos_xys[-1])
        
        
    def advect(self, date1, date2, ice_drift, ice_concn):
        """
        calculate advection of the particle

        by Hiroshi Sumata / 2020.06.10

        inputs
        ------
        date1      : datetime.date object, start date of advection
        date2      : datetime.date object, end date of advection
        ice_drift  : list of trajectory_util.IceDrift objects, 
                     representing drift of [day_prev, day, nday_next]
        ice_concn  : list of trajectory_util.IceConcn objects,
                     representing concentration of [day_prev, day, nday_next] 

        attributes to be modified by the advection 
        ------------------------------------------
        stamps    : time stamp (datetime.date object) is added at the end of stamps
        positions : [lon, lat] is added at the end of positions
        status    : status of used data is added at the end of status
        drift_uv  : ice drift velocity of each day is added at the end of positions
        conc      : ice concentration of each day is added at the end of positions

        """

        verbose = False
        
        # consistency check ----

        if verbose: print('trajectory_util.advect: @00, consistency check') 
        if self.stamps[-1] != date1:
            print('ERROR: trajectory_util.Particle.advection (1)')
            print('       inconsistent time stamp')
            print('       self.stamps[-1] =', self.stamps[-1])
            print('       date1           =', date1)
            sys.exit()
            
        if ice_drift[1].stamp != date1:
            print('ERROR: trajectory_util.Particle.advection (2)')
            print('       inconsistent time stamp between date and ice_drift')
            print('       date1           =', date1)
            print('       ice_drift.stamp =', ice_drift.stamp)
            sys.exit()

        if ice_concn[1].stamp != date1:
            print('ERROR: trajectory_util.Particle.advection (3)')
            print('       inconsistent time stamp between date and ice_concn')
            print('       date1           =', date1)
            print('       ice_concn.stamp =', ice_concn.stamp)
            sys.exit()                  
                  
        # calculate dt ----

        if verbose: print('trajectory_util.advect: @01, define t and u, v')
        
        dt = (date2 - date1).total_seconds()   # unit: [sec]

        u    = ice_drift[1].u          # velocity in x-direction, [m/s]
        v    = ice_drift[1].v          # velocity in y-direction, [m/s]

        x_uv = ice_drift[1].x          # x-coordinate of drift data u & v, [m]
        y_uv = ice_drift[1].y          # x-coordinate of drift data u & v, [m]
        
        pos_x = self.pos_xys[-1][0]    # current x position of particle, [m]
        pos_y = self.pos_xys[-1][1]    # current y position of particle, [m]
        
        # define distance between particle position and UV grid points ----

        if verbose: print('trajectory_util.advect: @02, calculate interpolated u and v')
        
        dist_x = np.full_like(x_uv, pos_x) - x_uv  # 1D array of x-distance, [m]
        dist_y = np.full_like(y_uv, pos_y) - y_uv  # 1D array of y-distance, [m]

        dist = np.ndarray((y_uv.shape[0], x_uv.shape[0])) # 2D array of distance, [m]

        for (i, dx) in enumerate(dist_x):
            for (j, dy) in enumerate(dist_y):
                dist[j, i] = np.sqrt(dx**2 + dy**2)

        # define 2D weighting matrix for interpolation ----

        a = 25.0e3             # use grid size [m] as a measure of e-folding
        #a = 50.0e3             # use 2 * grid size [m] as a measure of e-folding
        
        weight = np.exp(-2.0 * (dist**2 / a**2))  # Gaussian-type function, Note(114) p93
                                                  
        # calculate distance weighted u and v at particle point ----

        mask = np.where(dist > 2.0 * a, np.nan, weight) # points within "2 * a" are used
        wmask = np.copy(mask)
        
        sum_weight = np.nansum(mask)
        if sum_weight > 0:
            u_weighted = np.nansum(u * mask) / sum_weight   # wieghted u at (pos_x, pos_y), [m/s]
            v_weighted = np.nansum(v * mask) / sum_weight   # weighted v at (pos_x, pos_y), [m/s]
        else:
            u_weighted = np.nan
            v_weighted = np.nan

        if verbose: print('trajectory_util.advect: @03, calculate advection')
            
        if u_weighted == u_weighted and v_weighted == v_weighted: # calculate advection if not np.nan
            deltax = u_weighted * dt   # advection in x-direction
            deltay = v_weighted * dt   # advection in y-direction

        else: # stop calculation if drift vector is not available within distance 'a'

            deltax = 0.0; deltay = 0.0
            self.terminate = True
            self.termination_flag = 'No ice drift data'
             
        # check sea ice concentration ----

        if verbose: print('trajectory_util.advect: @04, check ice conc')
        
        if ice_concn[1].nodata == True:
            if ice_concn[0].nodata == False:
                ice_concn[1] = ice_concn[0] # use date_prev if no data for the day
                print('NO DATA WARNING: ice_concn[1] is defined by ice_concn[0]')
                #ice_concn[0].nodata = True
            elif ice_concn[2].nodata == False:
                ice_concn[1] = ice_concn[2] # use date_next if no data for the day
                print('NO DATA WARNING: ice_concn[1] is defined by ice_concn[2]')                
                #ice_concn[0].nodata = True
            else:
                print('WARNING: there is no ice concentration data for 3 consecutive days!')
                print('         Effect of ice concentration is not taken into account')

        # define ice conc. at the particle's location ----
        
        if ice_concn[1].nodata == False:
            ice_conc = ice_concn[1]
            conc = ice_conc.conc                  # unit [%] 
            xc = ice_conc.xc; yc = ice_conc.yc    # x and y coordinate system for concentration
            dist_x = np.full_like(xc, pos_x) - xc # 1D array of x-distance, [m]
            dist_y = np.full_like(yc, pos_y) - yc # 1D array of x-distance, [m]
            dist_conc = np.ndarray((xc.shape[0], yc.shape[0]))

            for (i, dx) in enumerate(dist_x):
                for (j, dy) in enumerate(dist_y):
                    dist_conc[j, i] = np.sqrt(dx**2 + dy**2)
                    
            a1 = 12.5e3  # user grid size [m] of conc. as a measure of e-folding scale
            weight = np.exp(-2.0 * (dist_conc**2 / a1**2))

            # calculate distance-weighted conc at particle point ----

            mask = np.where(dist_conc > 2.0 * a1, np.nan, weight)
            sweight = np.nansum(mask)
            conc_w = np.nansum(conc * mask) / sweight if sweight > 0.0 else 0.0 # weighted conc [%]
                
            # stop calculation if ice conc. < 15% ----
            #
            # Since np.nansum gives zero for sum of np.nans, calculation stops as well 
            # when the particle encounters into land mass.
            
            if conc_w < 15.0:
                self.terminate = True
                self.termination_flag = 'ice conc. lower than 15%'

        else: # ice conc. data is not available for 3 consecutive days ---
            conc_w = np.nan
                
        # update particle location and ice properties ----

        if verbose: print('trajectory_util.advect: @05, updating particle properties')
        
        self.pos_xys.append([pos_x + deltax, pos_y + deltay])
        self.stamps.append(date2)
        self.u.append(u_weighted)
        self.v.append(v_weighted)
        self.conc.append(conc_w)
        self.div.append(div(ice_drift[1], wmask))
        
        ease = pyproj.CRS("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 " \
                          + "+a=6371228 +b=6371228 +units=m +no_defs")  
        wgs84 = pyproj.CRS("EPSG:4326")  # latlon with WGS84 datum
        lat, lon = pyproj.transform(ease, wgs84, pos_x + deltax, pos_y + deltay) 
        self.lonlats.append([lon, lat]) 

        # check advection ----

        if verbose:
            print('date1 =', date1, ',    date2 =', date2, '-----------')
            print('        u, v        =', u_weighted, v_weighted)
            print('        dx, dy      =', deltax, deltay)
            print('        pos_xys[-1] =', self.pos_xys[-1])
            print('        lonlats[-1] =', self.lonlats[-1])
            print('        ice conc.   =', self.conc[-1])
            print('        ice div     =', self.div[-1])
            print('')

        
class Empty:
    """
    class of empty object

    by Hiroshi Sumata / 2020.03.31
    """
    def __init__(self):
        pass

    
class IceDrift:
    """
    Class definition of IceDrift object

    by Hiroshi Sumata / 2020.06.12

    inputs
    ------
    date        : datetime.date object, define time of the ice drift field
    product_name: character, name of ice drift product

    outputs
    -------
    IceDrift object

    attributes
    ----------
    stamp     : datetime.date object representing the date of the ice drift field
    x         : projection x coordinate
    y         : projection y coordinate
    lons      : [y, x] 2-dimensional np.ndarray giving longitude of the used coordinate system, [deg]
    lats      : [y, x] 2-dimensional np.ndarray giving latitude of the used coordinate system, [deg]
    u         : [y, x] 2-d np.ndarray, sea ice x velocity, [m/s]
    v         : [y, x] 2-d np.ndarray,o sea ice y velocity, [m/s]
    div       : 2-d field of convergence/divergence, to be defined later on..

    """
    def __init__(self, date, product_name):
        self.stamp = date
        self.product_name = product_name
    
        if self.product_name == 'NSIDCv4p1':

            # check time stamp ----

            fname = get_NSIDCv4p1_fname_ease(self.stamp)

            if os.path.exists(fname):

                # read NetCDF file ---
                
                nc = netCDF4.Dataset(fname, 'r')
                self.proj4_string = nc.variables['crs'].proj4text
                self.x = nc.variables['x'][:]
                self.y = nc.variables['y'][:]
                self.lats = nc.variables['latitude'][:]
                self.lons = nc.variables['longitude'][:]
                nctime = nc.variables['time'][:]
                t_calendar = nc.variables['time'].calendar
                t_units = nc.variables['time'].units

                ncstamp = [datetime.datetime.strptime(str(netCDF4.num2date(time, \
                           units = t_units))[0:10], '%Y-%m-%d').date() \
                           for time in nctime]
                index = ncstamp.index(date)
                
                u = nc.variables['u'][index, :, :] 
                v = nc.variables['v'][index, :, :] 
                fill_value = nc.variables['u']._FillValue

                u = np.where(u <= fill_value * 0.1, np.nan, u)
                v = np.where(v <= fill_value * 0.1, np.nan, v)
                
                self.u = u * 1.0e-2 # [y, x],  unit [m/s]                
                self.v = v * 1.0e-2 # [y, x],  unit [m/s]
                
            else:
                sys.exit('ERROR: data does not exit, \    fname =' + fname)
                
        else:
            sys.exit('ERROR: ice drift product named ' + product_name + ' is not defined')



class IceConcn:
    """
    class definition of IceConcn object for trajectory calculation

    by Hiroshi Sumata / 2020.06.11
    
    inputs
    ------
    date        : datetime.date object, define time of the ice concentration field
    product_name: character, name of ice concentration product

    outputs
    -------
    IceConcn object

    attributes
    ----------
    proj4_strig : proj4 string
    stamp       : datetime.date object representing the date of the ice concentration field
    xc          : x-coordinate projection (eastings), unit: [km]
    yc          : y-coordinate projection (northings), unit: [km]
    lons        : 2-dimensional np.ndarray giving longitude of the used coordinate system
    lats        : 2-dimensional np.ndarray giving latitude of the used coordinate system
    conc        : 2-d np.ndarray, sea ice concentration,  unit [%]

    """
    def __init__(self, date, product_name):
        self.stamp = date
        self.product_name = product_name

        if self.product_name == 'OSI-409-430':

            fname = get_OSI409_430_fname_ease(self.stamp)
            
            # read netcdf file of the day ----

            if os.path.exists(fname):
                nc = netCDF4.Dataset(fname, 'r')
                self.nodata = False
                self.proj4_string = nc.variables['Lambert_Azimuthal_Grid'].proj4_string
                self.xc = nc.variables['xc'][:] * 1.0e3   # unit [m]
                self.yc = nc.variables['yc'][:] * 1.0e3   # unit [m]
                self.lats = nc.variables['lat'][:]
                self.lons = nc.variables['lon'][:]
                nctime = nc.variables['time'][:]
                t_units = nc.variables['time'].units
                t_cal = nc.variables['time'].calendar

                ncstamp = [datetime.datetime.strptime(str(netCDF4.num2date(time, \
                           units = t_units, calendar = t_cal))[0:10], '%Y-%m-%d').date() \
                           for time in nctime]
                t_index = ncstamp.index(self.stamp)

                conc = nc.variables['ice_conc'][:]       # unit: [0-100 %]
                fill_value = nc.variables['ice_conc']._FillValue

                self.conc = conc[t_index, :, :]
                self.conc = np.where(self.conc < fill_value * 0.1, np.nan, self.conc)
                nc.close()    
                    
            else:
                self.nodata = True

        else:
            sys.exit('ERROR: ice drift product named ' + product_name + ' is not defined')

    
def get_OSI409_430_fname(date):
    """
    function to return daily OSI-409 or OSI-430 filename

    by Hiroshi Sumata / 2020.06.10

    input
    -----
    date    : datetime.date object

    return
    ------
    fname   : character, describing path and filename of netcdf file

    """
    path_OSI409 = '/home/hiroshi/DATA_work/obs_data/sea_ice/concentration_OSI-409_v1p2_daily/'
    path_OSI430 = '/home/hiroshi/DATA_work/obs_data/sea_ice/concentration_OSI-430_v1p2_daily/'    

    cyear = str(date.year)
    c_mon = str(date.month) if date.month > 9 else '0' + str(date.month)

    fname_OSI409 = 'ice_conc_nh_polstere-100_reproc_YYYY.MM.mergetime.nc'
    fname_OSI430 = 'ice_conc_nh_polstere-100_cont-reproc_YYYYMM_mergetime.nc'

    if date < datetime.date(2015, 4, 1):
        fname = path_OSI409 + fname_OSI409.replace('YYYY', cyear).replace('MM', c_mon)
    else:
        fname = path_OSI430 + fname_OSI430.replace('YYYY', cyear).replace('MM', c_mon)
    
    return fname

def get_OSI409_430_fname_ease(date):
    """
    function to return daily ease grid OSI-409 or OSI-430 filename

    by Hiroshi Sumata / 2020.06.11

    input
    -----
    date    : datetime.date object

    return
    ------
    fname   : character, describing path and filename of netcdf file
    
    """

    path_OSI409 = '/home/hiroshi/DATA_work/ICE_CONCENTRATION/OSI-409_v1.2/' \
                  + 'reprocessed/ice/conc/v1p2/YYYY/MM/'
    path_OSI430 = '/home/hiroshi/DATA_work/ICE_CONCENTRATION/OSI-430/osisaf.met.no/' \
                  + 'reprocessed/ice/conc-cont-reproc/v1p2/YYYY/MM/'

    cyear = str(date.year)
    c_mon = str(date.month) if date.month > 9 else '0' + str(date.month)
    c_day = str(date.day) if date.day > 9 else '0' + str(date.day)
    
    if date < datetime.date(2015, 4, 16):
        fname = path_OSI409.replace('YYYY', cyear).replace('MM', c_mon) \
                + 'ice_conc_nh_ease-125_reproc_' + cyear + c_mon + c_day + '1200.nc'
    else:
        fname = path_OSI430.replace('YYYY', cyear).replace('MM', c_mon) \
                + 'ice_conc_nh_ease-125_cont-reproc_' + cyear + c_mon + c_day + '1200.nc'
                
    return fname
                

def get_NSIDCv4p1_fname_ease(date):
    """
    function to return file name of daily NSIDCv4.1 data defined on ease grid

    by Hiroshi Sumata / 2020.06.12

    input
    -----
    date   : datetime.date object

    return
    ------
    fname  : character, describing path and filename of netcdf file

    """

    path = '../../../ICE_DRIFT_DATA/NSIDCv4.1/icedrift_daily/daily/'

    cyear = str(date.year)
    c_mon = str(date.month) if date.month > 9 else '0' + str(date.month)
    c_day = str(date.day) if date.day > 9 else '0' + str(date.day)

    if datetime.date(1979, 1, 1) <= date <= datetime.date(2018, 12, 31):
        fname = path + 'icemotion_daily_nh_25km_YYYY0101_ZZZZ1231_v4.1.nc'
        fname = fname.replace('YYYY', cyear).replace('ZZZZ', cyear)

    else:
        sys.exit('ERROR: date of ice drift data is out of range, date =' + date)

    return fname


def div(ice_drift, wmask):
    """
    function to calculate convergence of sea ice motion at given date and location

    NOTE: The divergence field is check by fig_check_convergence.py and 
          fig_check_convergence_sensecutive_days.py in ../36_ice_drift_divergence/

    by Hiroshi Sumata / 2020.06.19

    inputs
    ------
    ice_drift : IceDrift object
    wmask     : 2D mask used to define interpolation weight of (u, v) field

    outputs
    -------
    div      : divergence of sea ice motion

    """

    u = ice_drift.u
    v = ice_drift.v

    x_uv = ice_drift.x    # x-coordinate of ice drift data u & v, [m]
    y_uv = ice_drift.y    # y-coordinate of ice drift data u & v, [m]

    # calculate divergence field ---
    
    dudx = np.gradient(u, x_uv, axis = 1)
    dvdy = np.gradient(v, y_uv, axis = 1)
    div = dudx + dvdy                        # unit [s^-1]

    # calculate distance weighted divergence at particle point ---

    sum_weight = np.nansum(wmask)
    div_weighted = np.nansum(div * wmask) / sum_weight  if sum_weight > 0 else np.nan # unit [s^-1]

    return div_weighted


def ice_trajectory(name, date_init, lon_init, lat_init, tracking_days):
    """
    sea ice trajectory calculation (forward or backward)

    by Hiroshi Sumata / 2020.06.22

    inputs
    ------
    name         : character string, name of particle
    date_init    : datetime.date object, giving the initial date of particle tracking
    lon_init     : float, initial longitude of ice particle
    lat_init     : float, initial latitude of ice particle
    tracking_days: integer, number of tracking days. if tracking_days < 0, then backward 
                   trajectory will be calculated.

    outputs
    -------
    particle     : particle object

    """
    particle = trajectory_util.Particle(date_init, lon_init, lat_init)
    particle.name = name
    step = 1 if tracking_days >= 0 else -1
    dates = [date_init + datetime.timedelta(days = n) for n in range(0, tracking_days, step)]
        
    cyear = str(date_init.year)
    c_mon = str(date_init.month) if date_init.month > 9 else '0' + str(date_init.month)
    c_day = str(date_init.day) if date_init.day > 9 else '0' + str(date_init.day)
    stamp = cyear + '-' + c_mon + '-' + c_day

    for (n, date) in enumerate(dates):
        if n < len(dates) - 1:

            if particle.terminate == False:

                date_prev = date + datetime.timedelta(days = - 1)
                date_next = date + datetime.timedelta(days = + 1)
        
                # get daily ice drift field ----

                product = 'NSIDCv4p1'
                ice_drift = [trajectory_util.IceDrift(date_prev, product),
                             trajectory_util.IceDrift(date,      product),
                             trajectory_util.IceDrift(date_next, product)
                             ]
                
                # get daily ice concentraton field ----
        
                product = 'OSI-409-430'
                ice_concn = [trajectory_util.IceConcn(date_prev, product),
                             trajectory_util.IceConcn(date,      product),
                             trajectory_util.IceConcn(date_next, product)
                             ]

                # advect particles ----
        
                particle.advect(date, dates[n + 1], ice_drift, ice_concn)
                           
            else:
                print('calculation finished, @0')
                break

        elif n == len(dates) - 1:
            print('calculation finished, @1')

    return particle               


def read_IABP(data_path, year, buoys_data):
    """
    Read IABP buoy position data

    by Hiroshi Sumata / 2020.06.19
                      / 2020.06.24: add exception rules 


    inputs
    ------
    path      : character string, data path
    year      : integer, year of IABP_C buoy data
    buoys_data: list of buoy data. 
    
    returns
    -------
    buoys: list of buoy object, buoy object is given by trajectory_util.Empty() class.
           attributes of buoy object is as follows;

           buoy.name : character, id of buoy
           buoy.dates: list of datetime.date object giving date of buoy track data
           buoy.lons : list of float, giving longitude of the buoy location
           buoy.lats : list of float, giving latitude of the buoy location
    """

    #data_path = './IABP_C/'
    fname = data_path + 'C' + str(year)

    buoy_names = []
    buoys = []
    
    # read existing buoy names in buoys_data:

    if os.path.exists(buoys_data):
        with open(buoys_data, 'rb') as f:
            buoys = pickle.load(f)
            for buoy in buoys:
                buoy_names.append(buoy.name)
    
    # gather buoy names ----
    
    f = open(fname, 'r')
    for (n, line) in enumerate(csv.reader(f)):
        buoy_id = line[0].split()[4]
        
        if not buoy_id in buoy_names:
            buoy_names.append(buoy_id)

            # define new buoy data if it doesn't exist ----
            
            buoy = trajectory_util.Empty()
            buoy.name = buoy_id
            buoy.dates = []
            buoy.lons = []
            buoy.lats = []
            buoys.append(buoy)            
    f.close()            

    # read data and put them into buoy object ----   
        
    f = open(fname, 'r')
    for (n, line) in enumerate(csv.reader(f)):
        year, month, day, hour, name, lat, lon = line[0].split()

        for buoy in buoys:
            if name == buoy.name and int(hour) == 12: # use hour == 12 data only !!
                
                if 1 <= int(day) <= 31 and -180.0 <= float(lon) <= 180 and 0.0 <= float(lat) <= 90.0:
                    #print(int(year), int(month), int(day))
                    date = datetime.date(int(year), int(month), int(day))
                    buoy.dates.append(date)
                    buoy.lons.append(float(lon))
                    buoy.lats.append(float(lat))
                else:
                    print('WARNING: erroneous data skipped:',
                          year, '-', month, '-', day,
                          ': name', name, ', lat, lon:', lat, lon)
    f.close()

    # save buoy data with a sufficient time record ----

    buoys_update = []
    for (n, buoy) in enumerate(buoys):
        if len(buoy.dates) > 5:
            buoys_update.append(buoy)
            
    # Save buoys_data ----

    with open(buoys_data, 'wb') as f:
        pickle.dump(buoys_update, f)

    return buoys_update


def advect(particle, date1, date2, ice_drift, ice_concn):
    """
    calculate advection of the particle 
    This function does the same thing with the particle.advect method as an independent function

    by Hiroshi Sumata / 2020.07.03
    
    inputs
    ------
    particle   : trajectory_util.Particle object
    date1      : datetime.date object, start date of advection
    date2      : datetime.date object, end date of advection
    ice_drift  : list of trajectory_util.IceDrift objects, 
                 representing drift of [day_prev, day, nday_next]
    ice_concn  : list of trajectory_util.IceConcn objects,
                 representing concentration of [day_prev, day, nday_next] 

    attributes to be modified by the advection 
    ------------------------------------------
    stamps    : time stamp (datetime.date object) is added at the end of stamps
    positions : [lon, lat] is added at the end of positions
    status    : status of used data is added at the end of status
    drift_uv  : ice drift velocity of each day is added at the end of positions
    conc      : ice concentration of each day is added at the end of positions

    returns
    -------
    particle   : updated particle object

    """

    verbose = False
        
    # consistency check ----

    if verbose: print('trajectory_util.advect: @00, consistency check') 
    if particle.stamps[-1] != date1:
        print('ERROR: trajectory_util.Particle.advection (1)')
        print('       inconsistent time stamp')
        print('       particle.stamps[-1] =', particle.stamps[-1])
        print('       date1           =', date1)
        sys.exit()
            
    if ice_drift[1].stamp != date1:
        print('ERROR: trajectory_util.Particle.advection (2)')
        print('       inconsistent time stamp between date and ice_drift')
        print('       date1           =', date1)
        print('       ice_drift.stamp =', ice_drift.stamp)
        sys.exit()
        
    if ice_concn[1].stamp != date1:
        print('ERROR: trajectory_util.Particle.advection (3)')
        print('       inconsistent time stamp between date and ice_concn')
        print('       date1           =', date1)
        print('       ice_concn.stamp =', ice_concn.stamp)
        sys.exit()                  
                  
    # calculate dt ----

    if verbose: print('trajectory_util.advect: @01, define t and u, v')
        
    dt = (date2 - date1).total_seconds()   # unit: [sec]

    u    = ice_drift[1].u          # velocity in x-direction, [m/s]
    v    = ice_drift[1].v          # velocity in y-direction, [m/s]

    x_uv = ice_drift[1].x          # x-coordinate of drift data u & v, [m]
    y_uv = ice_drift[1].y          # x-coordinate of drift data u & v, [m]
        
    pos_x = particle.pos_xys[-1][0]    # current x position of particle, [m]
    pos_y = particle.pos_xys[-1][1]    # current y position of particle, [m]
        
    # define distance between particle position and UV grid points ----

    if verbose: print('trajectory_util.advect: @02, calculate interpolated u and v')
        
    dist_x = np.full_like(x_uv, pos_x) - x_uv  # 1D array of x-distance, [m]
    dist_y = np.full_like(y_uv, pos_y) - y_uv  # 1D array of y-distance, [m]

    dist = np.ndarray((y_uv.shape[0], x_uv.shape[0])) # 2D array of distance, [m]

    for (i, dx) in enumerate(dist_x):
        for (j, dy) in enumerate(dist_y):
            dist[j, i] = np.sqrt(dx**2 + dy**2)

    # define 2D weighting matrix for interpolation ----

    a = 25.0e3             # use grid size [m] as a measure of e-folding
    #a = 50.0e3             # use 2 * grid size [m] as a measure of e-folding
        
    weight = np.exp(-2.0 * (dist**2 / a**2))  # Gaussian-type function, Note(114) p93
                                                  
    # calculate distance weighted u and v at particle point ----

    mask = np.where(dist > 2.0 * a, np.nan, weight) # points within "2 * a" are used
    wmask = np.copy(mask)
        
    sum_weight = np.nansum(mask)
    if sum_weight > 0:
        u_weighted = np.nansum(u * mask) / sum_weight   # wieghted u at (pos_x, pos_y), [m/s]
        v_weighted = np.nansum(v * mask) / sum_weight   # weighted v at (pos_x, pos_y), [m/s]
    else:
        u_weighted = np.nan
        v_weighted = np.nan

    if verbose: print('trajectory_util.advect: @03, calculate advection')
            
    if u_weighted == u_weighted and v_weighted == v_weighted: # calculate advection if not np.nan
        deltax = u_weighted * dt   # advection in x-direction
        deltay = v_weighted * dt   # advection in y-direction

    else: # stop calculation if drift vector is not available within distance 'a'

        deltax = 0.0; deltay = 0.0
        particle.terminate = True
        particle.termination_flag = 'No ice drift data'
             
    # check sea ice concentration ----

    if verbose: print('trajectory_util.advect: @04, check ice conc')
        
    if ice_concn[1].nodata == True:
        if ice_concn[0].nodata == False:
            ice_concn[1] = ice_concn[0] # use date_prev if no data for the day
            print('NO DATA WARNING: ice_concn[1] is defined by ice_concn[0]')
            #ice_concn[0].nodata = True
        elif ice_concn[2].nodata == False:
            ice_concn[1] = ice_concn[2] # use date_next if no data for the day
            print('NO DATA WARNING: ice_concn[1] is defined by ice_concn[2]')                
            #ice_concn[0].nodata = True
        else:
            print('WARNING: there is no ice concentration data for 3 consecutive days!')
            print('         Effect of ice concentration is not taken into account')

    # define ice conc. at the particle's location ----
        
    if ice_concn[1].nodata == False:
        ice_conc = ice_concn[1]
        conc = ice_conc.conc                  # unit [%] 
        xc = ice_conc.xc; yc = ice_conc.yc    # x and y coordinate system for concentration
        dist_x = np.full_like(xc, pos_x) - xc # 1D array of x-distance, [m]
        dist_y = np.full_like(yc, pos_y) - yc # 1D array of x-distance, [m]
        dist_conc = np.ndarray((xc.shape[0], yc.shape[0]))

        for (i, dx) in enumerate(dist_x):
            for (j, dy) in enumerate(dist_y):
                dist_conc[j, i] = np.sqrt(dx**2 + dy**2)
                    
        a1 = 12.5e3  # user grid size [m] of conc. as a measure of e-folding scale
        weight = np.exp(-2.0 * (dist_conc**2 / a1**2))

        # calculate distance-weighted conc at particle point ----

        mask = np.where(dist_conc > 2.0 * a1, np.nan, weight)
        sweight = np.nansum(mask)
        conc_w = np.nansum(conc * mask) / sweight if sweight > 0.0 else 0.0 # weighted conc [%]
                
        # stop calculation if ice conc. < 15% ----
        #
        # Since np.nansum gives zero for sum of np.nans, calculation stops as well 
        # when the particle encounters into land mass.
            
        if conc_w < 15.0:
            particle.terminate = True
            particle.termination_flag = 'ice conc. lower than 15%'

    else: # ice conc. data is not available for 3 consecutive days ---
        conc_w = np.nan
                
    # update particle location and ice properties ----

    if verbose: print('trajectory_util.advect: @05, updating particle properties')
        
    particle.pos_xys.append([pos_x + deltax, pos_y + deltay])
    particle.stamps.append(date2)
    particle.u.append(u_weighted)
    particle.v.append(v_weighted)
    particle.conc.append(conc_w)
    particle.div.append(div(ice_drift[1], wmask))
        
    ease = pyproj.CRS("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 " \
                      + "+a=6371228 +b=6371228 +units=m +no_defs")  
    wgs84 = pyproj.CRS("EPSG:4326")  # latlon with WGS84 datum
    lat, lon = pyproj.transform(ease, wgs84, pos_x + deltax, pos_y + deltay) 
    particle.lonlats.append([lon, lat]) 

    # check advection ----

    if verbose:
        print('date1 =', date1, ',    date2 =', date2, '-----------')
        print('        u, v        =', u_weighted, v_weighted)
        print('        dx, dy      =', deltax, deltay)
        print('        pos_xys[-1] =', particle.pos_xys[-1])
        print('        lonlats[-1] =', particle.lonlats[-1])
        print('        ice conc.   =', particle.conc[-1])
        print('        ice div     =', particle.div[-1])
        print('')

    return particle


def get_ERA5_2m_temp(stamp, pos_xy, radius):
    """
    function to get 2m temperature of ERA5 data in given month at specified position, 
    using Gaussian-type weighting function.

    NOTE: This function requires '../43_backward_history/ERA5-ease-coord.pkl' to avoid
          recalculation of EASE-grid for ERA5. Currently the valid range of this program
          is north of 40 deg.N. To change this, clat should be modifed and the could 
          should be executed again to remake 'ERA5-ease-coord.pkl' .
          

          ---> recalculation of ERA5-ease-coord.pkl is now running... (2020.06.14)


    by Hiroshi Sumata / 2020.06.14

    inputs
    ------
    stamp     : datetime.date object, giving year and month
    pos_xy    : [x, y] point defined on the EASE grid
    radius    : influence radius to calculate areal mean

    returns
    -------
    temp      : weighted 2m temperature

    """
    # Define parameters ----

    verbose = False
    
    if verbose: print('@0')
    
    fname = '../../../Reanalysis/ERA5_monthly_avg_1979to2019/ERA5.monmean.1979to2019.nc'

    # read netCDF file ----
    
    nc = netCDF4.Dataset(fname, 'r')
    lons = nc.variables['longitude'][:]
    lats = nc.variables['latitude'][:]
    nctime = nc.variables['time'][:]
    t_units = nc.variables['time'].units
    t_cal = nc.variables['time'].calendar

    # 2m temperature ----
    
    #t2m = nc.variables['t2m'][:]
    t2m_FillValue = nc.variables['t2m']._FillValue
    t2m_scale_factor = nc.variables['t2m'].scale_factor
    t2m_add_offset = nc.variables['t2m'].add_offset

    ncstamps = [datetime.datetime.strptime(str(netCDF4.num2date(time, \
                units = t_units, calendar = t_cal))[0:10], '%Y-%m-%d').date() \
                for time in nctime]

    # specify index of time stamp ----

    if verbose: print('@1')
    
    index = -1
    for (n, ncstamp) in enumerate(ncstamps):
        if ncstamp.year == stamp.year and ncstamp.month == stamp.month:
            index = n
            
    t2m = nc.variables['t2m'][index, 0, :, :] - 273.0  # 2D field of 2m temp. [deg.C]    
    #
    # NOTE: t2m is in [lats, lons] shape
    #
    #if verbose:
    #    print('lats.shape =', lats.shape)
    #    print('lons.shape =', lons.shape)
    #    print('t2m.shape  =', t2m.shape)
    
    for (i, lat) in enumerate(lats):
        if lat < 45.0:
            t2m[i, :] = np.nan

    
    ## check temperature field ---- [OK]
    #
    # It seems that add_offset and scale_factor are automatically applied when read
    # the data is read by netCDF4 lib.
    #
    #print('t2m_add_offset   =', t2m_add_offset)
    #print('t2m_scale_factor =', t2m_scale_factor)
    #
    #temp = t2m - 273.0 
    #
    #fig = plt.figure(figsize = (8, 6))
    #ax = fig.add_subplot(1, 1, 1)
    #mappable = ax.pcolormesh(temp)
    #cbar = fig.colorbar(mappable, ax = ax, shrink = 0.8)
    #plt.show()
    #
    
    # define EASE grid ----

    if verbose: print('@2')
    
    fname = '../43_backward_history/ERA5-ease-coord.pkl'
    
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            (x_array, y_array) = pickle.load(f)
            
    else:
        clat = 40.0  # x_array and y_array have valid value north of this latitude [deg]
        
        if verbose: print('Recalculating ERA5-ease-coord.pkl: x_array and y_array..')
        if verbose: print('   x_array and y_array have valid value north of ', 
                          clat, '[deg]')
        
        ease = pyproj.CRS("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 " \
                          + "+a=6371228 +b=6371228 +units=m +no_defs")  
        wgs84 = pyproj.CRS("EPSG:4326") # latlon with WGS84 datum

        x_array = np.zeros((lons.shape[0], lats.shape[0]))
        y_array = np.zeros((lons.shape[0], lats.shape[0]))

        for (i, lat) in enumerate(lats):
            if verbose: print('   calculating, lat =', lat)
            if lat < clat:  
                x_array[:, i] = np.nan
                y_array[:, i] = np.nan
            else:
                for (j, lon) in enumerate(lons):
                    xpos, ypos = pyproj.transform(wgs84, ease, lat, lon)
                    x_array[j, i] = xpos
                    y_array[j, i] = ypos
                    
        with open(fname, 'wb') as f:
            pickle.dump((x_array, y_array), f)
                    
    # calculate weighting mask in accordance with distance ----

    if verbose: print('@3')
    
    dist = np.zeros((lons.shape[0], lats.shape[0]))
    dist[:, :] = np.sqrt((x_array[:, :] - pos_xy[0])**2 + (y_array[:, :] - pos_xy[1])**2)

    ## check spatial pattern of distance array, 'dist' -----[OK]
    #
    #check_2d_polar_field(x_array, y_array, dist)

    # calculate weighting function ---
    
    weight = np.zeros((lons.shape[0], lats.shape[0]))
    weight[:, :] = np.exp(-1.0 * (dist[:, :] / (radius * 1.0e3))**2) # use [km] as unit

    ## check weighting function ----[OK]
    #
    #check_2d_polar_field(x_array, y_array, weight)
    #
    #if verbose:
    #    print('x_array.shape =', x_array.shape)
    #    print('y_array.shape =', y_array.shape)
    #    print('t2m.T.shape   =', t2m.T.shape)
    #    print('np.nanmax(t2m) =', np.nanmax(t2m))
    #    print('np.nanmin(t2m) =', np.nanmin(t2m))
    #    
    #    temp = t2m.T
    #    check_2d_polar_field(x_array, y_array, temp)

    # calculate temperature at the given point using Gaussian-type weighting function ---
    #
    # NOTE: x_array, y_array, weight are defined as (1440, 721), while t2m is defined
    #       as (721, 1440).

    temp = np.nansum(weight * t2m.T) / np.nansum(weight)

    return temp


def check_2d_polar_field(x_array, y_array, dist):
    """
    check spatial pattern of value of 2D field

    by Hiroshi Sumata / 2020.07.14
    """
    # Define the projection used to display the circle:

    lon = 0.0; lat = 76.0
    resolution = '50m'
    proj = ccrs.NorthPolarStereo(central_longitude = 0)
    pad_radius = compute_radius(proj, 6)
    
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1, projection = proj)
    
    ax.set_xlim([-pad_radius, pad_radius])
    ax.set_ylim([-pad_radius, pad_radius])
    ax.imshow(np.tile(np.array([[cfeature.COLORS['water'] * 255]], \
                               dtype=np.uint8), [2, 2, 1]), \
              origin='upper', transform=ccrs.PlateCarree(), \
              extent=[-180, 180, -180, 180])
    
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', \
    #                                            'land', resolution, \
    #                                            edgecolor='black', \
    #                                            facecolor=cfeature.COLORS['land']))
                    
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', \
    #                                            'land', resolution, \
    #                                            edgecolor='black', \
    #                                            facecolor=cfeature.COLORS['land']))
    
    mappable = ax.contourf(x_array, y_array, dist)
    #mappable = ax.pcolormesh(x_array, y_array, dist)    
    cbar = fig.colorbar(mappable, ax = ax, shrink = 0.8)

    plt.show()

def compute_radius(ortho, radius_degrees):
    lon = 0.0; lat = 76.0
    resolution = '50m'
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)


def ice_trajectory_restart(particle, tracking_days):
    """
    sea ice trajectory calculation in addition to existing trajectory 
    (forward or backward)

    by Hiroshi Sumata / 2020.07.17

    inputs
    ------
    particle     : uls_util.Particle object
    tracking_days: integer, number of tracking days. if tracking_days < 0, then backward 
                   trajectory will be calculated.

    outputs
    -------
    particle     : particle object

    """
    step = 1 if tracking_days >= 0 else -1
    dates = [particle.stamps[-1] + datetime.timedelta(days = n)
             for n in range(0, tracking_days, step)]
    
    for (n, date) in enumerate(dates):
        if n < len(dates) - 1:

            if particle.terminate == False:

                date_prev = date + datetime.timedelta(days = - 1)
                date_next = date + datetime.timedelta(days = + 1)
        
                # get daily ice drift field ----

                product = 'NSIDCv4p1'
                ice_drift = [trajectory_util.IceDrift(date_prev, product),
                             trajectory_util.IceDrift(date,      product),
                             trajectory_util.IceDrift(date_next, product)
                             ]
                
                # get daily ice concentraton field ----
        
                product = 'OSI-409-430'
                ice_concn = [trajectory_util.IceConcn(date_prev, product),
                             trajectory_util.IceConcn(date,      product),
                             trajectory_util.IceConcn(date_next, product)
                             ]

                # advect particles ----
        
                particle.advect(date, dates[n + 1], ice_drift, ice_concn)
                           
            else:
                print('calculation finished, @0')
                break

        elif n == len(dates) - 1:
            print('calculation finished, @1')

    return particle               
