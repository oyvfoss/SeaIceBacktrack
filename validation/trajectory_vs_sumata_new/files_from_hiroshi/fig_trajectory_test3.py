#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

import trajectory_util

lon = 0.0; lat = 76.0

def compute_radius(ortho, radius_degrees):
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)

def main():
    """
    Test program to plot backward sea ice trajectory

    by Hiroshi Sumata / 2020.06.15
    """

    # Define the projection used to display the circle:
    proj = ccrs.NorthPolarStereo(central_longitude = 0)
    pad_radius = compute_radius(proj, 6)

    # define image properties
    width = 800
    height = 800
    dpi = 96
    resolution = '50m'

    # create figure
    
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_xlim([-pad_radius, pad_radius])
    ax.set_ylim([-pad_radius, pad_radius])
    ax.imshow(np.tile(np.array([[cfeature.COLORS['water'] * 255]], \
                               dtype=np.uint8), [2, 2, 1]), \
              origin='upper', transform=ccrs.PlateCarree(), \
              extent=[-180, 180, -180, 180])
    
    ax.add_feature(cfeature.NaturalEarthFeature('physical', \
                                                'land', resolution, \
                                                edgecolor='black', \
                                                facecolor=cfeature.COLORS['land']))


    #fname = 'track_2015-08-01_days=-700_floes=1.pkl'
    #fname = './output/FS.2018.12.backward_traj.100_days.n_56.pkl'
    #fname = './output/FS.2017.12.backward_traj.100_days.n_56.pkl'
    #fname = './output/FS.2018.09.backward_traj.731_days.n_56.pkl'
    #fname = './output/FS.2014.09.backward_traj.731_days.n_56.pkl'
    #fname = './output/FS.2011.09.backward_traj.731_days.n_56.pkl'
    #fname = './output/FS.2005.09.backward_traj.731_days.n_56.pkl'
    #fname = './output.2000-2018.backup/FS.2016.09.backward_traj.731_days.n_8.pkl'
    fname = 'FS.2018.01.backward_traj.731_days.n_8.pkl'    
    
    
    with open(fname, 'rb') as f:
        ice_floes = pickle.load(f)

        stamp = ice_floes[0].stamps[0]
        cyear = str(stamp.year)
        c_mon = str(stamp.month) if stamp.month > 9 else '0' + str(stamp.month)
        c_day = str(stamp.day) if stamp.day > 9 else '0' + str(stamp.day)
        cstamp = cyear + '-' + c_mon + '-' + c_day
        ax.set_title('Backward ice trajectory, date:' + cstamp \
                     + ', back to ' + str((stamp - ice_floes[0].stamps[-1]).days) \
                     + ' days')

        for ice_floe in ice_floes:
        
            lonlats = ice_floe.lonlats
            tstamps = ice_floe.stamps

            day_bak = [(tstamps[0] - stamp).days for stamp in tstamps]
            print('day_bak =', day_bak)
            
            lons = []; lats = []
            for lonlat in lonlats:
                lons.append(lonlat[0])
                lats.append(lonlat[1])

            if len(day_bak) > 3:    
                mappable = ax.scatter(lons, lats, c = day_bak,
                                      transform = ccrs.Geodetic(),
                                      s = 3, marker = 'o', cmap = plt.cm.rainbow,
                                      vmin = 0, vmax = 730)

            #ax.plot(lons, lats, transform = ccrs.Geodetic(), color = 'red')


    cbar = fig.colorbar(mappable, ax = ax, shrink = 0.8)
    cbar.set_label('Days leading up to Fram Strait arrival [days]')
    
    plt.show()


if __name__ == '__main__':
    main()

