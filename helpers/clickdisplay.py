# __author__ = 'Prakhar'
# Created 10/5/2016
# Last edit 10/5/2016

#Purpose: (a) To create an interface for taking an claickable input from user
#         (b) To use the coordinates from click and plot AQ trend for last 10 years

#Output expected:


#Terminology used:
#       df_gdpcor - gdp and aq correlation series
#       st_name - state name




import numpy as np
from numpy import *
import csv
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import zipfile
import os.path

import gdal
from PIL import Image
import pandas as pd
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from glob import glob
import seaborn as sns
from osgeo import gdal, gdalnumeric, ogr, osr
import scipy.stats as stats

#Pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from  spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct


# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

aod = Raster_file()
aod.path = gb_path + 'Data/Data_process/Meanannual_MODIS/Clean/'
# aod.path = gb_path + 'Data/Data_process/Meanmonth_AQ/MODIS/'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample =  gb_path + '/Data/Data_raw/MOD04L2/L3/MOD04L2.A2015308.AOD.Global'
aod.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//MODIS_georef.tif'

ang = Raster_file()
ang.path = gb_path + 'Data/Data_process/Meanannual_MODIS/Clean/'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path + '/Data/Data_raw/MOD04L2/L3/MOD04L2.A2015308.AOD.Global'
ang.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//MODIS_georef.tif'

so2 = Raster_file()
so2.path = gb_path + r'/Data/Data_process/Meanannual_OMI/Clean/'
so2.sat = 'OMI'
so2.prod = 'SO2'
so2.sample = gb_path + '/Data/Data_raw/OMI/L2G/OMI.NO2.20041003.Global'
so2.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//OMI_georef.tif'

no2 = Raster_file()
no2.path = gb_path + r'/Data/Data_process/Meanannual_OMI/Clean/'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = gb_path + '/Data/Data_raw/OMI/L2G/OMI.NO2.20041003.Global'
no2.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//OMI_georef.tif'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'


# * * * *  * * # * * * *  * * # * * * *  * *# # * *  Display map on Basemap & Click to get coordinates  # * * * *  * * # * * * *  * * # * * * *  * *# # *

# (a) Display map
def display_map():
    global fig
    fig = plt.figure()
    map = Basemap(projection='cyl',lat_0=8, lon_0=70,
    resolution = 'l', area_thresh = 1000.0,
        llcrnrlon=60, llcrnrlat=5,
        urcrnrlon=100, urcrnrlat=40
                  ) # using Cylindrical Equidistant projection.; map coordinates are shown in lat and long

    # Fill the globe with a blue color
    map.drawmapboundary(fill_color='aqua')
    # Fill the continents with the land color
    # map.fillcontinents(color='coral',lake_color='aqua')
    map.drawcountries()
    # map.drawcoastlines()
    map.bluemarble()

    # city locs
    lon = [77.21,  85.13, 78.17,  72.58 , 80.92,  78.39,  80.33, 74.87, 75.84, 81.84, 78.01,73.02,  72.82,    88.36,  80.27]
    lat = [28.67, 25.62, 26.23, 23.03, 26.85,27.15, 26.47, 31.64, 30.91, 25.45, 27.19, 26.29, 18.96, 22.57, 13.09]
    labels = ['Delhi',  'Patna', 'Gwalior', 'Ahmadabad', 'Lakhnau',  'Firozabad', 'Kanpur', 'Amritsar', 'Ludhiana', 'Allahabad', 'Agra', 'Jodhpur', 'Mumbai', 'Calcutta', 'Madras']
    x,y = map(lon, lat)
    map.plot(x,y,'ro', markersize=5)
    for label, xpt, ypt in zip(labels, x, y):
        plt.text(xpt, ypt, label)



    plt.show()
    return fig

#(b) Get coordinates
# coords = []

def onclick(event):

    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)

    global coords
    coords.append((ix, iy))

    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)

    aq_plotter(coords)

    return coords
# display_map()
# cid = fig.canvas.mpl_connect('button_press_event', onclick)



# * * * *  * * # * * * *  * * # * * * *  * *# # * * Using coordinate and plotting  # * * * *  * * # * * * *  * * # * * * *  * *# # *

def aq_plotter(coords):

    aq_aod = []
    aq_ang = []
    aq_so2 = []
    aq_no2 = []
    raw_input("press key")

    city_list = pd.read_csv(gb_path + '/Data/Data_process/GIS/City_list.csv', header=0)
    city_list['rel_dist'] = (city_list['Lat']-coords[-1][1])**2+ (city_list['Long']-coords[-1][0])**2
    near_city = city_list['City'][city_list['rel_dist'].argmin()]
    near_dist = city_list['rel_dist'][city_list['rel_dist'].argmin()]*110
    prodT = aod
    pixloc = ct.latLonToPixel(prodT.georef, [[float(coords[-1][1]), float(coords[-1][0])]])  # input>>lat, long   output >>pix_y, pix_x
    for prodT.file in glob(os.path.join(prodT.path, '*'+prodT.prod+'*')) :
        year = int(prodT.file[-8:-4])
        aq_val =  prodT.raster_as_array()[pixloc[0][1], pixloc[0][0]]
        aq_aod.append([year, aq_val ])
    df_aq_aodtrend = pd.DataFrame(aq_aod, columns=['year', prodT.prod])  # strores regression coefficients for each city

    prodT = ang
    pixloc = ct.latLonToPixel(prodT.georef, [[float(coords[-1][1]), float(coords[-1][0])]])  # input>>lat, long   output >>pix_y, pix_x
    for prodT.file in glob(os.path.join(prodT.path, '*'+prodT.prod+'*')) :
        year = int(prodT.file[-8:-4])
        aq_val =  prodT.raster_as_array()[pixloc[0][1], pixloc[0][0]]
        aq_ang.append([year, aq_val ])
    df_aq_angtrend = pd.DataFrame(aq_ang, columns=['year', prodT.prod])  # strores regression coefficients for each city

    prodT = so2
    pixloc = ct.latLonToPixel(prodT.georef, [[float(coords[-1][1]), float(coords[-1][0])]])  # input>>lat, long   output >>pix_y, pix_x
    for prodT.file in glob(os.path.join(prodT.path, '*'+prodT.prod+'*')) :
        year = int(prodT.file[-8:-4])
        aq_val =  prodT.raster_as_array()[pixloc[0][1], pixloc[0][0]]
        aq_so2.append([year, aq_val ])
    df_aq_so2trend = pd.DataFrame(aq_so2, columns=['year', prodT.prod])  # strores regression coefficients for each city

    prodT = no2
    pixloc = ct.latLonToPixel(prodT.georef, [[float(coords[-1][1]), float(coords[-1][0])]])  # input>>lat, long   output >>pix_y, pix_x
    for prodT.file in glob(os.path.join(prodT.path, '*'+prodT.prod+'*')) :
        year = int(prodT.file[-8:-4])
        aq_val =  prodT.raster_as_array()[pixloc[0][1], pixloc[0][0]]
        aq_no2.append([year, aq_val ])
    df_aq_no2trend = pd.DataFrame(aq_no2, columns=['year', prodT.prod])  # strores regression coefficients for each city


    fig, ax = plt.subplots()
    ax.plot(df_aq_aodtrend['year'], df_aq_aodtrend['AOD'], 'ko--', label= 'AODtrend', alpha=.8)
    ax.plot(df_aq_angtrend['year'], df_aq_angtrend['ANG']/3, 'o--', label='ANGtrend', alpha=.8)
    ax.plot(df_aq_so2trend['year'], df_aq_so2trend['SO2'], '^--', label='SO2rend', alpha=.8)
    ax.plot(df_aq_no2trend['year'], df_aq_no2trend['NO2']/10, '>--', label='NO2trend', alpha=.8)
    ax.legend(loc=4)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('AQ', fontsize=16)
    plt.title('AQ Trends (Nearest town is ' + near_city +'-'+ str('%.2f' % near_dist) + 'km)', fontsize=20)
    plt.tight_layout()



# * * * *  * * # * * * *  * * # * * * *  * *# # * *  RUN this   # * * * *  * * # * * * *  * * # * * * *  * *# # *
display_map()
coords=[]
global df_aq_trend
cid = fig.canvas.mpl_connect('button_press_event', onclick)









