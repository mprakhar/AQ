# __author__ = 'Prakhar'
# Created 20171111
# Last edit 20171111

#Purpose: (1) to generate INDIA complete subset file of MODIS mean

#Output expected:


#Terminology used:
import rasterio as rio
import numpy as np
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
from scipy import stats
from glob import glob
import pandas as pd
import rasterio as rio
import sys
import ntpath
import matplotlib.patches as mpatches

#Pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from spatialop.classRaster import Raster_file
from spatialop.classRaster import Image_arr
from spatialop import my_plot as mpt
from spatialop import infoFinder as info
from spatialop import shp_rstr_stat as srs


# initialize
gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path
input_zone_polygon = gb_path + r'/Data/Data_raw/Shapefiles/India_SHP/India_merged.shp'
AQpath = gb_path + r'/Data/Data_process/GlobalMean/'
AQout = gb_path + r'/Data/Data_process/Meanmonth_AQ_20171111/MODIS/'
Airpath = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/AirRGB/AirRGB20171202Imasu/Global/Annual/'
Airoutpath = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/AirRGB/AirRGB20171202Imasu/India/Annual/'


def subsetAQ(AQpath, outpath, prod):
    #function to subset AirRGB over India only
    for file in glob(os.path.join(AQpath,  '*' + prod +'*'+ '.tif')):
        head, tail = ntpath.split(file)
        print tail
        yearmon = tail[9:15]
        srs.rio_zone_mask(input_zone_polygon, file, outpath + 'clean'+prod+str(yearmon)+'.tif')

def subsetAirRGB(Airpath, Airoutpath, prod):
    #function to subset AirRGB over India only
    for file in glob(os.path.join(Airpath,  '*'+ '.tif')):
        head, tail = ntpath.split(file)
        print tail
        yearmon = tail[-8:-4]
        srs.rio_zone_mask(input_zone_polygon, file, Airoutpath + 'clean'+prod+str(yearmon)+'.tif')



subsetAQ(AQpath, AQout, prod = 'AOD')
subsetAQ(AQpath, AQout, prod = 'ANG')

subsetAirRGB(Airpath, Airoutpath, 'RGB')
