#__author__ = 'Prakhar MISRA'
# Created 3/26/2016
#Last edit 11/1/2017

# Purpose:
#          (2) Perform interannual claibration for DMSP sensors based on  Wu et al (Intercalibration of DMSP-OLS night-time light data by the invariant region method)
#          (3) Check the accuracy of calibration equations developed in (1) by comparing TNL at district level

#Dependencies:
#OLS_intercalibration_coeff_Wu.csv
# Location of output: E:\Acads\Research\AQM\Data process\CSVOut



import numpy as np
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
from scipy import stats
from glob import glob
import pandas as pd
import sys

#Pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from spatialop.classRaster import Raster_file
from spatialop.classRaster import Image_arr
from spatialop import my_plot as mpt
from spatialop import infoFinder as info
from spatialop import shp_rstr_stat as srs

#INPUT Info

gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

ver = 'v1'          #version number of output
dor = '20160325'    #date of run
YEARM = '2013'         # year for which two images are being compared or analyzed
START_VRS = 0 # data from which VRS is avaliable.. a constant value, hence all in caps.showuld be 201400 in case of monthly VRS
sat_aq = [('MODIS','ANG'),('MODIS','AOD'),('OMI', 'SO2'), ('OMI', 'NO2')] # Air quality satellites
file_path = {'MODIS': gb_path + r'/Data/Data_raw/MOD04L2/L4//',
             'OMI': r'G:\Data OMI MODIS\OMI\L3//',
             'VIIRS': gb_path + r'/Data/Data_raw/VIIRS Composite/75N060E//',
             'DMSP': gb_path + r'Data/Data_raw/DMSP OLS/Subset_downloaded_20161031//',
             'GIS': gb_path + r'/Data/Data_raw/GIS data//'}

input_zone_polygon = gb_path+'/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
satT = 'DMSP'       # options VIIRS, MODIS, OMI
prodT = 'OLS'        # options DNB, AOD, ANG, NO2, SO2, DNBRAD refers to radiance values obtained from DN values

# ImageT: target reolsution to be achieved, ImageS: source image to be resampled
raster_pathT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\\'
input_value_rasterT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\F182013.v4c_web.stable_lights.avg_vis.tif'
georef_raster = gb_path + r"Data/Data_process/Georef_img/DMSP_georef.tif" #reference image that has the georef info

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'








# find resampled image of DNB at OLS level for 2013 - DNBr
# for the DNBr find TNL at ditrict level
# for OLS convert it to radiance by the two two techniques and find their district level TNL nd compare it with (2)
#      *1)  DNB=0 if OLSrad = 0
#           DNB=0.3205397924*exp(0.0704915218*OLSrad)
#      *2)  DNB = OLSrad*0.1159872231 (OLSrad<262.53)
#           DNB = 4.5*exp(0.0612148962*OLSrad) (OLSrad>=262.53)

dor='20160815'
ver = 'v1'
ols = Raster_file()
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/DMSP OLS//'
ols.sat = 'DMSP'
ols.prod = 'OLS'
ols.sample = ols.path + 'F182010.v4/F182010.v4d_web.stable_lights.avg_vis.tif'
ols.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//DMSP_georef.tif'

input_zone_polygon_0 = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
input_zone_polygon_3 = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm3.shp'

ols.file = ols.path +  'F182013.v4/F182013.v4c_web.stable_lights.avg_vis.tif'





# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * *  Task II DMSP Interannual Calibration  *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Type I: Wu et al's region invariant method

# since DMSP  OLS havent beein sensor calibrated, I am using the method of "Intercalibration of DMSP-OLS night-time light data by the invariant region method" by Wu et al for
# calibration. the regression function is of the power function form; every image thus gets calibrate to 2012 image
# DNc +1 = a*(DNo + 1)^b
# Where,
# DNc is calibrated DN (NOTE: needs to be converted to radiance if any operation such as rsampling or averaging of pixels is to take place
# DNo is original DN
# a and b are coefficient for each year as mentioned below http://www.tandfonline.com/action/downloadTable?id=T0002&doi=10.1080%2F01431161.2013.820365&downloadType=CSV


def DMSPcalib_Wu(sensorcode, raster_path, input_zone_polygon):
    dfOLSc = pd.read_csv(csv_in_path + r'OLS_intercalibration_coeff_Wu.csv',
                         header=0)  # OLSc = OLS calibration. this file calibrates to 2012

    yearm = str(sensorcode[0])
    satcode = sensorcode[1]

    a = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['a'])
    b = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['b'])
    input_value_raster = raster_path +  satcode + yearm + '.v4b_web.stable_lights.avg_vis.lzw.tif'

    if (int(yearm) > 2010):
        a = 0.8114
        b = 1.0849
        input_value_raster = raster_path + satcode + yearm + '.v4\\' + satcode + yearm + '.v4c_web.stable_lights.avg_vis.tif'
    elif (int(yearm) >= 2008):
        input_value_raster = raster_path + satcode + yearm + '.v4\\' + satcode + yearm + '.v4c_web.stable_lights.avg_vis.tif'

    imgarray, datamask = srs.zone_mask(input_zone_polygon, input_value_raster)

    # mpt.histeq(imgarray)

    # calibration step
    imgarray_c = a * np.power(imgarray + 1, b)

    # georef step
    outpath = gb_path + '/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
    srs.arr_to_raster(imgarray_c, georef_raster, outpath +satcode + str(yearm) + ".tif")


# function end

# -----------

# Type II: Zhang, Pandey Ridge regression method


# since DMSP  OLS havent beein sensor calibrated, I am using the method of "A robust method to generate a consistent time series from dmsp ols nighttime light data" by qingling zhang, bhartedu pandey
# calibration. the regression function is of the power function form; every image thus gets calibrate to 2000 image
# DNc = aDNo + bDNo**2 + c
# Where,
# DNc is calibrated DN (NOTE: needs to be converted to radiance if any operation such as rsampling or averaging of pixels is to take place
# DNo is original DN
# a and b, c are coefficient for each year as mentioned in the paper "A robust method to generate a consistent time series from dmsp ols nighttime light data"


def DMSPcalib_Zhang(sensorcode, raster_path, input_zone_polygon):
    dfOLSc = pd.read_csv(csv_in_path + 'OLS_intercalibration_coeff_Pandey.csv',
                         header=0)  # OLSc = OLS calibration. this file calibrates to 2012

    yearm = str(sensorcode[0])
    satcode = sensorcode[1]

    # input_value_raster = raster_path + satcode+yearm+'.v4b.avg_lights_x_pct\\'+satcode+yearm+'.v4b.avg_lights_x_pct.tif'
    if (int(yearm) == 2013):
        a = 0.355542
        b = 0.007962
        c = 3.866698
        # input_value_raster = raster_path + satcode+yearm+'.v4\\'+satcode+yearm+'.v4c_web.stable_lights.avg_vis.tif'
    else:
        a = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['a'])
        b = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['b'])
        c = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['c'])

    input_value_raster = glob(os.path.join(raster_path, satcode + yearm + '*' + '.tif'))[0]
    imgarray, datamask = srs.zone_mask(input_zone_polygon, input_value_raster)
    # mpt.histeq(imgarray)

    # calibration step
    imgarray_c = a * imgarray + b * np.power(imgarray, 2) + c

    # georef step
    outpath = gb_path + '/Data/Data_process/DMSPInterannualCalibrated_20160512/Zhang/'
    srs.arr_to_raster(imgarray_c, georef_raster, outpath + satcode + str(yearm) + ".tif")


# function end

# ----- #    - - - - - - - -calibrate here - - - - - - - -  -#--------#

raster_path = file_path['DMSP']  # T - how do you want the resolution of S to end up as
# input_value_raster = raster_path + 'F162007.v4b.avg_lights_x_pct\F162007.v4b.avg_lights_x_pct.tif'
# input_value_raster = raster_path + 'F182012.v4\F182012.v4c_web.stable_lights.avg_vis.tif'
DMSP_sensor = [(1999, 'F14'), (2000, 'F15'), (2001,  'F15'), (2002, 'F15'), (2003, 'F16'), (2004, 'F16'), (2005, 'F16'),
               (2006, 'F16'), (2007, 'F16'),
               (2008, 'F16'), (2009, 'F16'), (2010, 'F18'), (2011, 'F18'), (2012, 'F18')]

# run calib function
for sensorcode in DMSP_sensor:
    # trying method 1
    DMSPcalib_Wu(sensorcode, raster_path, input_zone_polygon)









