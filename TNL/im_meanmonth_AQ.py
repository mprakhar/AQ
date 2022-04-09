# __author__ = 'Prakhar'
# Created 8/24/2016
# Last edit 8/28/2016

#Purpose: (1) to generate monthly mean images from MODIS and OMI daily data by removing 2%-98% outliers nad taking mean
#         (2) to generate annual urban images from monthly images
#         (3) to generate a MODIS image for whole of india based on pixel level calibrations developed

#Output expected:


#Terminology used:


import numpy as np
from numpy import *
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import os.path
import cv2
import gdal
from PIL import Image
import pandas as pd
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from glob import glob
import seaborn as sns

#Pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
from spatialop.classRaster import Image_arr
from spatialop  import coord_translate as ct


# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

aod = Raster_file()
aod.path = gb_path + '/Data/Data_raw/MOD04L2/L3//'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = gb_path + '/Data/Data_raw/MOD04L2/L3//'+ 'MOD04L2.A2015308.AOD.Global'
aod.georef = gb_path + '/Data/Data_process/Georef_img//MODIS_georef.tif'

ang = Raster_file()
ang.path = gb_path + '/Data/Data_raw/MOD04L2/L3//'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path + '/Data/Data_raw/MOD04L2/L3//' + 'MOD04L2.A2015308.ANG.Global'
ang.georef = gb_path + '/Data/Data_process/Georef_img//MODIS_georef.tif'

no2 = Raster_file()
no2.path = gb_path + '/Data/Data_raw/OMI/L2G//'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = gb_path + '/Data/Data_raw/OMI/L2G//' + 'OMI.NO2.20050203.Global'
no2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

so2 = Raster_file()
so2.path = gb_path + '/Data/Data_raw/OMI/L2G//'
so2.sat = 'OMI'
so2.prod = 'SO2'
so2.sample = gb_path + '/Data/Data_raw/OMI/L2G//' + 'OMI.SO2.20050203.Global'
so2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

input_zone_polygon = gb_path + '/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'

ols = Raster_file()
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
ols.sat = 'DMSP'
ols.prod = 'OLS'
ols.sample = ols.path + r'F162008.tif'
ols.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//DMSP_georef.tif'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'


# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task I  Clean mean monthly images     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

for prodT in [so2, no2]:

    start_date= date(2004, 10, 1)
    end_date= date(2015, 12, 31)
    a =0
    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        mon1 = date_m       # start of month
        mon2 = date_m + relativedelta(months=1) - relativedelta(days=1)     # end of month
        ls = []
        print date_m
        for date_d in rrule.rrule(rrule.DAILY,dtstart=mon1, until=mon2 ):
            for prodT.file in glob(os.path.join(prodT.path, '*.'+prodT.prod+'*'+'.Global')) :
                date_y_d = map(int, prodT.yeardayfinder())  # date as a list with integer year and daycount as return
                # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                if (date_y_d == [date_d.year, date_d.timetuple().tm_yday]) & os.path.isfile(prodT.file):  # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                    # print 'file found'
                    try:
                        prod_arr, mask = prodT.zone_mask(input_zone_polygon)
                        ls.append(prod_arr)
                    except AttributeError:
                        print ' some error in ',  date_d
        ls_arr = np.array(ls)
        ls_98 = np.percentile(ls_arr, 98.0)
        ls_02 = np.percentile(ls_arr, 02.0)
        ls_arr[ls_arr>=ls_98] = np.nan
        ls_arr[ls_arr <= ls_02] = np.nan
        ls_arr[ls_arr == 0] = np.nan
        ls_mean = np.nanmean(ls_arr, axis=0)
        ls_obj =  Image_arr(ls_mean)
        ls_obj.georef = prodT.georef
        ls_obj.arr_to_raster(img_save_path + '/Meanmonth_AQ/'+ prodT.sat+ '/clean' + prodT.prod + str(date_m.year) + str('%02d'%date_m.month) + '.tif')

print 'done'




# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task II  Clean mean annual images     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

for prodT in [so2, no2]:
    prodT.path = '/home/prakhar/Research/AQM_research/Data/Data_process/Meanmonth_AQ/' + prodT.sat + r'//'
    start_date= date(2014, 1, 1)   # 2004, 10, 1
    end_date= date(2014, 12, 31)      #2016, 6, 1
    for date_m in rrule.rrule(rrule.YEARLY, dtstart=start_date, until=end_date):
        mon1 = date_m       # start of month
        mon2 = date_m + relativedelta(months=12) - relativedelta(months=1)     # end of month
        ls = []
        print date_m
        for date_d in rrule.rrule(rrule.MONTHLY,dtstart=mon1, until=mon2 ):
            for prodT.file in glob(os.path.join(prodT.path, '*'+prodT.prod+'*'+'.tif')) :
                date_y_d = int(prodT.file[-10:-4])  # date as a list with integer year and daycount as return
                # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                if (date_y_d == date_d.year*100+ date_d.timetuple().tm_mon) & os.path.isfile(prodT.file):  # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                    print 'file found'
                    try:
                        prod_arr, mask = prodT.zone_mask(input_zone_polygon)
                        ls.append(prod_arr)
                    except AttributeError:
                        print ' some error in ',  date_d
        ls_arr = np.array(ls)
        ls_arr[ls_arr == 0] = np.nan
        ls_mean = np.nanmean(ls_arr, axis=0)
        ls_obj = Image_arr(ls_mean)
        ls_obj.georef = prodT.georef
        ls_obj.arr_to_raster(img_save_path + '/Meanmonth_AQ/' + 'clean' + prodT.prod + str(date_m.year) + '.tif')




# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task III Finding annual urban AQ image     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# (a) FBecasue all of them already intracalibrated, no need to do any reverse-calibration


THRESH = 20.0     # it hsoukd be 29 but we make it 20 because if you check and recall, we got threshold of 29 by keeping the minval of 20. so this is the minval.
nl_path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
dilation = 0
for year in range(2014,2015): #2001, 2015
    for ols.file in glob(os.path.join(ols.path, '*'+str(year)+ '*'+'.tif')) :
        # imgarrayS, datamaskS = ols.zone_mask(input_zone_polygon)
        # print year, mth.RosinThreshold(imgarrayS, 20)

        for prodT in [ang, aod, so2, no2]:
            prodT.path = '/home/prakhar/Research/AQM_research/Data/Data_process/Meanannual_' + prodT.sat + r'//'
            if glob(os.path.join(prodT.path, 'clean/'+'*'+prodT.prod + str(year)+ '*'+'.tif')):

                prodT.file = glob(os.path.join(prodT.path, 'clean/'+ '*'+prodT.prod + str(year)+ '*'+'.tif'))[0]
                imgarrayT, datamaskT = prodT.zone_mask(input_zone_polygon)
                imgarrayS, datamaskS = ols.zone_mask(input_zone_polygon)
                yfactor = (float((np.shape(imgarrayS))[0])) / (float((np.shape(imgarrayT))[0]))
                xfactor = (float((np.shape(imgarrayS))[1])) / (float((np.shape(imgarrayT))[1]))
                imgarrayS = imgarrayS ** (1.5)
                imgarrayR = block_reduce(imgarrayS.data, block_size=(int(ceil(yfactor)), int(ceil(xfactor))), func=np.mean)  # resampled image R ----- Imp Step

                # binary mask preparation
                imgarrayR[imgarrayR < (THRESH) ** 1.5] = 0.0
                imgarrayR[imgarrayR>=(THRESH)**1.5] = 1.0
                if dilation ==1:
                    kernel = np.ones((2,2), np.uint8)
                    imgarrayR = cv2.dilate(imgarrayR, kernel, iterations=1)

                #urban array
                imgarrayU = imgarrayT*imgarrayR     #urban array

                U_obj = Image_arr(imgarrayU)
                U_obj.georef = prodT.georef
                U_obj.arr_to_raster('urban_' + prodT.prod + str(year) + '.tif')
                print ('urban_' + prodT.prod + str(year) + '.tif')


# ----------------------------------------------------------------------------------------------------------------
# preparing clean urban image for AirRGB
def clean_urbanAIRGB():
    THRESH = 20.0  # it hsoukd be 29 but we make it 20 because if you check and recall, we got threshold of 29 by keeping the minval of 20. so this is the minval.
    nl_path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
    dilation = 0
    for year in range(2001, 2015):  # 2001, 2015
        for ols.file in glob(os.path.join(ols.path, '*' + str(year) + '*' + '.tif')):
            # imgarrayS, datamaskS = ols.zone_mask(input_zone_polygon)
            # print year, mth.RosinThreshold(imgarrayS, 20)


                Airpath =  '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/AirRGB/AirRGB20171202Imasu/India/Annual/'
                if glob(os.path.join(Airpath, 'clean' + '*' + '.tif')):

                    filepath = glob(os.path.join(Airpath, 'clean' + '*' + str(year) + '*' + '.tif'))[0]

                    imgarrayT, datamaskT = srs.zone_mask(input_zone_polygon, filepath)
                    imgarrayS, datamaskS = ols.zone_mask(input_zone_polygon)
                    yfactor = (float((np.shape(imgarrayS))[0])) / (float((np.shape(imgarrayT))[0]))
                    xfactor = (float((np.shape(imgarrayS))[1])) / (float((np.shape(imgarrayT))[1]))
                    imgarrayS = imgarrayS ** (1.5)
                    imgarrayR = block_reduce(imgarrayS.data, block_size=(int(ceil(yfactor)), int(ceil(xfactor))),
                                             func=np.mean)  # resampled image R ----- Imp Step

                    # binary mask preparation
                    imgarrayR[imgarrayR < (THRESH) ** 1.5] = 0.0
                    imgarrayR[imgarrayR >= (THRESH) ** 1.5] = 1.0
                    if dilation == 1:
                        kernel = np.ones((2, 2), np.uint8)
                        imgarrayR = cv2.dilate(imgarrayR, kernel, iterations=1)

                    # urban array
                    imgarrayU = imgarrayT * imgarrayR  # urban array

                    srs.rio_arr_to_raster(imgarrayU, aod.georef, 'urban_' + 'R' + str(year) + '.tif')
                    print ('urban_' + str(year) + '.tif')










