#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 10/12/2017
# Last edit 9/01/2017

# Purpose: To plot trend of AQ and NL . Doing this activity in a new way.

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
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

#Pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from spatialop.classRaster import Raster_file
from spatialop.classRaster import Image_arr
from spatialop import my_plot as mpt
from spatialop import my_math as mth
from spatialop import infoFinder as info
from spatialop import shp_rstr_stat as srs

#basic initialization
gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path
input_zone_polygon = gb_path + r'/Data/Data_raw/Shapefiles/India_SHP/India_merged.shp'
AirRGBpath = gb_path + r'/Data/Data_process/AirRGB/AirRGB20170910/Month/'
sampleAir = r'/Data/Data_process/AirRGB/AirRGB20170910/MonthRGB201110.tif'
DNBpath = gb_path + '/Data/Data_raw/VIIRS Composite/75N060E/'
LULCpath = gb_path+ r'/Data/Data_process/GIS/'
#path to saveDNB in natove resolution i=subset over Indai
DNBoutIndia = gb_path + r'/Data/Data_process/VIIRS/DNB_India/'
#path to saveDNB in MODIS resolution subset over Indai
DNBout10India = gb_path + r'/Data/Data_process/VIIRS/DNB10km_India/'
#path to save subsetAIrRGB
AirRGBout = gb_path + r'/Data/Data_process/AirRGB/India/'
#NLbuffer
NLbufferpath = gb_path +'/Data/Data_process/India_urban_buffer/Urbanbuffers.tif'


#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'




# ----------------------------- Step 0 basic subsetting funcitons -------------------------------

def subsetAQ(AQpath, outpath):
    #function to subset AirRGB over India only
    for file in glob(os.path.join(AQpath,  '*' + '.tif')):
        head, tail = ntpath.split(file)
        print tail
        yearmon = tail[8:14]
        srs.rio_zone_mask(input_zone_polygon, file, outpath + 'MonthRGB'+str(yearmon)+'.tif')

def subsetNL(NLpath, outpath):
    #function to subset AirRGB over India only
    for file in glob(os.path.join(NLpath,  '*' + '.tif')):
        head, tail = ntpath.split(file)
        print tail
        yearmon = tail[10:16]
        try:
            srs.rio_zone_mask(input_zone_polygon, file, outpath + 'DNB'+str(yearmon)+'.tif')
        except:
            print 'problem with ', str(yearmon)

def resample(arr, reseampfactorx, reseampfactory, type):
    # funtion to resample nightlight iages
    func_dict={'mean': np.mean,
               'sum':np.sum}
    return block_reduce(arr, block_size=(reseampfactory, reseampfactorx), func = func_dict[type])

def resample_LULC(LULCpath, targetrespath, thresh, GIS_C):  # 13 is the class number for urban):
    # prepare the file with urban pixels only. resolution 10km.
    # function to resample an array to bigger array based on the given threshold and mathcing pixel value
    srcarr = rio.open(LULCpath).read(1)
    dstarr = rio.open(targetrespath).read(1)

    yfactor = (float((np.shape(srcarr))[0])) / (float((np.shape(dstarr))[0]))
    xfactor = (float((np.shape(srcarr))[1])) / (float((np.shape(dstarr))[1]))

    # remove all other classes
    srcarr[srcarr!=GIS_C] = 0
    srcarr[srcarr == GIS_C] = 1

    #resmaple
    resarr = resample(srcarr, int(xfactor), int(yfactor), type = 'mean') # resampled image R ----- Imp Step

    #consider a pixel as urban if mean >= thresh
    resarr = np.where(resarr>=thresh, 1, 0)

    #return the resampled GIS class
    return resarr

def resample_NL(NLpath, targetrespath, type = 'sum'):
    #funciton to resample NL
    srcarr = rio.open(NLpath).read(1)
    dstarr = rio.open(targetrespath).read(1)

    #make sure the resolution align by resample the NL to MODIS
    yfactor = (float((np.shape(srcarr))[0])) / (float((np.shape(dstarr))[0]))
    xfactor = (float((np.shape(srcarr))[1])) / (float((np.shape(dstarr))[1]))

    #resampled NLarr
    resNLarr = resample(srcarr, int(xfactor), int(yfactor), type=type)

    return resNLarr



# ------------------------------------  analyzing NL and AQ ---------------------------------------------

def scatterAQNL(year,Rarr, NLarr, LULCarr, NLbuffer):
    #scatter all NL and AQ calues to check if there is a kuznet curve or anything

    #plt.figure(figsize=(8, 6))
    plt.scatter(NLarr, Rarr, c='blue', marker='o', label='all pixel', alpha = 0.2)
    plt.scatter( NLarr[NLbuffer == 1],Rarr[NLbuffer == 1], c='green', marker='o', label='urban buffer',alpha = 0.5 )
    plt.scatter( NLarr[LULCarr==1],Rarr[LULCarr==1], c='red', marker='^', label='urban')

    plt.title('AirRGB R & NL ', fontsize=20)

    plt.xlabel('NL', fontsize=18)
    plt.ylabel('AirRGB R', fontsize=18)
    plt.legend(fontsize=15)
    plt.show()
    plt.savefig(plt_save_path + 'AQNLClassurban_' + str(year) + '.png')



def trendAQNL(df_urbanclass):
    #function to plot tend of different classes AQNL
    x = pd.to_datetime(df_urbanclass.yearmon, format = '%Y%m')

    plt.figure(figsize=(10,4))
    plt.plot(x, df_urbanclass.HLHP, 'ro-' , label='HLHP')
    plt.plot(x, df_urbanclass.LLHP, 'b>-' ,label='LLHP')
    plt.plot(x, df_urbanclass.HLLP, 'g^-', label='HLLP')
    plt.plot(x, df_urbanclass.LLLP, 'k<-' , label='LLLP')

    plt.title('AirRGB R & NL trend', fontsize=20)

    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Urban pixel count', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=15)
    plt.show()
    plt.tight_layout()

def get_AQNL(year, Rarr, NLarr, thresh, figure = True):
    # get all the thresholds
    Rthresh = thresh[0]
    Bthresh = thresh[1]
    NLthresh = thresh[2]  # make a blank array for HLLP code
    AQNL = np.zeros(Rarr.shape)

    # make binary mask based on threshold values
    bin_AQarrlt = np.ma.masked_less_equal(Rarr, Rthresh).mask
    bin_NLarrlt = np.ma.masked_less_equal(NLarr, NLthresh).mask

    # classify
    AQNL[np.where(np.logical_and(bin_AQarrlt, bin_NLarrlt))] = 1  # LLLP
    AQNL[np.where(np.logical_and(np.logical_not(bin_AQarrlt), bin_NLarrlt))] = 2  # LLHP
    AQNL[np.where(np.logical_and(np.logical_not(bin_AQarrlt), np.logical_not(bin_NLarrlt)))] = 3  # HLHP
    AQNL[np.where(np.logical_and(bin_AQarrlt, np.logical_not(bin_NLarrlt)))] = 4  # HLLP

    #plot if required
    if figure:
        nv_patch = mpatches.Patch(color='navy', label='LLLP')
        aq_patch = mpatches.Patch(color='aqua', label='HLHP')
        yl_patch = mpatches.Patch(color='red', label='HLLP')
        mr_patch = mpatches.Patch(color='yellow', label='HLLP')
        plt.legend(handles=[nv_patch, aq_patch, yl_patch, mr_patch])
        plt.savefig(plt_save_path + 'AQNLClass_' + str(year) + '.png')
        plt.imshow(AQNL)

    return AQNL


def get_summary(year, resNLfile, Airfile, resLULCpath, NLbufferpath, thresh, figure = True):
    #funciton to prepare dataframe that generates annual scatter plot of AQ and NL with different markers for urban and non-urban areas
    # the df contains

    #load the relevant arrays
    #NIghtligh arra
    NLarr = rio.open(resNLfile).read(1)

    # R component of AirRGB
    Rarr = rio.open(Airfile).read(1)

    # B component of AirRGB
    Barr = rio.open(Airfile).read(3)

    #open the nightloght buffer
    # multiply by LCbuffer to get rid of city outskirts where pollution is high only becasue of chemicaltrasnport
    NLbuffer = rio.open(NLbufferpath).read(1)

    #open resampled LULC arr
    LULCarr = rio.open(resLULCpath).read(1)

    if figure:
        #scatter plot
        scatterAQNL(year, Rarr, NLarr, LULCarr, NLbuffer)

    #get classfied AQNLarr
    AQNL = get_AQNL(year, Rarr, NLarr, thresh, figure)

    #countng the number of each kind
    AQNLu = AQNL*LULCarr
    AQNLsummary = [int(year), np.size(np.where(AQNLu==1)), np.size(np.where(AQNLu==2)), np.size(np.where(AQNLu==3)), np.size(np.where(AQNLu==4))]

    return [AQNL, AQNLsummary]



def get_df(resNLpath, Airpath, resLULCpath, NLbufferpath, thresh):
    # function to compile each time step result from get_summary in to a df
    ls = []

    #datees between which to consider the dataframe
    start_date = datetime(2013,1,1)
    end_date = datetime(2017,1,1)

    for date_d in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

        #get the yearmonth
        yearmon =  (str(date_d.year) + str('%02d' % date_d.month))

        # preapre filenames
        resNLfile = resNLpath +  'DNB10'+str(yearmon) +'.tif'
        Airfile = Airpath + 'MonthRGB'+str(yearmon)+'.tif'

        #run if both of them exist
        if (os.path.isfile(Airfile)&os.path.isfile(resNLfile)):
            print yearmon
            ls.append(get_summary(yearmon, resNLfile, Airfile, resLULCpath, NLbufferpath, thresh, figure = False)[1])

    # make historgams and determine the thresholds to be used in the next steps
    df_thresh = pd.DataFrame(ls, columns=['yearmon', 'LLLP', 'LLHP', 'HLHP', 'HLLP'])
    df_thresh.to_csv('df_AQNLtrend.csv', index=False, header=True)

def corr_ndarr(A, B):
    A_m = A - A.mean(0)
    B_m = B - B.mean(0)

    ssA = (A_m**2).sum(0)
    ssB= (B_m ** 2).sum(0)

    return np.dot(A_m, B_m.T)/np.sqrt(np.dot(ssA[:,None], ssB[None]))


def get_AQNLshapecorrelation(Airpath, NLpath, shapefilepath):

def get_AQNLpixcorrelation(Airpath, resNLpath):
    # function to correlation between NL and AQ at each pixel level
    lsNL = []
    lsAQ = []

    #datees between which to consider the dataframe
    start_date = datetime(2013,1,1)
    end_date = datetime(2017,1,1)

    for date_d in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

        #get the yearmonth
        yearmon =  (str(date_d.year) + str('%02d' % date_d.month))

        # preapre filenames
        resNLfile = resNLpath +  'DNB10'+str(yearmon) +'.tif'
        Airfile = Airpath + 'MonthRGB'+str(yearmon)+'.tif'

        #run if both of them exist
        if (os.path.isfile(Airfile)&os.path.isfile(resNLfile)):
            print yearmon
            #append all the relvant
            lsNL.append(rio.open(resNLfile).read(1))
            lsAQ.append(rio.open(Airfile).read(1))

    #convert back to array
    lsNL_arr = np.array(lsNL)
    lsAQ_arr = np.array(lsAQ)

        #finding the correaltion




#-------------------------------------------------------------------------------------------------------------------
#------------------------------------------ Running functions --------------------------------------------------------
# *****---------- ONE TIME RUN FUNCTION ------*****
# 1 subset AQ
subsetAQ(AirRGBpath, AirRGBout)
# subset NL
subsetNL(DNBpath, DNBoutIndia)

# 2 resample NL
targetrespath = AirRGBout + 'MonthRGB201501.tif'

for file in glob(os.path.join(DNBoutIndia,  '*' + '.tif')):
    head, tail = ntpath.split(file)
    print tail
    yearmon = tail[3:9]

    #name of resampled raster
    out_raster = DNBout10India + 'DNB10'+str(yearmon) +'.tif'

    resNLarr = resample_NL(file, targetrespath, type = 'sum')
    srs.rio_arr_to_raster(resNLarr, targetrespath, out_raster)


# 3 resample LULC
targetrespath = AirRGBout + 'MonthRGB201501.tif'
LULCsavepath = LULCpath+ '/LCType_India10km2010.tif'
resLULCarr = resample_LULC(LULCpath+'/LCType_India2010.tif', targetrespath, thresh=.5, GIS_C = 13)
srs.rio_arr_to_raster(resLULCarr, targetrespath, LULCsavepath )


# 4 resample NLbuffer
NLbuffersavepath = gb_path +'/Data/Data_process/India_urban_buffer/urbanbuffer10km.tif'
resNLbufferarr = resample_LULC(NLbufferpath, targetrespath, thresh=1, GIS_C = 1)
srs.rio_arr_to_raster(resNLbufferarr, targetrespath, NLbuffersavepath )



# RUN THEM Iteratively

# 5 plot AQNL, scatter plot and get summary
# PLEASE FIND A SUITABLE THRESHOLD
thresh = [50,30,4000] #[R, B, sum of TNL at MODIS res]

resLULCpath = LULCsavepath
for resNLfile in glob(os.path.join(DNBout10India,  '*' + '.tif')):
    head, tail = ntpath.split(file)
    print tail
    yearmon = tail[3:8]
    Airfile = AirRGBout + 'MonthRGB'+str(yearmon)+'.tif'
    NLbuffersavepath = gb_path +'/Data/Data_process/India_urban_buffer/urbanbuffer10km.tif'
    if os.path.isfile(Airfile):
        #yearmon = '201501'
        get_summary(yearmon, resNLfile, Airfile, resLULCpath, NLbuffersavepath, thresh)



# 6 Get trend of AQNL by first getting its df
resNLpath = DNBout10India
Airpath = AirRGBout
df_urbanclass = get_df(resNLpath, Airpath, resLULCpath, NLbufferpath, thresh)

# plot trend of AQNL
df_urbanclass = pd.read_csv(os.getcwd()+'/df_AQNLtrend.csv', header=0)
trendAQNL(df_urbanclass)





#try for correaltion
A_m = A - A.mean(0)
B_m = B - B.mean(0)

ssA = (A_m**2).sum(0)
ssB= (B_m ** 2).sum(0)

c = (A[:,:,None]*B).sum(axis =1)

