#!/usr/bin/env python
# -*- coding: utf-8 -*-
#__author__ = 'Prakhar MISRA'
# Created 9/27/2017
#Last edit 9/27/2017

'''# Purpose: This code will read the MOD14 product to count the the pixels above certain FRP threshold to realte with agro fires '''

#Contents:
# 1. cosnider 10km, 20km, 30km radius neughborhood of the central pixel
# 2. count pixels greater than a threshold.
# 3. save this information at daily level and the annual level




import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os.path
from glob import glob
from datetime import timedelta, date
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from itertools import product
from adjustText import adjust_text
from sklearn import datasets, linear_model
# import os
# os.chdir('/home/prakhar/Research/AQM_research/Codes/')
import sys

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
# pvt imports
from spatialop import *
# pvt imports
from spatialop import infoFinder as info
from spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
from spatialop import im_mean_temporal as im_mean
from spatialop import coord_translate as ct


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
#Input

#prefix
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path from my folders
#mother godown
FRPpath = r'/home/wataru/research/MODIS/MOD14/forMIRAIKAN/'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut/AQFRP/'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/AQFRP/'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'

# citylist : funxtion to generate distance image from city centre
citylist_path = csv_in_path + '/CityList.csv'



def get_val(dffrp, lat, lon, radius):
    # function to return the list of FRP of pixels surrounding the centre as frplist.
    # considers the datafrme of frp 'dffrp'
    # and lat and lon in degrees
    # and radius in kilometres
    #return a list frplist


    # 1 deg  = ~110km; 10km =~0.1deg; 1km = ~.01deg
    radiusdeg = radius*0.01

    #list of FRP of pixels surrounding the centre
    frplist = dffrp[(np.abs(dffrp.lat-lat)<=radiusdeg) & (np.abs(dffrp.lon-lon)<=radiusdeg)]['FRP'].tolist()

    return frplist
#function end

def summaryfrp(frplist, frpthreshold):
    # function to return the number of pixels above frp threshold and mean of those pixels
    # frplist  list of FRP of pixels surrounding the centre
    # frpthreshold threshold above frp is consiedered as fire

    abovefrp = [x  for x in frplist if x>= frpthreshold]

    return [np.size(abovefrp), np.nanmean(abovefrp)]
#function end

def getFRPinfo(frp_filepath,df_citylist, date, radius =200, frpthreshold = 30 ):
    #read the filepath and calls previous for the cities mentioned in citylistpath and returns a distilled frp list/df or whateva
    dffrp = pd.read_csv(frp_filepath)

    #assings name to columns becuase these are essentially column liess files
    dffrp.columns = ['lat', 'lon', 'FRP']

    #check for all cities
    lscount = []
    lsmean = []
    for index, cityrow in df_citylist.iterrows():
        #print cityrow['City']
        #get all qualifying frp vallues
        #provide the distane upto  whch fire must be counted
        frplist = get_val(dffrp, cityrow['Lat'], cityrow['Lon'], radius = radius)

        # get count above threshold and their mean
        #provide the threshold abov e which  to consider the FRP
        summary = summaryfrp(frplist, frpthreshold = frpthreshold)

        lscount.append(summary[0])
        lsmean.append(summary[1])
    lscount.append(date)
    lsmean.append(date)

    return (lscount,lsmean)
#function end


def getFRP(FRPpath,citylist_path, FRPTHRESHOLD = 30, RADIUS = 200 ):


    #set radius and frpthreshold



    #function to produce df containing FRP mewtric of all cities
    lsfrpcount = []
    lsfrpmean = []

    df_citylist = pd.read_csv(citylist_path, header=0)

    for filepath in glob(os.path.join(FRPpath, '*'+'.csv')):

        #get date of the filepath
        date = filepath[-16:-8]
        print date

        try :
            (lscount, lsmean) = getFRPinfo(filepath,df_citylist, date, radius =RADIUS, frpthreshold = FRPTHRESHOLD)
            lsfrpcount.append(lscount)
            lsfrpmean.append(lsmean)
        except:
            continue

    #column names for df
    columns = df_citylist.Count.tolist()
    columns.append('date')
    df_count = pd.DataFrame(lsfrpcount, columns = columns )
    df_mean = pd.DataFrame(lsfrpmean, columns=columns)
    # save the df
    df_count.to_csv(csv_save_path+ 'df_20cityFRPcount_'+str(RADIUS)+'_'+str(FRPTHRESHOLD)+'.csv', header=True)
    df_mean.to_csv(csv_save_path + 'df_20cityFRPmean_'+str(RADIUS)+'_'+str(FRPTHRESHOLD)+'.csv', header=True)


def plotfrp_annual(df, Countid, cityname, labely, rolling = False):
    #function to plot the FRP trend for each city in the city

    #Countid = 'C02'
    #convert to datettime
    df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d')

    #getting the doy
    df['doy'] = df['date'].dt.dayofyear

    # new figure
    plt.figure()
    ax = plt.subplot(111)
    ax.set_color_cycle(sns.color_palette('coolwarm_r', 16))
    #plt.style.use('seaborn-dark-palette')
    #plotting for each year
    for yy in range(2001, 2017):

        #extract df for yy
        df_yy = df[df.date.dt.year==yy]

        #variables
        if rolling:
            x = pd.rolling_mean(df_yy.doy, 8)
            y = pd.rolling_mean(df_yy[Countid], 8)

        else:
             x = df_yy.doy
             y = df_yy[Countid]
        #y.fillna(0, inplace = True)

        #plot for this year
        ax.plot(x,y, label = np.str(yy))

    plt.legend()
    plt.xlabel('DOY')
    plt.ylabel(labely)
    plt.title(cityname)
    plt.savefig(plt_save_path+'AnnualFRP'+labely+Countid +'.png')
    plt.show()
#functio ned

def saveplot(FRPTHRESHOLD = 0, RADIUS = 200):
    #call the plotting function
    df_citylist = pd.read_csv(citylist_path, header=0)
    labely = 'count'
    df = pd.read_csv(csv_save_path+ 'df_20cityFRP'+labely+'_'+str(RADIUS)+'_'+str(FRPTHRESHOLD)+'.csv', header = 0)
    for index, row in df_citylist.iterrows():
        plotfrp_annual(df, row.Count, row.City, labely=labely, rolling = False)


#to run
#on 20170928, parameters - FRPTHRESHOLD = 30, RADIUS = 200
#generate csv pf all FRP
getFRP(FRPpath,citylist_path, FRPTHRESHOLD = 0, RADIUS = 1500 )
#generate and save plots
saveplot(FRPTHRESHOLD = 0, RADIUS = 1500)
















