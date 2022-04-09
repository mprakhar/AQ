#__author__ = 'Prakhar MISRA'
# Created 2/16/2016
#Last edit 2/29/2016

# Purpose: This code will gather the statistics ( read statistics for Indian shapefile based on rasters) generated from the
# shp_rstr_stat module for each shapefile over a range of different reasters and then write it into a csv file.
# One can tweak the gatherd info to calculate caoorelation, scatter plot or whatever.
# Location of output: E:\Acads\Research\AQM\Data process\Code results

from osgeo import gdal, gdalnumeric, ogr, osr
from PIL import Image, ImageDraw
import os, sys, fnmatch
import numpy
from matplotlib import pyplot as plt
import shapefile
from subprocess import call
import shp_rstr_stat as srs
import pandas as pd
from glob import glob
import datetime
import infoFinder as info

# Inputs required
raster_path= r'D:/75N060E'      # the path where all the raster are stored
# raster_path='F:\Data OMI MODIS\MODIS\L4 MODIS\L4\\'
# raster_path = 'F:\Data OMI MODIS\OMI\L3\\'
shapefile_path = 'E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp'  #IND_adm3_ID_3__299.shp  # the place all the split shapefiels are stored
df_shpatt = pd.read_csv('E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp\IND_adm3.csv', header=0)  # List of all shape file. making dataframe of shape file attribute list
f_type = 'm'        # m is monthly, d is daily
sat = 'VIIRS'       # options VIIRS, MODIS, OMI
prod = 'DNBRAD'        # options DNB, AOD, ANG, NO2, SO2, DNBRAD refers to radiance values obtained from DN values
ver = 'v1'          #version number of output
dor = '20160229'    #date of run


# * * * *  * * # * * * *  * * # * * *     Creating statistic csv file for each shapefile X image      *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *

def all_stt(sat, prod, raster_path, shapefile_path):  # full india scorecard :)

    ind2 =  [tuple( ['ID_3']+ range( min((df_shpatt['ID_3'])), 1+max((df_shpatt['ID_3'])) ) )]  # list of all ID_3 numbers  concerned
    filenamelist = info.filenamefinder(raster_path, sat, prod)
    # print filenamelist
    #opening the raster, filewise from filenamelist
    for raster_name in filenamelist:
        blank=[None for i in range(min(df_shpatt['ID_3']), 1+max(df_shpatt['ID_3']))]
        print raster_name
        year = info.yearfinder(raster_name, sat)   #extract the yearmonth e,g, 201401
        if int(year) <=201400: # condition becasue VIIRS start from 201401 but MODIS and others starrt from way earlier. doing this saves time for MODIS processing
            continue
        dl_sum=['sum'+year]+blank
        dl_mean=['mean'+year] + blank
        dl_med=['med'+year] + blank
        dl_std=['std'+year] + blank
        sd=os.path.normpath(raster_name)

        #opening each vector and storing their info
        for vector in info.findVectors(shapefile_path, '*.shp'):
            (vinfilepath, vinfilename)= os.path.split (vector)
            input_zone_polygon = vinfilepath+ '/' + vinfilename
            sd=os.path.normpath(vector)
            id, (dl_sum[int(id)], dl_mean[int(id)], dl_med[int(id)], dl_std[int(id)]) = list(srs.loop_zonal_stats(input_zone_polygon, raster_name))

        ind2.append(tuple(dl_sum))
        ind2.append(tuple(dl_mean))
        ind2.append(tuple(dl_med))
        ind2.append(tuple(dl_std))

    df = pd.DataFrame(ind2)
    df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_'+sat+prod+'_'+dor+ver+'.csv', index=False, header=False)

all_stt(sat, prod, raster_path, shapefile_path)




# * * * *  * * # * * * *  * * # * * * *  * *#       Creating csv of correlations per shapefile statistic      *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *
# some inputs
date1 = datetime.date(2014, 1, 1)
date2 = datetime.date(2014, 12, 31)
prod1='DNB'
prod2='NO2'
statlist = ['sum', 'mean', 'med', 'std'] # update here if any new stat added
dfhdr = pd.read_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_OMINO2_20160226v1.csv', header=0) #df from any hdr file e.g. aod, ang, no2, so2
dfvrs = pd.read_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_VIIRSDNB.csv', header=0) # df for VIIRS


daydiff = (date2-date1).days   #difference in number of days
mdiff = int(daydiff/365)*12+int((daydiff%365)/30) #difffernece n number of months
nstat= len(statlist) #number of stat features being considered sum , medean, mean , std
# dfhdr.ix[0:mdiff*nstat+nstat-1:nstat,0]
for k in range(0,nstat): #k is counter for type of stat required
    corrlist=[]
    meanhv=[('ID_3', prod2+'list', prod1+'list')]
    j=0
    for i in dfvrs.dtypes.index[1:]:
        j+=1
        # print dfhdr[j], dfvrs[i]
        # to find the those dfrs vaues which have mean , dfvrs[dfvrs['ID_3'].str.contains(statlist[k)]['ID_3']
        hdrlist=pd.to_numeric(dfhdr.ix[k:mdiff*nstat+k-1:nstat,j], errors=coerce) #step by nstat..because step takes place after the range ends too..therefore (-1) to bring back
        # vrslist=pd.to_numeric(dfvrs[dfvrs['ID_3'].str.contains(statlist[k])][i], errors=coerce) #pd.numeric is a very useful function for pandas
        vrslist = pd.to_numeric(dfvrs.ix[k:mdiff*nstat+k-1:nstat,j], errors=coerce)
        corrlist.append((j, numpy.corrcoef(hdrlist,vrslist)[1][0]))
        meanhv.append((j, numpy.mean(hdrlist),numpy.mean(vrslist)))
        print 'loop',i

    df = pd.DataFrame(corrlist)
    df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\Shpcorr_'+statlist[k]+'_'+prod1+prod2+'_'+dor+ver+'.csv', index=False, header=False)
    df = pd.DataFrame(meanhv)
    df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\Shpmean_'+statlist[k]+'_'+prod1+prod2+'_'+dor+ver+'.csv', index=False, header=False)
# * * * *  * * # * * * *  * * # * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *









