#__author__ = 'Prakhar MISRA'
# Created 2/16/2016
#Last edit 11/7/2017

# Purpose: This code will gather the statistics ( read statistics for Indian shapefile based on rasters) generated from the
# shp_rstr_stat module for each shapefile over a range of different reasters and then write it into a csv file.
# One can tweak the gatherd info to calculate caoorelation, scatter plot or whatever.
# Location of output: E:\Acads\Research\AQM\Data process\Code results


import os, sys, fnmatch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import datetime

#private imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
from  spatialop  import shp_rstr_stat as srs
from  spatialop  import infoFinder as info

# Inputs required
# the path where all the raster are stored
raster_path='F:\Data OMI MODIS\MODIS\L4 MODIS\L4\\'
# raster_path = 'F:\Data OMI MODIS\OMI\L3\\'

# the place all the split shapefiels are stored
shapefile_path = 'E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp'  #IND_adm3_ID_3__299.shp


f_type = 'm'        # m is monthly, d is daily
sat = 'VIIRS'       # options VIIRS, MODIS, OMI
prod = 'DNBRAD'        # options DNB, AOD, ANG, NO2, SO2, DNBRAD refers to radiance values obtained from DN values
ver = 'v1'          #version number of output
dor = '20160229'    #date of run


# * * * *  * * # * * * *  * * # * * *     Creating statistic csv file for each shapefile X image      *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *

# List of all shape file. making dataframe of shape file attribute list
df_shpatt = pd.read_csv('E:\Acads\Research\AQM research\Data\Data process\Ind_adm3_splitshp\IND_adm3.csv', header=0)

# full india scorecard :)
def all_stt(sat, prod, raster_path, shapefile_path):

    ind2 =  [tuple( ['ID_3']+ range( min(df_shpatt['ID_3']), 1+max(df_shpatt['ID_3']) ) )]  # list of all ID_3 numbers  concerned
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
    return df

#function end

#run fucntion
df = all_stt(sat, prod, raster_path, shapefile_path)
df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_'+sat+prod+'_'+dor+ver+'.csv', index=False, header=False)



# * * * *  * * # * * * *  * * # * * * *  * *#       Creating csv of correlations per shapefile statistic      *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *
# some inputs
date1 = datetime.date(2014, 1, 1)
date2 = datetime.date(2014, 12, 31)
prod1='DNB'
prod2='NO2'
statlist = ['sum', 'mean', 'med', 'std'] # update here if any new stat added

#df from any hdr file e.g. aod, ang, no2, so2
dfhdr = pd.read_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_OMINO2_20160226v1.csv', header=0)

# df for VIIRS
dfvrs = pd.read_csv('E:\Acads\Research\AQM research\Codes\CSVOut\\df_ShpStat_VIIRSDNB.csv', header=0)


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
        aodlist=pd.to_numeric(dfhdr.ix[k:mdiff*nstat+k-1:nstat,j], errors=coerce) #step by nstat..because step takes place after the range ends too..therefore (-1) to bring back

        # vrslist=pd.to_numeric(dfvrs[dfvrs['ID_3'].str.contains(statlist[k])][i], errors=coerce) #pd.numeric is a very useful function for pandas
        vrslist = pd.to_numeric(dfvrs.ix[k:mdiff*nstat+k-1:nstat,j], errors=coerce)

        corrlist.append((j, np.corrcoef(aodlist,vrslist)[1][0]))
        meanhv.append((j, np.mean(aodlist),np.mean(vrslist)))
        print 'loop',i

    df = pd.DataFrame(corrlist)
    df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\Shpcorr_'+statlist[k]+'_'+prod1+prod2+'_'+dor+ver+'.csv', index=False, header=False)

    df = pd.DataFrame(meanhv)
    df.to_csv('E:\Acads\Research\AQM research\Codes\CSVOut\Shpmean_'+statlist[k]+'_'+prod1+prod2+'_'+dor+ver+'.csv', index=False, header=False)
# * * * *  * * # * * * *  * * # * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *# * * * *  * *




# 3. Next comparing resampled DNB and calibrated radiance from OLS at taluk level by Creating statistic csv file for each shapefile X image . Original in read_India_stats.py


#IND_adm3_ID_3__299.shp  # the place all the split shapefiels are stored
shapefile_path = r'/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/Ind_adm3_splitshp//'

# List of all shape file. making dataframe of shape file attribute list
df_shpatt = pd.read_csv( shapefile_path + 'IND_adm3.csv', header=0)

def all_stt(input_value_raster, shapefile_path,df_shpatt, out_file, ID):  # full india scorecard :)

    ind2 =  [tuple( [ID]+ range( min((df_shpatt[ID])), 1+max((df_shpatt[ID])) ) )]  # list of all ID_3 numbers  concerned
    blank=[None for i in range(min(df_shpatt[ID]), 1+max(df_shpatt[ID]))]
    dl_sum=['sum']+blank
    dl_mean=['mean'] + blank
    dl_med=['med'] + blank
    dl_std=['std'] + blank
    sd=os.path.normpath(input_value_raster)

    #opening each vector and storing their info
    for vector in info.findVectors(shapefile_path, '*.shp'):
        (vinfilepath, vinfilename)= os.path.split (vector)
        input_zone_polygon = vinfilepath+ '/' + vinfilename
        sd=os.path.normpath(vector)
        id, (dl_sum[int(id)], dl_mean[int(id)], dl_med[int(id)], dl_std[int(id)]) = list(srs.loop_zonal_stats(input_zone_polygon, input_value_raster, ID))

    ind2.append(tuple(dl_sum))
    ind2.append(tuple(dl_mean))

    df = pd.DataFrame(ind2).transpose()
    df.to_csv(csv_save_path+'df_ShpStat_'+out_file, index=False, header=False)
# function end

# Running funct all_stt for VIIRS 2013
#file for which the stat are to be found out
dnb.file = r'/home/prakhar/Research/AQM_research/Data/Data_process/Resampl_img/ResamplDNB_OLS_2013.tif'
all_stt(dnb.file, shapefile_path,df_shpatt,out_file= 'DNB2013', ID='ID_3')

# Running funct all_stt for OLSrad calibration1 2013
ols.file = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512//' + 'RadOLS2013.tif'
all_stt(ols.file, shapefile_path,df_shpatt,out_file= 'RadOLS2013', ID='ID_3')

# Running funct all_stt for OLSrad calibration1 2013
ols.file = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512//' + 'Calib1OLS2013.tif'
all_stt(ols.file, shapefile_path,df_shpatt,out_file= 'Calib1OLS2013', ID='ID_3')

# # Running funct all_stt for OLSrad calibration1 2013
# ols.file = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512//' + 'Calib2_OLS2013.tif'
# all_stt(ols.file, shapefile_path,df_shpatt,out_file= 'Calib2_OLS2013', ID='ID_3')
#

# input_zone_polygon =  '/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/Ind_adm3_splitshp/IND_adm3_ID_3__999.shp'
# input_value_raster = dnb.file




# * * * *  * * # * * * *  * * # * * * *  * *# # *  Task IV OLS-TNL sum for GDP verfication at country level*  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Zhang/'
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
input_zone_polygon = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'

for year in range(2001, 2014):
    ols.file = glob(os.path.join(ols.path, '*'+str(year)+ '*'+'.tif'))[0]
    print year, srs.zonal_stats(input_zone_polygon, ols.file)[0]

df_nlgdp = pd.read_csv(csv_in_path+'OLS_intercalibration_GDP.csv', header=0 )
df_nlgdp[0:13].corr("pearson")
# 2001 - 2013coorelation:  Wu 0.89, Zhang - 0.87

#function to plot th4e TNL and GDP trends with time
def plot_NLGDP(df_nlgdp):

    fig, ax1 = plt.subplots()
    ax1.plot(df_nlgdp.Year, df_nlgdp.TNL_Wu, 'ko--', label='Wu calibration')
    ax1.plot(df_nlgdp.Year, df_nlgdp.TNL_Zhang, 'bo--', label='Zhang calibration')
    ax1.set_xlabel('Year', fontsize=16)
    ax1.set_ylabel('TNL', fontsize=16)
    ax1.legend(loc=4, fontsize =16)
    ax2=ax1.twinx()
    ax2.plot(df_nlgdp.Year, df_nlgdp.GDP_per_capita,  'r-', label='GDP')
    ax2.set_ylabel('GDP (US$)', fontsize=16, color='r')
    # ax2.legend()
    plt.title('Comparison of GDP with calibrated DMSP-OLS images \n by Wu method and Zhang method', fontsize=20)
    plt.show()








