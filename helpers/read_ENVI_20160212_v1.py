#__author__ = 'Prakhar'
# Created 2/12/2016
#Last edit 2/12/2016

# Purpose: This code will read the ENVI files that have aready been processed at Monthly level (by specifying the kind of file MODIS and the type of data product ) in batch mode. also it will give the values for select pixel coodridinates for the time series data.
# the purpose os not only to read the Monthly level provcessed files but also to find outliers in the data acquisition
# Location of output: E:\Acads\Research\AQM\Data process\Code results

import numpy as np
import spectral.io.envi as envi
import datetime
import time
import sys
import os
import csv
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
from glob import glob
from sets import Set
from array import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

# import city_master_MODIS_v1 as cm
import city_master_OMI_v1 as cm


# INPUTS
# path = "F:\Data OMI MODIS\MODIS\L4 MODIS\L4\\"
path = "F:\Data OMI MODIS\OMI\L3\\"
f_type = 'm'        # m is monthly, d is daily
sat = 'OMI'       # options MODIS, OMI
prod = 'SO2'        # options AOD, ANG, NO2, SO2
ver = 'v2'          #version number of output
dor = '20160212'    #date of run
city_c = 20         # count of number of cities
dl = [['yearm', 'year', 'mon',  'C1_Agra', 'C2_Ahmedabad' , 'C3_Allahabad'  , 'C4_Amritsar' , 'C5_Chennai' , 'C6_Firozabad' , 'C7_Gwalior' , 'C8_Jodhpur' , 'C9_Kanpur' , 'C10_Kolkata' , 'C11_Lucknow' , 'C12_Ludhiana' ,
          'C13_Mumbai' , 'C14_New_Delhi' , 'C15_Patna' , 'C16_Raipur' , 'C17_Bangalore', 'C18_Hyderabad', 'C19_Jaipur', 'C20_Pune']]

# PROCESSING

# all the AOD and ANG files names are read as follows
sufx = '' if sat == 'MODIS' else 'ave'
filename = glob(os.path.join(path, '*.'+prod+'*'+sufx+'.Global'))
num_files = len(filename)   #total files of each kind

# AOD, ANG processing outer body begin dl = data list
for i in range (0, num_files):
    print i
    # opening the files
    img = envi.open(filename[i]+".hdr", filename[i])
    # exctracting the dates from the filenames: for MODIS filenames:
    # at daily level it should be- yearjd -18:-11, year -18:-14, jd -14:-11
    # at monthly level it should be- yearjd -17:-11, year -17:-13, jd -13:-11
    yearjd = int(filename[i][-17:-11])
    year = int(filename[i][-17:-13])
    jd = int(filename[i][-13:-11])    #jd is the julian date

    city_val=[yearjd , year, jd] # City_value is the VALUE MATRIX start reading values of the products fro different city into the matrix by appending different cities adding infor for diff dates

    for j in range(1,city_c+1):
        city_loc = cm.options[j]()
        # -1 in subsequent statements because in ENVI coordinates start form 1,1 while in python they start from 0,0
        # the coordinates stored in functions above are those of ENVi images
        val = int(img[city_loc[0]-1,city_loc[1]-1])
        city_val.insert(j+2,val)

    dl.append(city_val)


# OUTPUT SAVING

# converting the datalist to dataframe df
df = pd.DataFrame(dl)
# master file of AOD and ANG values ith DeepBlue alogorithm
df.to_csv('CSVOut\df_city_'+prod+'_'+f_type+'_'+dor+'_'+ver+'.csv', index=False, header=False)



#- - - - - - - -- - - - - - - - - - - - - - - - - P A R T II - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - -

#                   ________________Performing operations on the data read to CSV _________________

# reading from the CSV as a dataframe
df = pd.read_csv('CSVOut\df_'+prod+'_'+f_type+'_'+dor+'_'+ver+'.csv', header=0)


# df['ymd1'] = pd.to_datetime(df['yearm'], format='%Y%m')
df['ymd'] = pd.to_datetime(df['yearm'].astype(str), format='%Y%m')

# assigning 'ymd' as time based index
df = df.set_index('ymd')

# assigning seasons based on the month
df['season0'] = ['W' if x==1 or x==2 or x==11 or x==12 else
                 'S' if x==3 or x==4 or x==5 or x==6 else
                 'R' for x in df['mon'] ]                           #my weather classification

df['season1'] = ['W' if x==1 or x==2 else
                 'S' if x==3 or x==4 or x==5 else
                 'R' if x==6 or x==7 or x==8 or x==9 else
                 'P' for x in df['mon'] ]  #this classification is "Climate profile of India, IMD"; P refers to postmonsoon

df['season3'] = ['W' if x==12 or x==1 or x==2 or x==3 else
                 'S' if x==4 or x==5 or x==6 else
                 'R' if x==7 or x==8 or x==9 else
                 'P' for x in df['mon'] ]  #this classification is wiki IMD clasification

# dividing seasons into brackets of Y1, Y2, Y3, Y4 for analysis: 'seasbrack' = season bracket. The number denotes the serial order of a particular month for the season it belolngs to
# the step below is inefficient as it causes to visit the whole list twice.
df['seasbrack_y'] = df.year.map(str)+['-1' if x==3 or x==7 or x==11 else
                                      '-2' if x==4 or x==8 or x==12 else
                                      '-3' if x==5 or x==9 or x==1 else
                                      '-4' for x in df['mon']]

df.loc[df['mon'] == 1, 'seasbrack_y'] = (df.year-1).map(str)+'-3'
df.loc[df['mon'] == 2, 'seasbrack_y'] = (df.year-1).map(str)+'-4'

df['seasbrack_s'] = df.season0.map(str) + ['-1' if x==3 or x==7 or x==11 else
                                      '-2' if x==4 or x==8 or x==12 else
                                      '-3' if x==5 or x==9 or x==1 else
                                      '-4' for x in df['mon']]

#                   ---- performing analysis on most recent 5 years( or any other duration) at composite level -----

# time after which analysis needs to be performed
startTime = pd.datetime(2001, 12, 1, 0, 0, 0)
endTime = pd.datetime(2015, 12, 1, 0, 0, 0)   # CAUTION : important to note the the duration should be a integral multiple of a year/12 months, else incorrect slopes etc may be reported



# only gathering data after startTime as dfs
dfs = df[df.index>=startTime]
dfs = dfs[dfs.index<=endTime]

# explore following 2 metrics by setting up different starttime .. e.g. 2005, 2010, 2012
# the trend slope for all cities from start to end
for i in range (0,city_c):
    z = np.polyfit(range(0,len(dfs)),dfs[cm.city_list_C[i]], deg=1)
    print cm.city_list [i] , z[0]

# to find the mean prod value from startTime to current for different cities
for i in range(3,city_c+3):
    print dfs.ix[:, i].mean(), dfs.columns[i]


# dfaod[(dfaod['season0']=='W')&(dfaod['year']==2006)]['C1_Agra']

# plt.plot(str(dfaod[dfaod['season0']=='R']['ymd']), dfaod[(dfaod['season0']=='R')]['C1_Agra'])
# c=np.corrcoef([x for x in range(0 , len(dfaod[(dfaod['season0']=='R')]['C1_Agra'])) ], (dfaod[(dfaod['season0']=='R')]['C1_Agra']))
# plt.plot(mdates.date2num((dfaod[(dfaod['season0']=='R')&(dfaod['year']==2006)]['ymd'])), dfaod[(dfaod['season0']=='R')&(dfaod['year']==2006)]['C1_Agra'])
# plt.plot(np.array(dfaod[dfaod['season0']=='R']['ymd']), dfaod[(dfaod['season0']=='R')]['C1_Agra'])

# some examples on plotting
df.drop(['yearm', 'year', 'mon'], axis=1).plot(colormap='jet')  # to plot all overall
df[df['season0']=='R']['C1_Agra'].plot(colormap='jet')          # to plot for one city in one season
df[df.season0=='R'].drop(['yearm', 'year', 'mon'], axis=1).plot(colormap='jet') # to plot for all cities in a season



#                   ------  performing  analysis based on seasonal brackets and plotting graphs -------

# time after which analysis needs to be performed
startTime = pd.datetime(2010, 6, 1, 0, 0, 0)
endTime = pd.datetime(2015, 6, 1, 0, 0, 0)

#                   ---- trial area for plot designs  -----
df[df['season0']=='R']['seasbrack_c'] = [x for x in range(1,100)]

df[df.season0=='R']['C1_Agra'].plot(x='seasbrack_y', style='o')
df[df.season0=='R']['C1_Agra'].plot(x='seasbrack_y', style='--')
df[df.season0=='R'].plot( x='seasbrack_y', y='C1_Agra', style='--')

ax = df[df.season0=='R'].plot( x='seasbrack_y', y='C1_Agra', style='--', label = 'Rain', color = 'Green')
ax.set_xticklabels(list(df['seasbrack_y']))
df[df.season0=='W'].plot( x='seasbrack_y', y='C1_Agra', style='--', label = 'Winter', color = 'Blue', ax=ax)
df[df.season0=='S'].plot( x='seasbrack_y', y='C1_Agra', style='--', label = 'Summer', color = 'Red', ax=ax)
ax.plot(x, fit[0] * x + fit[1], color='red')
fig.show()
#                   ---- trial area  finish -----

# to find the coefficients of regressions for all seasons. we can find for which seasons the situation is getting worse in the cities
for i in range (0,16):
    zr = np.polyfit(range(0,(len(dfs[dfs['season0']=='R']))),dfs[dfs['season0']=='R'][city_list_C[i]], deg=1)
    zs = np.polyfit(range(0,(len(dfs[dfs['season0']=='S']))),dfs[dfs['season0']=='S'][city_list_C[i]], deg=1)
    zw = np.polyfit(range(0,(len(dfs[dfs['season0']=='W']))),dfs[dfs['season0']=='W'][city_list_C[i]], deg=1)

    print city_list [i] , zr[0], zs[0], zw[0], zangr[0], zangs[0], zangw[0]


p = np.poly1d(z)
xr = (len(df[df['season0']=='R']['C1_Agra']))
ax = plt.plot(xr, p(xr), "-r")

# a new method to plot
grouped = df.groupby(['year', 'season0'])
ax = df[df.season0=='R'].groupby('year').mean().plot( y='C1_Agra', style='--', kind='area')
df[df.season0=='W'].groupby('year').mean().plot( y='C1_Agra', style='--', label = 'Winter', color = 'Blue', ax=ax, kind='area', stacked=True)



# -----------------------    Seasonal Plots     HARD CODED - need to improve tremendously         -----------------------------------------
# hard coding because time nahi hai abhi

grouped_Ro = df[df.season0=='R'].groupby('year').mean()
grouped_Wo = df[df.season0=='W'].groupby('year').mean()
grouped_So = df[df.season0=='S'].groupby('year').mean()

pd.concat([grouped_R['C1_Agra'],grouped_S['C1_Agra'], grouped_W['C1_Agra']], axis=1, join_axes=[grouped_R.index] )

for j in range (2,18):
    #  processing for prod
    df_seas_chart=[['Year', 'Rain', 'Summer', 'Winter']]
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    for i in range (0,15):
        Year = 2001+i
        Rain = grouped_Ro.iat[i,j]
        Summer = grouped_So.iat[i,j]
        Winter = grouped_Wo.iat[i,j]
        temp = [Year, Rain, Summer, Winter]
        df_seas_chart.append(temp)

    df_seas_chart = pd.DataFrame(df_seas_chart)
    df_seas_chart.columns = df_seas_chart .iloc[0]
    df_seas_chart = df_seas_chart.reindex(df_seas_chart.index.drop(0))
    ax1 = df_seas_chart.plot(x='Year',kind='area', title = 'Annual trend-Mean seasonal '+prod+': '+city_list[j-2], figsize=(4.83,2.71), legend= False )
    ax1.set_xlabel("Year")
    ax1.set_ylabel(prod)
    fig = ax1.get_figure()
    fig.savefig('PlotOut\\'+prod+'_Y_s_'+city_list[j-2]+'.png', bbox_inches='tight') # save image at year and season level




