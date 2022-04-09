# __author__ = 'Prakhar'
# Created 8/28/2016
# Last edit 11/22/2016

#Purpose: (1) to gather all GDP from state levell csv in df
#         (2) to gather all sumAQ from urban and clean AQ images at state level in df
#         (3) to put together result from (1) and (2) in comon df state wise
#         (4) perform AQ GDP correlation and difference and plot correaltion
#         (5) to perform Granger casuality test
#         (6) Normalize GDP each state sector and find PCA i All, ii GCT

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
from scipy.optimize import curve_fit
from scipy import stats
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
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import sys

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
#Pvt imports
import shp_rstr_stat as srs
import coord_translate as ct
import my_math as mth
import my_plot as mpt
import classRaster
from classRaster import Raster_file
from classRaster import Image_arr
import infoFinder as info
from GDP_sectors import rename_dict, ren_sectors, sectors, ren_series


# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path


aod = Raster_file()
aod.path = gb_path + '/Data/Data_process/Meanannual_MODIS//'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = gb_path + '/Data/Data_process/Meanannual_MODIS/'+ '/clean/cleanAOD2001.tif'
aod.georef = gb_path + '/Data/Data_process/Georef_img//MODIS_georef.tif'

ang = Raster_file()
ang.path = gb_path + '/Data/Data_process/Meanannual_MODIS//'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path +  '/Data/Data_process/Meanannual_MODIS/'+ '/clean/cleanANG2001.tif'
ang.georef = gb_path + '/Data/Data_process/Georef_img//MODIS_georef.tif'

no2 = Raster_file()
no2.path = gb_path + '/Data/Data_process/Meanannual_OMI/'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = gb_path +  '/Data/Data_process/Meanannual_OMI/'+ '/clean/cleanNO22005.tif'
no2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

so2 = Raster_file()
so2.path = gb_path + '/Data/Data_process/Meanannual_OMI/'
so2.sat = 'OMI'
so2.prod = 'SO2'
so2.sample = gb_path + '/Data/Data_process/Meanannual_OMI/'+ '/clean/cleanSO22005.tif'
so2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

shapefile_path = gb_path+r'/Data/Data_process/Shapefiles/Ind_adm1_splitshp//'  #IND_adm3_ID_3__299.shp  # the place all the split shapefiels are stored
df_shpatt = pd.read_csv( shapefile_path + 'IND_adm1.csv', header=0)  # List of all shape file. making dataframe of shape file attribute list
eco_dst = gb_path +r'/Data/Data_process/Economics/District_level_GDP//'

input_zone_polygon_0 = gb_path+'/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
input_zone_polygon_1 = gb_path+'/Data/Data_raw/Shapefiles/IND_adm1/IND_adm1.shp'

csv_in_path = gb_path+r'/Codes/CSVIn//'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'


# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task I  Gather GDP from all state csv     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

st_path = gb_path + r'/Data/Data_process/Economics/State_level_GSDP//'

for st_gdp in glob(os.path.join(st_path, '*'+'.csv')):
    st_name = os.path.basename(st_gdp)[23:-20]
    print st_name
    df_stgdp = pd.read_csv(st_gdp, header=0) # state level  sector wise GDP
    df_stgdp = df_stgdp.transpose()
    df_stgdp.columns = df_stgdp.iloc[1]
    # df_stgdp.reindex(df_stgdp.index.drop(1))
    df_stgdp.drop(df_stgdp.index[0], inplace=True)     # remving old header
    df_stgdp.drop(df_stgdp.index[[0]], inplace=True)        # removinf duplicate header
    df_stgdp['year'] = df_stgdp.index.str.split('-').str.get(0)
    df_stgdp.to_csv( csv_save_path + 'GDP_AQ/df_'+st_name+'.csv')




# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task II  Gather TAQ from all state aq images     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# Function to all AQ values for all years for all states in a file

def TAQ(type):
    lstid = []  # ls for state and their id
    shp = ogr.Open(input_zone_polygon_1)
    lyr = shp.GetLayer()
    for i in lyr:
        # print  i.GetField('NAME_1')
        lstid.append([i.GetField('ID_1'), i.GetField('NAME_1')])
    df=pd.DataFrame(lstid)
    df.columns=['ID_1', 'NAME_1']

    for prodT in [aod, ang, so2, no2]:
        for prodT.file in glob(os.path.join(prodT.path + type + '/',  '*'+ prodT.prod + '*' + '.tif')):
            year = int(prodT.file[-8:-4])
            df_staq =  srs.all_stt(prodT.file, shapefile_path, 'ID_1', prodT.prod + str(year))
            df_staq1 = df_staq.filter(regex='sum')
            df[prodT.prod + str(year)] = df_staq1
    df.to_csv(csv_save_path + '/GDP_AQ/'+'df_StateAQ_'+type+'.csv', header=True, index=False)
    print ('done')
# function end

# Run twice:once for 'Urban' images and once of 'Clean' imeages
type = 'clean'  # urban or clean?
TAQ(type)

# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task III  Putting all results together     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def put_together(type):
    df_staq = pd.read_csv(csv_save_path + '/GDP_AQ/' + 'df_StateAQ_' + type + '.csv', header=0)

    for st_name in df_staq['NAME_1']:

        if st_name not in ['Dadra and Nagar Haveli','Daman and Diu', 'Lakshadweep'] :

            df_stgdp = pd.read_csv(csv_save_path + 'GDP_AQ/df_' + st_name + '.csv')  # state gdp
            df_stgdpc = df_stgdp.copy(deep=True)
            df_stgdpc['year1'] = df_stgdpc['year']
            df_stgdpc = df_stgdpc.set_index('year1')

            for prodT in [aod, ang, so2, no2]:
                print st_name

                df_staq_prod = df_staq.filter(regex= prodT.prod)        #_prod for a sepcific prodict
                df_staq_prod['NAME_1'] = df_staq['NAME_1']
                df_staq_prod['ID_1'] = df_staq['ID_1']
                df_staq_prod = df_staq_prod[df_staq_prod.NAME_1==st_name]
                df_stgdpc['sum'+type+prodT.prod]='' # c is copy
                for yr in df_stgdpc['year']:
                    if type in 'urban' and yr>2014:
                        continue
                    # df_stgdpc.set_value(int(yr), ('sum' + type + prodT.prod), float(df_staq_prod[prodT.prod + str(int(yr))]))

                    if (prodT.sat is 'OMI') and (yr>=2004):
                        df_stgdpc.set_value(int(yr), ('sum' + type +prodT.prod), float(df_staq_prod[prodT.prod+str(int(yr))]))
                    if prodT.sat is 'MODIS':
                        df_stgdpc.set_value(int(yr), ('sum' + type + prodT.prod), float(df_staq_prod[prodT.prod + str(int(yr))]))

                fname = csv_save_path + '/GDP_AQ/' + 'df_' + 'AQGDP' + type +'_'+ st_name + '.csv'
                df_stgdpc.to_csv(fname, header=True)
# function end

type = 'urban'  # urban or clean?
put_together(type)
# data stored as df_AQGDPclean_Manipur/ df_AQGDPurban_Manipur



# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task IV  Correlation and Plot     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# (a) Correalte

df_staq = pd.read_csv(csv_save_path + '/GDP_AQ/' + 'df_StateAQ_' + type + '.csv', header=0)
sectors = ['Agriculture',
        'Forestry & logging',
        'Fishing',
        'Ag & Allied',
        'Mining & quarrying',
        'Manufacturing',
        'Registered',
        'Unregistered',
        'Construction',
        'Electricity,gas and Water supply',
        'Industry',
        'Transport,storage & communication',
        'Railways',
        'Transport by other means',
        'Storage',
        'Communication',
        'Trade,hotels and restaurants',
        'Banking & Insurance',
        'Real estate,ownership of dwellings and business services',
        'Public administration',
        'Other services',
        'Services']

ls =[]
for prodT in [aod, ang, so2, no2]:
    for st_name in df_staq['NAME_1']:
        if st_name not in ['Dadra and Nagar Haveli', 'Daman and Diu', 'Lakshadweep']:
            fnameu = csv_save_path + '/GDP_AQ/' + 'df_' + 'AQGDP' + 'urban' + '_' + st_name + '.csv'        # fname urban - urban means including urban area
            fnamec = csv_save_path + '/GDP_AQ/' + 'df_' + 'AQGDP' + 'clean' + '_' + st_name + '.csv'        # fname clean - clean means including all state
            df_stgdpcu = pd.read_csv(fnameu, header=0)[:-1]     #urban
            df_stgdpcc = pd.read_csv(fnamec, header=0)     #clean
            for sec in sectors:
                ls.append([st_name, sec, prodT.prod, np.corrcoef(df_stgdpcc[sec], df_stgdpcc['sum'+'clean'+prodT.prod] )[1][0], np.corrcoef(df_stgdpcu[sec], df_stgdpcu['sum'+'urban'+prodT.prod] )[1][0], np.corrcoef(df_stgdpcu[sec], (df_stgdpcc['sum'+'clean'+prodT.prod]-df_stgdpcu['sum'+'urban'+prodT.prod]).dropna() )[1][0]])

df_ls = pd.DataFrame(ls, columns=['NAME_1', 'sector', 'prod', 'corr_clean', 'corr_urban', 'corr_non_urban' ])
df_ls.to_csv(csv_save_path + '/GDP_AQ/' + 'df_' + 'AQGDPCorr.csv' ,header=True, index=False)


# (b) Plotting

import seaborn as sns
df_gdpcor = pd.read_csv(csv_save_path + '/GDP_AQ/' + 'df_' + 'AQGDPCorr.csv' , header=0)


# (b.1) Producing all plots individually by AQ
# st_name='Uttar Pradesh'
for st_name in df_gdpcor.NAME_1.unique():
    for prodT in [aod, ang, so2, no2]:
        df_gdpcor1=df_gdpcor[df_gdpcor.sector.isin(sectors)].copy(deep=True)
        df_gdpcor1.replace({'sector':rename_dict}, inplace=True)
        df_gdpcor_st = df_gdpcor1[(df_gdpcor1.NAME_1==st_name) &(df_gdpcor1['prod']==prodT.prod)]
        # df_gdpcor_st=df_gdpcor_st.reset_index(drop=True)
        df_gdpcor_st.reset_index(drop=True, inplace=True)

        f, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df_gdpcor_st.index, df_gdpcor_st['corr_urban'], width=.9, color="#278DBC", align="center", label="Urban")
        # ax.bar(df_gdpcor_st.index, df_gdpcor_st['corr_clean'], width=.65, color="#000099", align="center", label="Clean")
        ax.bar(df_gdpcor_st.index, df_gdpcor_st['corr_non_urban'], width=.65, color="#000099", align="center", label="Non-urban")
        ax.set(xlim=(min(df_gdpcor_st.index)-1, max(df_gdpcor_st.index)+1), xticks=df_gdpcor_st.index, xticklabels=df_gdpcor_st.sector.values)
        plt.title('Sector GDP and '+prodT.prod+' correlation: '+st_name, fontsize=17)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=90, fontsize = 14)
        ax.xaxis.grid(True)
        sns.despine(left=True)
        ax.legend(ncol=2, loc=1, fontsize=14);
        plt.tight_layout()
        plt.savefig(plt_save_path + 'GDPAQ_' +prodT.prod+'_'+ st_name + '.png')

# (b.2) Producing all plot statewise with al AQ together
# st_name='Uttar Pradesh'
for st_name in df_gdpcor.NAME_1.unique():
    df_gdpcor1=df_gdpcor[df_gdpcor.sector.isin(sectors)].copy(deep=True)
    df_gdpcor1.replace({'sector':rename_dict}, inplace=True)
    df_gdpcor_st1 = df_gdpcor1[(df_gdpcor1.NAME_1==st_name) &(df_gdpcor1['prod']=='AOD')]
    df_gdpcor_st1.reset_index(drop=True, inplace=True)
    df_gdpcor_st2 = df_gdpcor1[(df_gdpcor1.NAME_1==st_name) &(df_gdpcor1['prod']=='ANG')]
    df_gdpcor_st2.reset_index(drop=True, inplace=True)
    df_gdpcor_st3 = df_gdpcor1[(df_gdpcor1.NAME_1==st_name) &(df_gdpcor1['prod']=='SO2')]
    df_gdpcor_st3.reset_index(drop=True, inplace=True)
    df_gdpcor_st4 = df_gdpcor1[(df_gdpcor1.NAME_1==st_name) &(df_gdpcor1['prod']=='NO2')]
    df_gdpcor_st4.reset_index(drop=True, inplace=True)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(15, 12), sharex=True)
    ax1.bar(df_gdpcor_st1.index, df_gdpcor_st1.corr_urban, width=.9, color="#278DBC", align="center", label="Urban")
    # ax1.bar(df_gdpcor_st1.index, df_gdpcor_st1.corr_clean, width=.65, color="#000099", align="center", label="All")
    ax1.bar(df_gdpcor_st1.index, df_gdpcor_st1.corr_non_urban, width=.65, color="#000099", align="center", label="Non-urban")
    ax1.set_title('Sector GDP and AQ correlation: ' + st_name, fontsize=17)
    ax1.text((max(df_gdpcor_st1.index))+0.5, 0.0, 'AOD',  fontsize=20, fontweight='bold')

    ax2.bar(df_gdpcor_st2.index, df_gdpcor_st2.corr_urban, width=.9, color="#278DBC", align="center", label="Urban")
    # ax2.bar(df_gdpcor_st2.index, df_gdpcor_st2.corr_clean, width=.65, color="#000099", align="center", label="All")
    ax2.bar(df_gdpcor_st2.index, df_gdpcor_st2.corr_non_urban, width=.65, color="#000099", align="center", label="Non-urban")
    ax2.text((max(df_gdpcor_st2.index))+0.5, 0.0, 'ANG',  fontsize=20, fontweight='bold')

    ax3.bar(df_gdpcor_st3.index, df_gdpcor_st3.corr_urban, width=.9, color="#278DBC", align="center", label="Urban")
    # ax3.bar(df_gdpcor_st3.index, df_gdpcor_st3.corr_clean, width=.65, color="#000099", align="center", label="All")
    ax3.bar(df_gdpcor_st3.index, df_gdpcor_st3.corr_non_urban, width=.65, color="#000099", align="center", label="Non-urban")
    ax3.text((max(df_gdpcor_st3.index))+0.5, 0.0, 'SO$_2$',  fontsize=20, fontweight='bold')

    ax4.bar(df_gdpcor_st4.index, df_gdpcor_st4.corr_urban, width=.9, color="#278DBC", align="center", label="Urban")
    # ax4.bar(df_gdpcor_st4.index, df_gdpcor_st4.corr_clean, width=.65, color="#000099", align="center", label="All")
    ax4.bar(df_gdpcor_st4.index, df_gdpcor_st4.corr_non_urban, width=.65, color="#000099", align="center", label="Non-urban")
    ax4.text((max(df_gdpcor_st4.index))+0.5, 0.0, 'NO$_2$',  fontsize=20, fontweight='bold')
    ax4.set(xlim=(min(df_gdpcor_st4.index)-1, max(df_gdpcor_st4.index)+1), xticks=df_gdpcor_st1.index, xticklabels=df_gdpcor_st1.sector.values)

    labels = ax4.get_xticklabels()
    plt.setp(labels, rotation=90, fontsize = 20)
    ax1.xaxis.grid(True)
    sns.despine(left=True)
    ax1.legend(ncol=2, loc=1, fontsize=14);
    plt.tight_layout()
    plt.savefig(plt_save_path + 'GDPAQ_' + st_name + '.png')





# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task V  Granger causality Test     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# (a) Test for stationarity

# adfuller test of stationarity
def adfullertest(dataseries):
    d_order0=ts.adfuller(dataseries,1, autolag='AIC')
    print 'adf: ', d_order0[0]
    print 'p-value: ', d_order0[1]
    print'Critical values: ', d_order0[4]       # critical values from 'Critical values for cointegrastion tests', James MacKinon, 2010

    if d_order0[0]> d_order0[4]['10%']:
        print 'Time Series is  nonstationary'
        # print d
    else:
        print 'Time Series is --STATIONARY--'
        # print d
    return d_order0
#func end

# differencing function
def differencing(dataseries, i):
        if i ==1:
            # print 'diff 1'
            return (np.diff(dataseries, 1))

        if i == 2:
            # print 'diff 2'
            return (np.diff(dataseries, 2))

        if i == 3:
            # print 'roll 2'
            return (pd.rolling_mean(pd.Series(dataseries), 2).dropna())

        if i == 4:
            # print 'roll 3'
            return (pd.rolling_mean(pd.Series(dataseries), 3).dropna())
        if i== 5:
            # print 'log 1'
            return (np.diff(np.log(dataseries), 1))

#func end

# Trying different differencing methods to achieve stationarity
ls_stdiff =[]       # stores all the stationarity techniques
for st_name in df_gdpcor.NAME_1.unique():
    df_stgdpaq = pd.read_csv(csv_save_path + '/GDP_AQ//' + 'df_AQGDPurban_'+st_name+'.csv', header=0)
    df_stgdpaq.rename(index=str, columns=rename_dict, inplace=True)
    print st_name

    # running for all state-series
    for ser in ren_series:
        print ser
        dataseries = (df_stgdpaq[ser]).dropna()
        # for i in [1,2,3,4,5]:
        try:
            # differencing(dataseries,i)
            d_order01 = adfullertest(differencing(dataseries,1))
            d_order05 = adfullertest(differencing(dataseries, 5))
            if d_order01[0]<d_order05[0]:
                d_order0 = d_order01
                i =1
            else:
                d_order0 = d_order05
                i =5

            if d_order0[0] < d_order0[4]['10%']:
                val = i
            elif (d_order0[0] > d_order0[4]['10%']) & (d_order0[0] <= d_order0[4]['10%']+0.5):
                val = i*0.1
            elif (d_order0[0] > d_order0[4]['10%']+0.5) & (d_order0[0] <= d_order0[4]['10%']+0.77):
                val = i*0.01
            elif (d_order0[0] > d_order0[4]['10%']+0.77) :
                val = 0

        except ValueError:
            print 'Value Error in:', ser
            val = -99
        except KeyError:
            print 'Key Error in:', ser
            val = -99
        except np.linalg.LinAlgError:
            print 'Empy series LinAlgerror Error in:', ser
            val = -99
        print '\n'
        ls_stdiff.append([st_name, ser, val])

# Index for stationarity_report_tab
#    1 = diff1
#    2 = diff2
#    3 = roll2
#    4 = roll3
#    0 = impossible/NAN
#    5 = cosnider despite imposs
#    2.1 = 10%criticla satisfied for method 2
#    0.2 = imposs but diff 2 is closest ; usually if imp or just near 10% critical value

# Index for stationarity_report_v2_20161128
#    1 simple difference, statioanry with CI=10%
#    5 log difference, statioanry with CI=10%
#    0.1 less statioanry with CI=10% + 0.5
#    0.01 less statioanry with CI=10% + 0.0.77
#    0 not stationary at all
#    -99 no data

#dataseries_ds = (pd.rolling_mean(pd.Series(dataseries), 2).dropna())
# dataseries_ds = (pd.rolling_mean(pd.Series(dataseries), 3).dropna())
#plt.plot(np.log(dataseries** 0.3))


# (b) Granger causality test

# the file Stationarity_report_tab prepared manually in previous part is important for it allows us to find which differencing method to use
# df_st = pd.read_csv(csv_save_path + 'GDP_AQ/' + 'Stationarity_report_tab_v1.csv', header=0)        #stationarity df
# order = {0:0, 1:1, 2:2, 3:1, 4:2, 5:1} # codename:order of stationarity
df_st = pd.read_csv(csv_save_path + 'GDP_AQ/' + 'Stationarity_report_v2_20161128.csv', header=0)        #stationarity df
order = {0:0, 1:1, 2:2, 3:1, 4:2, 5:1} # codename:order of stationarity
diff_key = {0:0, -99:-99, 1:1, 0.1:1, 0.01:1, 5:5, 0.5:5, 0.05:5}
prod_aq = ['sumurbanAOD', 'sumurbanANG', 'sumurbanSO2', 'sumurbanNO2']         # products with which we will compare

# GCT fucntion
def print_gct(ser, dataseries_d1a, prod, dataseries_d1b):
    try:
        print 'Alpha', prod, ' causes ', ser
        alpha1 = ts.grangercausalitytests(np.asarray(zip(dataseries_d1a, dataseries_d1b)), maxlag=2, verbose=True)
        print '/n'
        print 'Beta', ser, ' causes ', prod
        beta1 = ts.grangercausalitytests(np.asarray(zip(dataseries_d1b, dataseries_d1a)), maxlag=2, verbose=True)
        print '/n'
        lg=2        #lg is lag order

    except ValueError:
        print 'only one lag allowed'
        alpha1=NAN
        beta1 = NAN

        try:
            print 'Alpha', prod, ' causes ', ser
            alpha1 = ts.grangercausalitytests(np.asarray(zip(dataseries_d1a, dataseries_d1b)), maxlag=1, verbose=True)
            print '/n'
            print 'Beta', ser, ' causes ', prod
            beta1 = ts.grangercausalitytests(np.asarray(zip(dataseries_d1b, dataseries_d1a)), maxlag=1, verbose=True)
            print '/n'
            lg=1
        except ValueError:
            print 'error in values.. probably all 0000'
            alpha1=NAN
            beta1 = NAN
            lg=0
    return [alpha1, beta1, lg]
#func end

# Running GCT
ls = []
for st_name in df_st['NAME_1']:
    print ' *-*-*-*-*-    ',st_name, ' *-*-*-*-*-    '
    df_stgdpaq = pd.read_csv(csv_save_path + '/GDP_AQ//' + 'df_AQGDPurban_'+st_name+'.csv', header=0)
    df_stgdpaq.rename(index=str, columns=rename_dict, inplace=True)

    for sec in ren_sectors:
        dataseries_s = (df_stgdpaq[sec]).dropna()
        sec_st_type = int(diff_key[df_st[df_st['NAME_1']==st_name][sec].values[0]]) # type of stationarity to be run for sector

        # if sec_st_type == 0 and float(df_st[df_st['NAME_1']==st_name][sec].values)> 0 :     # for important sectors/AQ we have relaxed the stationarity criteria. instead of 0 they are marked with 0.2
        #     sec_st_type = float(df_st[df_st['NAME_1']==st_name][sec].values)*10

        for prod in prod_aq:
            gct_val = 0.0         # 0 : no causality, 1 alpha causality, 2 beta causality, 3 both directions; null = no cause
            gct_lg = 0.0          # lag run

            dataseries_p = (df_stgdpaq[prod]).dropna()
            prod_st_type = int(diff_key[df_st[df_st['NAME_1'] == st_name][prod].values[0]])  # type of stationarity to be run for aq parameter
            if (sec_st_type > 0) & (prod_st_type > 0):  # 0/-99 implies it wasnt found to get stationary/no data
                if order[sec_st_type] == order[prod_st_type]:           # that is both have same order
                    dataseries_sts = differencing(dataseries_s, sec_st_type)  # stationary series for sector
                    dataseries_stp = differencing(dataseries_p, prod_st_type)  # stationary series for product aq
                elif min(sec_st_type, prod_st_type)== sec_st_type:        # sector needs to be differenced once more
                    sec_st_type=sec_st_type+1
                    dataseries_sts = differencing(dataseries_s, sec_st_type)  # stationary series for sector
                    dataseries_stp = differencing(dataseries_p, prod_st_type)  # stationary series for product aq
                elif min(sec_st_type, prod_st_type)== prod_st_type:     # prodaq needs to differenced once more
                    prod_st_type=prod_st_type+1
                    dataseries_sts = differencing(dataseries_s, sec_st_type)  # stationary series for sector
                    dataseries_stp = differencing(dataseries_p, prod_st_type)  # stationary series for product aq

                [alpha1, beta1, lg] = print_gct(sec, dataseries_sts, prod, dataseries_stp)      # alpha --> AQ causes GDP, beta --> GDP causes AQ
                try:
                    if alpha1[1][0]['lrtest'][1]<=0.05 and alpha1[1][0]['ssr_chi2test'][1]<=0.05:       # p test result         # check for Lag =1
                        gct_val = 1.0
                        gct_lg = 1
                        if beta1[1][0]['lrtest'][1]<=0.05 and beta1[1][0]['ssr_chi2test'][1]<=0.05:       # p test result
                            gct_val = 3.0
                    elif beta1[1][0]['lrtest'][1] <= 0.05 and beta1[1][0]['ssr_chi2test'][1] <= 0.05:  # p test result
                        gct_val = 2.0
                        gct_lg = 1

                    #elif lg == 2: need to figure out how include lag port efficiecntly
                    if alpha1[2][0]['lrtest'][1]<=0.05 and alpha1[2][0]['ssr_chi2test'][1]<=0.05:       # p test result       # check for Lag =1
                        gct_val = 1.0
                        gct_lg = 2
                        if beta1[2][0]['lrtest'][1]<=0.05 and beta1[2][0]['ssr_chi2test'][1]<=0.05:       # p test result
                            gct_val = 3.0
                    elif beta1[2][0]['lrtest'][1] <= 0.05 and beta1[2][0]['ssr_chi2test'][1] <= 0.05:  # p test result
                        gct_val = 2.0
                        gct_lg = 2
                except TypeError:
                    gct_val = -99
                    gct_lg = -99

            ls.append([st_name,prod,sec,gct_val,gct_lg])
            df_gct = pd.DataFrame(ls, columns=['NAME_1', 'prod', 'sector', 'gct_val', 'gct_lag'])  # strores regression coefficients for each city
            df_gct['state'] = df_gct['NAME_1']
            df_gct.set_index('state', inplace=True)
            df_gct.to_csv(csv_save_path+'GDP_AQ/' + 'df_GrangerTest_all_v3_20161128.csv')
            for prod in ['ANG', 'AOD', 'SO2', 'NO2']:
                df_gctprod = df_gct[df_gct['prod']=='sumurban'+prod].pivot(index='NAME_1', columns='sector', values='gct_val')
                df_gctprod.to_csv(csv_save_path + 'GDP_AQ/' + 'df_GrangerTest_'+ prod +'_v3_20161128.csv')


# ts.grangercausalitytests(mdata[['realgdp', 'realcons']].view((float,2)), maxlag=3, verbose=True)
# mdata[['realgdp', 'realcons']].view((float,2))

# (c) Plot Granger causality results
# by method - I
def discrete_matshow(data):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('RdBu', np.max(data)-np.min(data)+1)
    ax.grid(False)
    mat = ax.matshow(data, cmap=cmap, vmin =np.min(data)-0.5, vmax = np.max(data)+0.5)
    ax.set_title('Granger Causality Test - ANG')
    # cax = plt.colorbar(mat, ticks = np.arange (np.min(data), np.max(data)+1))

prod = 'SO2'
try:
    df_gctprod = pd.read_csv(csv_save_path + 'GDP_AQ/' + 'df_GrangerTest_'+ prod +'.csv', header=0)
    df_statestat = pd.read_csv(csv_in_path+ 'state_stat.csv', header=0)
    df_gctprod = df_gctprod.fillna(4)
    df_gctprod = df_gctprod[['NAME_1', 'Agro', 'Construct', 'Unreg_Manuf', 'Reg_Manuf', 'Industry',
                             'Transport', 'Rly', 'Utility', 'Trade_hotel', 'Communication',
                             'PublicAdmin', 'RealEstate', 'Bank_Insurance','Other services']]
    df_gctprod1 = pd.merge(df_gctprod, df_statestat, on=['NAME_1'])
    df_gctprod1['sortmetric'] = df_gctprod1['GDP_percapita']#df_gctprod1[('sumurban'+prod)]#/df_gctprod1['Urban']
    df_gctprod1.sort_values(['sortmetric'], inplace=True, ascending=True)
    ar_gctprod = df_gctprod1[['Agro', 'Construct', 'Unreg_Manuf', 'Reg_Manuf', 'Industry',
                             'Transport', 'Rly', 'Utility', 'Trade_hotel', 'Communication',
                             'PublicAdmin', 'RealEstate', 'Bank_Insurance','Other services']].as_matrix()
    discrete_matshow(ar_gctprod)
except ValueError:
    print 'some problem'

# # another method - II
# label = [0,1,2,3,4,5]
# fig, ax = plt.subplots()
# ax.grid(False)
# ax.matshow(ar_gctprod, cmap='Accent', interpolation = 'nearest', alpha=0.5)
# cb = fig.colorbar(ticks = np.array(label)+0.5)
# cb.set_ticklabels(label)
# ax.xaxis.set_ticks(np.arange(0,14,2))
# ax.yaxis.set_ticks(np.arange(0,32,2))
# ax.legend()
# ax.set_title('sdh')
# plt.show()

# another methd III
# Credit http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
fig, ax = plt.subplots()
cmap = plt.get_cmap('RdBu', 4-0+1)
heatmap = ax.pcolor(ar_gctprod, cmap=cmap, alpha=0.8)

# Format
ax.set_title('Granger Causality Test -' + prod)
fig = plt.gcf()
fig.set_size_inches(7, 15)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(ar_gctprod.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(ar_gctprod.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels
labels = ['Agro', 'Construct', 'Unreg_Manuf', 'Reg_Manuf', 'Industry', 'Transport', 'Rly', 'Utility', 'Trade_hotel', 'Communication',
                             'PublicAdmin', 'RealEstate', 'Bank_Insurance','Other services']

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(df_gctprod1.NAME_1, minor=False)

# rotate the
plt.xticks(rotation=90)
plt.tight_layout()
ax.grid(False)

# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  Task V.I Analyze trend for each AQ-GDP  # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Lets now ceck what was the impact of doing stationarity graphically. Also try to relate if after differncing the GCT outcome can be predicted

styles = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'b-.', 'g-.', 'r-.', 'c-.', 'm-.', 'y-.', 'k-.',  'bs-', 'gs-', 'rs-', 'cs-']
df_stationary = pd.DataFrame()
for st_name in df_st['NAME_1']:
    df_stgdpaq = pd.read_csv(csv_save_path + '/GDP_AQ//' + 'df_AQGDPurban_' + st_name + '.csv', header=0)
    df_stgdpaq.rename(index=str, columns=rename_dict, inplace=True)
    df_stgdpaq.index=df_stgdpaq['year']

    # Scale all variables ebtween 0 to 1 by normalize
    df_norm = df_stgdpaq[ren_series]/(df_stgdpaq[ren_series]**2.0).sum()**.5       # basically same as preprocessing.normalize(); unit norm

    # 1. Original non-stationary series plot
    # fig = df_norm.plot(title='All GDP sector and AQ - ' + st_name, style=styles).get_figure()
    # fig.set_size_inches(15,10)
    # fig.savefig('Trend_' + st_name+'.png')


    # 2. Stationary series trend
    # create stationary series set
    df = pd.DataFrame()
    for sec in ren_series:
        dataseries_sts = 0
        dataseries_s = (df_norm[sec]).dropna()
        sec_st_type = int(diff_key[df_st[df_st['NAME_1']==st_name][sec].values[0]]) # type of stationarity to be run for sector
        if (sec_st_type > 0):  # 0/-99 implies it wasnt found to get stationary/no data
            dataseries_sts = differencing(dataseries_s, sec_st_type)  # stationary series for sector
        df[sec]= dataseries_sts
    df['NAME_1'] = st_name

    # Stationary series plot
    # fig = df.plot(title='Stationary All GDP sector and AQ - ' + st_name, style=styles).get_figure()
    # fig.set_size_inches(15,10)
    # fig.savefig('Stationary_' + st_name+ '.png')
    # df.plot(title='Stationary All GDP sector and AQ - ' + st_name, style=styles)
    df_stationary = pd.concat([df_stationary, df], ignore_index=True)
    df_stationary.to_csv(csv_save_path + 'df_allstationary.csv',  index=False, header=False )



















# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  Task VI Normalize GDP each state sector and find PCA i All, ii GCT  # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Reading GDP data with urban AQ

type = 'urban'

ls_stdiff = []      # key for holding the preferred differencing method to make each causal sector staitonary

# Reading the csv with all state names
df_staq = pd.read_csv(csv_save_path + '/GDP_AQ/' + 'df_StateAQ_' + type + '.csv', header=0)

prod = 'AOD'

#for st_name in df_staq['NAME_1']:

st_name = 'Tamil Nadu'
for prod in ['AOD', 'ANG', 'NO2', 'SO2']:
    # Read csv with state AQ and GDP values
    # df_stgdp = pd.read_csv(csv_save_path + 'GDP_AQ/df_' + st_name + '.csv', header=0)  # state gdp+AQ / obtained by AQ_GDP TaskII
    df_stgdpaq = pd.read_csv(csv_save_path + '/GDP_AQ//' + 'df_AQGDPurban_' + st_name + '.csv', header=0)
    df_stgdpaq.rename(index=str, columns=rename_dict, inplace=True)

    # Read the csv with Granger test results
    df_gctprod = pd.read_csv(csv_save_path + '/GDP_AQ/' + 'df_GrangerTest_'+ prod +'.csv', header=0)
    df_gctprod = df_gctprod.fillna(4)

    # Keep only those sectors with are identified as GCT cause (code =2)
    ls_cause = []
    df_cause = (df_gctprod[df_gctprod['NAME_1']== st_name] == 2)
    for sec in df_cause.columns:
        if df_cause[sec].bool() == True:
           print sec
           ls_cause.append(sec)
    ls_cause.append('sumurban'+prod)
    # Re-check for stationarity

    for ser in ls_cause:
        print ser
        dataseries = (df_stgdpaq[ser]).dropna()
        for i in [1, 3, 5]:
            try:
                print ser, ' ', i
                adfullertest(differencing(dataseries, i))
            except ValueError:
                print 'Error:', ser

            print '\n'
        val = raw_input('Which is the preferred method?')
        ls_stdiff.append([st_name, prod, ser, val])
        print ls_stdiff



# standardize causalGDP sectors
stdGDP = preprocessing.StandardScaler().fit_transform(df_stgdpaq[ls_cause])
# df_normgdp = df_stgdpaq[ren_sectors].divide(df_stgdpaq[ren_sectors].sum(axis=0), axis=1)

# eigendecomposition on covariance matrix
# cov_mat = np.cov(stdGDP.T)
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# for i in eig_pairs:
#     print (i[0])

# PCA from scikit
pca = PCA(n_components=2)
pcaGDP = pca.fit_transform(stdGDP)
print pca.explained_variance_ratio_

# formulate the regression equation









