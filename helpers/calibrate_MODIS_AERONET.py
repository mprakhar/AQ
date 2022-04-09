# __author__ = 'Prakhar'
# Created 7/30/2016
# Last edit 7/30/2016

#Purpose: (1) to calibrate first AEROENT and MODIS at pixel level and find their correlation and slope.
#         (2) to generate a MODIS image for whole of india based on pixel level calibrations developed

#Output expected:


#Terminology used:


import zipfile
import os.path
import gdal
import numpy
from PIL import Image
import pandas as pd
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
import csv
import matplotlib as plt
from glob import glob
import seaborn as sns
import scipy.stats as stats

# Pvt imports
import classRaster
from classRaster import Raster
from classRaster import Raster_file
import infoFinder as info
import my_plot as mpt
import coord_translate as ct



# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

gd_path = gb_path + r'Data/Data_process/AQ/AERONET//' #ground data
gd_info = gd_path + 'AERONET_station_info.csv'

aod = Raster_file()
aod.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/MOD04L2/L3//'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = aod.path + 'MOD04L2.A2015308.AOD.Global'
aod.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//MODIS_georef.tif'

ang = Raster_file()
ang.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/MOD04L2/L3//'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = ang.path + 'MOD04L2.A2015308.ANG.Global'
ang.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//MODIS_georef.tif'

no2 = Raster_file()
no2.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/OMI/L2G//'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = no2.path + 'OMI.NO2.20050203.Global'
no2.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//OMI_georef.tif'

# Output location
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'


# Trial area ~~~
# To unzip a file
# a= zipfile.ZipFile('20010101_20160530_Kanpur.zip')
# a.extractall('/home/prakhar/Research/AQM_research/Codes/')


# Get list all station names and  corresponding start and ending date + checking if all files are readable
file_list = info.filenamefinder(gd_path, '', '', '')
for file in file_list:
    file_name = os.path.split(file)[1]
    start = file_name[0:8]
    end = file_name[9:17]
    station = file_name[18:]
    print start, ',', end, ',', station
    csv_name = file+r'//'+file_name+'.csv'
    try:
        pd.read_csv(csv_name, header=6)
    except Exception as e:
        print 'still error'
        print e.message




# Trial area ~~~
# tif_file = gb_path + r'/Data/Data_process/Meanannual_MODIS/MODISANG2001.tif'
# ct.latLonToPixel(tif_file,[[28.630,77.175]])           # gives output as 144, 107. This corresponds to ENVI's (145,108) for python array read the pixel at (107,144)
# im = Image.open(tif_file)
# imarray = numpy.array(im)
# imarray(107,144)

# Reading the csv gd_info having info on all ground stations
df_gd = pd.read_csv(gd_info, header=0)
df_gd.index=df_gd.station
# Adding info about local image pixel coordinates



# Variables to be extracted from AERONET LEV15 csv
# Day_of_Year(Fraction)
# AOD_500nm
# AOD_440nm
# 440-870_Angstorm_Exponent
# NO2(Dobson)

# Finding earliest start date
dt = min(df_gd['stdt'])
start_date = date(int(str(dt)[0:4]), int(str(dt)[4:6]), int(str(dt)[6:8]))
dt = max(df_gd['eddt'])
end_date = date(int(str(dt)[0:4]), int(str(dt)[4:6]), int(str(dt)[6:8]))

start_date= date(2015, 8, 5)
end_date= date(2008, 7, 15)



# 1. Creating files with raw info from satellite and AERONET data within specific date-time bounds for each station

for prodT in [aod, no2]:
    # prodT = no2
    # Finding local locations of stations as per the input image used for prodT
    df_gd['pix_x'] = 0
    df_gd['pix_y'] = 0
    for city in df_gd['station']:
        df_gd.set_value(city, 'pix_x', ct.latLonToPixel(prodT.sample, [[ float(df_gd[df_gd['station']==city]['lat']), float(df_gd[df_gd['station']==city]['lon'])]])[0][1])
        df_gd.set_value(city, 'pix_y', ct.latLonToPixel(prodT.sample, [[float(df_gd[df_gd['station']==city]['lat']), float(df_gd[df_gd['station']==city]['lon'])]])[0][0])

    # Iterating over dates and opning corresponding prodT files. also checking if any station has equivalent ground data
    for date_c in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):     #date_c is date counter between min and amx dates
        for prodT.file in glob(os.path.join(prodT.path, '*.'+prodT.prod+'*'+'.Global')) :       # file extension ext is '' therefore, reading all corresponding prodT imeages
            date_y_d = map(int, prodT.yeardayfinder() )     #date as a list with integer year and daycount as return
            if (date_y_d == [date_c.year,date_c.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                prod_arr = prodT.raster_as_array()
                #print 'processing  ' + input_value_raster

                for city in df_gd['station']:

                    stn_file = str(df_gd.get_value(city, 'stdt')) + '_' + str(df_gd.get_value(city, 'eddt')) + '_' + str.strip(city)

                    # 1. reading from AERONET files
                    # Extract date/time relevant aeronet data.
                    df_stn = pd.read_csv(gb_path+r'Data/Data_process/AQ/AERONET/'+stn_file+'/'+stn_file+'.csv', header=6)
                    df_stn = df_stn.rename(
                        columns={'440-870_Angstrom_Exponent': 'AER_ANG440_870', 'Day_of_Year(Fraction)': 'doyf',
                                 'AOD_500nm': 'AER_AOD500', 'AOD_440nm': 'AER_AOD440', 'NO2(Dobson)': 'AER_NO2',
                                 'Date(dd-mm-yyyy)': 'dmy'})
                    df_stn2 = df_stn[['dmy', 'doyf', 'AER_AOD500', 'AER_AOD440', 'AER_ANG440_870', 'AER_NO2']]


                    # MODIS(Terra) comes at 10:30 IST, convert it to GMT and a buffer of 1.5hrs both ways.
                    # OMI(Aura) comes at 13:30 IST
                    if prodT.sat in 'MODIS':
                        df_stn3 = df_stn2[(df_stn2['dmy'] == date_c.strftime('%d:%m:%Y')) & (df_stn2['doyf'] % 1.0 > 4.3 / 24.0) & (df_stn2['doyf'] % 1.0 < 5.7 / 24.0)]
                    if prodT.sat in 'OMI':
                        df_stn3 = df_stn2[(df_stn2['dmy'] == date_c.strftime('%d:%m:%Y')) & (df_stn2['doyf'] % 1.0 > 7.3 / 24.0) & (df_stn2['doyf'] % 1.0 < 8.7 / 24.0)]

                    if (len(df_stn3.doyf) > 0 ):
                        if prodT.prod == 'AOD':
                            df_stn3 = df_stn3[['dmy', 'doyf', 'AER_AOD500', 'AER_AOD440', 'AER_ANG440_870']]
                        #if prodT.prod == 'ANG':
                        #    df_stn3 = df_stn3[['dmy', 'doyf','AER_ANG440_870']]
                        if prodT.prod == 'NO2':
                            df_stn3 = df_stn3[['dmy', 'doyf', 'AER_NO2']]

                        # 2. reading form MODIS/OMI files
                        df_stn3[prodT.sat + prodT.prod] = prod_arr[df_gd.get_value(city, 'pix_x'), df_gd.get_value(city, 'pix_y')]      # value from satellite imagery
                        fname = csv_save_path + 'df_CalibAER' + '_' + prodT.prod  + '_' +  city + '.csv'

                        # Checking if the cooresponding csv file fror city exists or not and creating it if not
                        if os.path.isfile(fname) == False:
                            df_stn3.to_csv(fname, index=False, header=True)
                            print 'opening csv ' + city
                            print 'processing  ' + prodT.file

                        else:
                            print 'appending ' + city
                            print 'processing  ' + prodT.file
                            with open(fname, 'a') as f:
                                df_stn3.to_csv(f, index=False, header=False)



# to diplay mpt.histeq(prod_arr)


# 2. Using the files created in previous step for finding and storing station wise correlations

# for prodT in [aod, no2]:
#     prodT = aod
#     for city in df_gd['station']:
#         fname = csv_save_path + 'df_CalibAER' + '_' + prodT.prod + '_' + city + '.csv'
#         df_calibaer = pd.DataFrame([city], columns=['station'])
#
#         df_csv = pd.read_csv(fname, header=0)  # each station's aq data from satellite and aeronet
#         df_csv = df_csv.fillna({
#             'AER_AOD500': -999,
#             'AER_AOD440': -999,
#             'AER_ANG440_870': -999,
#             'MODISAOD' : 0
#         })
#
#         df_calibaer_mean = df_csv.groupby(['dmy'])['AER_AOD500', 'AER_AOD440', 'AER_ANG440_870', 'MODISAOD'].mean()       # this mean is at date level only
#         #First check if any of the values of AOD500 are -999. if yes then use AOD440 for calculation
#         df_calibaer_mean['AERAOD'] = df_calibaer_mean['AER_AOD500']*(550/500)**(-1*df_calibaer_mean['AER_ANG440_870'])     #http://www.atmos-chem-phys.net/14/593/2014/acp-14-593-2014.pdf,
#         df_calibaer_std = df_calibaer_mean.std().to_frame().transpose()        #this std is the total std for that station
#         df_calibaer_std.rename(columns={'AERAOD': 'AERAOD', 'MODISAOD': 'MODISAODstd'}, inplace=True)
#
#         cov_param = df_calibaer_mean['AERAOD'].cov(df_calibaer_mean['MODISAOD'])
#         reg_param = numpy.polyfit(df_calibaer_mean['AERAOD'], df_calibaer_mean['MODISAOD'], 1)        # x,y,1;l output is[m,c]; y= mx+c
#         df_calibaer['n'] =  len(df_calibaer_mean)  #number of data counts
#         df_calibaer['cov'] = cov_param      #covariance b/w AERONET and satellite measurements
#         df_calibaer['reg_m'] = reg_param[0]     #slope of regression
#         df_calibaer['reg_c'] = reg_param[1]     #cosntant of regression
#         df_calibaer.join(df_calibaer_std)
#
#
#         # also run for no2 at night; no use as OMI and aeronet extract at different bandwidth
#
#         georef AIRS/download AIRS tonight
#
#         #cleaning up missing values

#----------------------------

# estimate AERONET curve


import os, sys, fnmatch
from glob import glob
import pandas as pd

path1 = gb_path + r'Codes/CSVOut/AERONET_Satellite calibration/AOD//'
filename = glob(os.path.join(path1, '*' + '.csv'))
ls = []
for file in filename:
    df_calibaero = pd.read_csv(file, header=0)
    df_grouped = df_calibaero.groupby('dmy').mean()
    df_grouped['AOD550'] = df_grouped.AER_AOD440 * (440.0/550.0)**df_grouped.AER_ANG440_870
    df_grouped['MODISAOD'] = df_grouped['MODISAOD']/1000
    x = df_grouped[(df_grouped.MODISAOD>0)&(df_grouped.AER_AOD440>0)]['AOD550']
    y = df_grouped[(df_grouped.MODISAOD>0)&(df_grouped.AER_AOD440>0)]['MODISAOD']
    # df_grouped['OMINO2'] = df_grouped['OMINO2'] / 1000
    # x = df_grouped[(df_grouped.OMINO2 > 0) & (df_grouped.AER_NO2 > 0)].AER_NO2
    # y = df_grouped[(df_grouped.OMINO2 > 0) & (df_grouped.AER_NO2 > 0)].OMINO2
    n = len(x)
    city = file[101:-4]

    try:
        [m, b, r_value, p_value, std_err] = stats.linregress(x, y)
    except ValueError:
        print 'empty ' + city

    ls.append([city, n, m, b, r_value, p_value, std_err])
    df_cityGDP_reg = pd.DataFrame(ls, columns=['city', 'count', 'm', 'b', 'rvalue', 'pvalue', 'stderr'])  # strores regression coefficients for each city
    df_cityGDP_reg.to_csv('df_Calibaer_AODreg.csv')
    # df_cityGDP_reg.to_csv('df_Calibaer_NO2reg.csv')

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ko', label='R$^2$ value = ' + str('%.2f' % r_value), alpha=.8)
    ax.plot(x, map(lambda z: z * m + b, x), 'r--', label='Best fit: y=' + str('%.2f' % m) + 'x+' + str('%.2f' % b), alpha=.8)
    # plt.plot(x, func(x, *popt), 'r-', label='line fit')
    try:
        plt.xlim([0, max(max(x), max(y))+0.2])
        plt.ylim([0, max(max(x), max(y))+0.2])
    except ValueError:
        print ''
    ax.legend(loc=4)
    plt.xlabel('AOD (AERONET at 550nm)', fontsize=16)
    plt.ylabel('AOD (MODIS)', fontsize=16)
    plt.title('AOD comparison for '+city, fontsize=20)

    plt.show()
    plt.savefig(plt_save_path + 'AERONET_Satellite_calibration/' + 'Calibaer_AOD_'+city+'.png')
    # plt.xlabel('NO$_2$ AERONET (DU)', fontsize=16)
    # plt.ylabel('NO$_2$ OMI (DU)', fontsize=16)
    # plt.title('NO$_2$ comparison for ' + city, fontsize=20)
    # # plt.show()
    # plt.savefig('Calibaer_NO2_' + city + '.png')

# -----



df_stn = pd.read_csv(r'/home/prakhar/Research/AQM_research//Data/Data_process/AQ/AERONET/20010101_20160530_Kanpur/20010101_20160530_Kanpur.lev15', header=6)
os.path.isfile(gb_path+r'/home/prakhar/Research/AQM_research//Data/Data_process/AQ/AERONET/20010101_20160530_Kanpur/20010101_20160530_Kanpur.lev15')
df_stn = df_stn.rename(columns={'440-870_Angstrom_Exponent': 'ANG440_870', 'Day_of_Year(Fraction)': 'doyf', 'AOD_500nm':'AOD500', 'AOD_440nm':'AOD440','NO2(Dobson)': 'NO2', 'Date(dd-mm-yyyy)':'dmy' })
df_stn2 = df_stn['dmy', 'doyf', 'AOD500', 'AOD440', 'ANG440_870', 'NO2']