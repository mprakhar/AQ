#__author__ = 'Prakhar'
# Created 3/26/2016
#Last edit 8/18/2016

# Purpose: (1) For a given pair of images with different resolutions, this will find a regression equation between the DN values of two images. It will then be used for calibration so as to continue legacy from one sensor to another. This function
#           also first resamples the images at same resolution to be able to compare the pixels
#          (2) Perform interannual claibration for DMSP sensors based on  Wu et al (Intercalibration of DMSP-OLS night-time light data by the invariant region method)
#          (3) Check the accuracy of calibration equations developed in (1) by comparing TNL at district level

#Dependencies:
#OLS_intercalibration_coeff_Wu.csv
# Location of output: E:\Acads\Research\AQM\Data process\CSVOut





import numpy as np
from numpy import *
import csv
import os
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
from scipy import stats
from glob import glob
import pylab

#Pvt imports
import shp_rstr_stat as srs
import pandas as pd
import my_plot as mpt
import classRaster
from classRaster import Raster_file
from classRaster import Image_arr
import infoFinder as info


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
satS = 'VIIRS'       # options VIIRS
prodS = 'DNB'        # options DNB refers to radiance values obtained from DN values
# ImageT: target reolsution to be achieved, ImageS: source image to be resampled
raster_pathT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\\'
input_value_rasterT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\F182013.v4c_web.stable_lights.avg_vis.tif'
raster_pathS=r'D:\2013 beta\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif'
input_value_rasterS= r'D:\2013 beta\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif'
csv_in_path = gb_path+r'/Codes/CSVIn//'


#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'


# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * *     Task I  DMSP-VIIRS 2013 LUT generation   *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#


# first resmapling the higher resolution image to the level of lower resolution image
pix_resample(input_zone_polygon, satS, prodS, input_value_rasterS, satT, prodT, input_value_rasterT, dor, ver)
# because the list and tuple arrangement has not been fixed, no \w manually remove " ' ( ) from the csv so created

# ------------------------------------------------------   Visualize scatter plots b/w two sat images     ------------------------------------------------------
ver1 = 'v1'          #version number of output
dor1 = '20160325'    #date of run

dfT = pd.read_csv(csv_save_path+'df_PixVal_'+satT+prodT+'_'+dor1+ver1+'.csv', header=0)
dfT=dfT.transpose()
dfT.columns=dfT.iloc[0]
dfTT=dfT.convert_objects(convert_numeric=True)
# understand the structure of data you have created, because you get and error Expected list, got tuple. correct and uncomment
dfTT.to_csv(csv_save_path+'df_PixVal0_'+satT+prodT+'_'+dor+ver+'.csv', index=False, header=True) # making 0 file. this is like a backup and also can be read more quickly as it is already transposed
# perform some cleaning in the csv file created. udnerstanding of making df, list, tuple is still not perfectly clear
dfT0 = pd.read_csv(csv_save_path+'df_PixVal0_'+satT+prodT+'_'+dor1+ver1+'.csv', header=0)
# iN case of OLS raw DN have to be converted to radiance values


# image R: the resampled image
dfR = pd.read_csv(csv_save_path+'df_PixVal_Resampl'+satT+'_'+satS+prodS+'_'+dor1 + ver1+'.csv', header=0)
dfR=dfR.transpose()
dfR.columns=dfR.iloc[0]
dfRR=dfR.convert_objects(convert_numeric=True)
dfRR.to_csv(csv_save_path+'df_PixVal_Resampl0'+satT+'_'+satS+prodS+'_'+dor1 + ver1+'.csv', index=False, header=True)
dfR0 = pd.read_csv(csv_save_path+'df_PixVal_Resampl0'+satT+'_'+satS+prodS+'_'+dor1 + ver1+'.csv', header=0)

dfTT=dfT0 #OLS
dfRR=dfR0 #DNB
prodTY=prodT+YEARM #OLS2013
prodRY=prodS+YEARM #DNB2013

dfTT['rad2013'] = dfTT[prodTY]**(3.0/2.0) #radis the radiacne values

y_label_text= {'ANG': '','AOD':'', 'SO2': '(DU)', 'NO2': '(10E13 molecules/cm2)', 'DNB': '( 10E(-13)nano-Watts/cm2 sr )', }

mpt.df2Plot2D(df1=dfTT, Xaxis=('rad'+YEARM), df2=dfRR, Yaxis=prodS+YEARM, subt='Pixel by pixel comaprison for '+YEARM, Xlabel=prodT+y_label_text[prodS], Ylabel=prodS+y_label_text[prodS], save_path=plt_save_path+prodS+prodT+YEARM+'.jpg', save=1)




# the following analysis is excusively for calibration between OLS and DNB based on 2013 images
# the following curve fitting gives a very poor regression and intercept.
polyfit(dfTT['rad2013'][1:4032154],dfRR[prodRY][1:4032154], deg=1)
# henxce we need to first clean the data of outliers and then run it again
# to clean the  data of outliers, I. sort DNB2013 on the basis of OLS2013. then for each OLS2013 group find the mean, median , std of DNB2013. for these 64 values then we can then make a plot using excel or whatever

dfM=dfTT.copy(deep=True) #creating a a new dataframe with merged columns from DFRR and dfTT
dfM[prodRY]=dfRR[prodRY]
dfG[prodRY].describe()
dfM.groupby(prodTY).mean()
dfN[prodRY+'_mean']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.mean()) #dfN = Navigate into unknown :)
dfN[prodRY+'_med']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.median())
dfN[prodRY+'_min']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.min())
dfN[prodRY+'_max']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.max())
dfN[prodRY+'_std']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.std())
dfN['rad2013']=pd.DataFrame(dfM.groupby(prodTY).rad2013.mean())
dfN[prodRY+'_count']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.count())
dfN.to_csv(csv_save_path+'df_LUT_stats_'+prodT+prodS+'_'+dor+ver+'.csv', index=True, header=True)    #generating a kind of LUT for calibration from OLS to DNB






# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * *  Task II DMSP Interannual Calibration  *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# Type I: Wu et al's region invariant method

#since DMSP  OLS havent beein sensor calibrated, I am using the method of "Intercalibration of DMSP-OLS night-time light data by the invariant region method" by Wu et al for
#calibration. the regression function is of the power function form; every image thus gets calibrate to 2012 image
# DNc +1 = a*(DNo + 1)^b
# Where,
# DNc is calibrated DN (NOTE: needs to be converted to radiance if any operation such as rsampling or averaging of pixels is to take place
# DNo is original DN
# a and b are coefficient for each year as mentioned below http://www.tandfonline.com/action/downloadTable?id=T0002&doi=10.1080%2F01431161.2013.820365&downloadType=CSV

georef_raster = "E:\Acads\Research\AQM research\Data\Data process\Georef_img\DMSP_georef.tif" #reference image that has the georef info
raster_path = file_path['DMSP']  # T - how do you want the resolution of S to end up as
# input_value_raster = raster_path + 'F162007.v4b.avg_lights_x_pct\F162007.v4b.avg_lights_x_pct.tif'
# input_value_raster = raster_path + 'F182012.v4\F182012.v4c_web.stable_lights.avg_vis.tif'
DMSP_sensor = [(1999, 'F14'), (2000, 'F15'), (2001, 'F15'), (2002, 'F15'), (2003, 'F16'), (2004, 'F16'), (2005, 'F16'), (2006, 'F16'), (2007, 'F16'),
               (2008, 'F16'), (2009, 'F16'), (2010, 'F18'),(2011, 'F18'), (2012, 'F18')]
dfOLSc = pd.read_csv('OLS_intercalibration_coeff_Wu.csv', header=0)  #OLSc = OLS calibration. this file calibrates to 2012
for i in DMSP_sensor:
    yearm = str(i[0])
    satcode = i[1]

    # a = float(dfOLSc[(dfOLSc.year==int(yearm)) & (dfOLSc.sat==satcode)]['a'])
    # b = float(dfOLSc[(dfOLSc.year==int(yearm)) & (dfOLSc.sat==satcode)]['b'])
    # input_value_raster = raster_path + satcode+yearm+'.v4b.avg_lights_x_pct\\'+satcode+yearm+'.v4b.avg_lights_x_pct.tif'
    if (int(yearm)>2010):
        a=0.8114
        b=1.0849
        input_value_raster = raster_path + satcode+yearm+'.v4\\'+satcode+yearm+'.v4c_web.stable_lights.avg_vis.tif'
    elif (int(yearm)>=2008):
        input_value_raster = raster_path + satcode + yearm + '.v4\\' + satcode + yearm + '.v4c_web.stable_lights.avg_vis.tif'
    imgarray, datamask = srs.zone_mask(input_zone_polygon, input_value_raster)


    # mpt.histeq(imgarray)


    # calibration step
    imgarray_c = a*power(imgarray+1 , b)

    #georef step
    srs.arr_to_raster(imgarray_c, georef_raster, satcode+str(yearm)+".tif")


#-----------

# Type II: Zhang, Pandey Ridge regression method


#since DMSP  OLS havent beein sensor calibrated, I am using the method of "A robust method to generate a consistent time series from dmsp ols nighttime light data" by qingling zhang, bhartedu pandey
#calibration. the regression function is of the power function form; every image thus gets calibrate to 2000 image
# DNc = aDNo + bDNo**2 + c
# Where,
# DNc is calibrated DN (NOTE: needs to be converted to radiance if any operation such as rsampling or averaging of pixels is to take place
# DNo is original DN
# a and b, c are coefficient for each year as mentioned in the paper "A robust method to generate a consistent time series from dmsp ols nighttime light data"

georef_raster = gb_path + '/Data/Data_process/Georef_img//DMSP_georef.tif'       #reference image that has the georef info
raster_path = file_path['DMSP']  # T - how do you want the resolution of S to end up as
DMSP_sensor = [(2001, 'F15'), (2002, 'F15'), (2003, 'F15'), (2004, 'F15'), (2005, 'F15'), (2006, 'F15'), (2007, 'F15'),
               (2008, 'F16'), (2009, 'F16'), (2010, 'F18'),(2011, 'F18'), (2012, 'F18'),(2013, 'F18') ]
dfOLSc = pd.read_csv('OLS_intercalibration_coeff_Pandey.csv', header=0)  #OLSc = OLS calibration. this file calibrates to 2012
for i in DMSP_sensor:
    yearm = str(i[0])
    satcode = i[1]

    # input_value_raster = raster_path + satcode+yearm+'.v4b.avg_lights_x_pct\\'+satcode+yearm+'.v4b.avg_lights_x_pct.tif'
    if (int(yearm)==2013):
        a=0.355542
        b=0.007962
        c=3.866698
        # input_value_raster = raster_path + satcode+yearm+'.v4\\'+satcode+yearm+'.v4c_web.stable_lights.avg_vis.tif'
    else:
        a = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['a'])
        b = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['b'])
        c = float(dfOLSc[(dfOLSc.year == int(yearm)) & (dfOLSc.sat == satcode)]['c'])

    input_value_raster = glob(os.path.join(raster_path, satcode + yearm + '*' + '.tif'))[0]
    imgarray, datamask = srs.zone_mask(input_zone_polygon, input_value_raster)
    # mpt.histeq(imgarray)

    # calibration step
    imgarray_c = a*imgarray +b*power(imgarray, 2) + c

    #georef step
    srs.arr_to_raster(imgarray_c, georef_raster, satcode+str(yearm)+".tif")


#-----------

# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * *  Task III OLS-DNB 2013 calibration regression function verification  *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

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

dnb = Raster_file()
dnb.path = r'/home/prakhar/Research/AQM_research/Data/Data_raw/VIIRS Composite/75N060E//'
dnb.sat = 'VIIRS'
dnb.prod = 'DNB'
dnb.sample = dnb.path + 'SVDNB_npp_20140201-20140228_75N060E_vcmslcfg_v10_c201507201053.avg_rade9.tif'
dnb.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//VIIRS_georef.tif'

input_zone_polygon_0 = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
input_zone_polygon_3 = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm3.shp'

dnb.file = dnb.path + 'BETA_npp_d_20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif'
ols.file = ols.path +  'F182013.v4/F182013.v4c_web.stable_lights.avg_vis.tif'

if os.path.isfile(ols.file) & os.path.isfile(dnb.file):
    print 'both files present'

# 1. First resmapling the higher resolution image to the level of lower resolution image
prodS = dnb
prodT = ols

imgarrayS, datamaskS = prodS.zone_mask(input_zone_polygon_0)  # actually no use of datamask in this step
imgarrayT, datamaskT = prodT.zone_mask(input_zone_polygon_0)
yfactor = (float((np.shape(imgarrayS))[0])) / (float((np.shape(imgarrayT))[0]))
xfactor = (float((np.shape(imgarrayS))[1])) / (float((np.shape(imgarrayT))[1]))
print "imgarray S resolution is", float((np.shape(imgarrayS))[0]), float((np.shape(imgarrayS))[1]), "imgarray T resolution is", float((np.shape(imgarrayT))[0]), float((np.shape(imgarrayT))[1])

imgarrayR = block_reduce(imgarrayS.data, block_size=(int(np.ceil(yfactor)), int(np.ceil(xfactor))), func=np.mean)  # resampled image R ----- Imp Step
imgarrayR=(imgarrayR[~imgarrayR.mask])
imgarrayR = np.ma.masked_array(imgarrayR, np.logical_not( datamaskT))  # necessary to apply the mask again because of the blockreduce the mask gets removed.
dnb_array = Image_arr(imgarrayR)
dnb_array.georef = prodT.georef #       Very Important--> since now DNB image has been resampled at OLS level, use the OLS georef file
dnb_array.arr_to_raster("ResamplDNB_OLS_2013.tif")


# 2. Creating two types of OLS radiance images for calibration

#      *1)  DNB=0 if OLSrad = 0
#           DNB=0.3205397924*exp(0.0704915218*OLSrad)
#      *2)  DNB = OLSrad*0.1159872231 (OLSrad<262.53)
#           DNB = 4.5*exp(0.0612148962*OLSrad) (OLSrad>=262.53)

# Converting DN to radiance
ols_arr = ols.zone_mask(input_zone_polygon_0)[0]
olsrad_arr = power(np.array(ols_arr, dtype=float), 3.0/2.0)
ols_arrobj = Image_arr(olsrad_arr)
ols_arrobj.georef = ols.georef
ols_arrobj.arr_to_raster('RadOLS2013.tif')

# finding fitting curve for OLSrad and DNB 2013
def func(x, a, b, c):
    #return a*x**2 + b*x + c        # quad function
    return 0.000008*a * np.exp(-0.0005*b * x) + 1.5*c       #exponential fucntion

lut=pd.read_csv(csv_save_path + 'df_LUT_2013OLSDNB.csv', header=0)

# Performing calibration by parts

# a) Exponential part
y = lut.DNB2013_mean_corrected[47:] # we first make exponential fitting using 47*. but later to trend fit only the acurate values, we take linear until 54 and exponetial 54:
x = lut.rad2013[47:]

popt, pcov = curve_fit(func, x, y)
a,b,c=popt #([  8.92266619e-05,  -9.34595854e-03,   1.28252397e+00]) -- for quadratic
plt.figure()
plt.plot(x, y, 'ko', label="DNB & OLSrad")
#l2=('Fitted curve : ' +str('%09f'%a)+'x$^2$+'+str('%09f'%b)+'x+'+str('%09f'%c)) # for quadratic eqn
l2 = ('Fitted curve : ' +str((0.000008*a))+'e('+str('%05f'%(-0.0005*b) )+'x)+'+str('%05f'% (1.5*c))) # for quadratic eqn
plt.plot(x, func(x, *popt), 'r-', label=l2)
s1=lut.DNB2013_std.apply(int)
plt.scatter(x, y,s = s1*10, color = 'black', alpha = 0.5, label = 'Standard deviation in DNB')
plt.legend()
plt.show()

# b) plotting by parts
DN_brkpt = 57
bp = DN_brkpt -4
y = lut.DNB2013_mean_corrected[0:]
x = lut.rad2013[0:]
plt.hold(True)
plt.figure()
plt.plot(x, y, 'ko', label="DNB & OLSrad")

#l2=('Quadratic curve : ' +str('%09f'%a)+'x$^2$+'+str('%09f'%b)+'x+'+str('%09f'%c)) # for quadratic eqn
l2 = ('Exponential fit: ' +str((0.000008*a))+'e('+str('%0.2f'%(-0.0005*b) )+'x)+'+str('%0.2f'% (1.5*c))) # for quadratic eqn
axes = plt.gca()
axes.set_ylim([0,80])
plt.plot(x[bp:], func(x[bp:], *popt), 'r-', label=l2) #Important. 53 corresponds to DN=57       # Exponential portion
plt.plot(x[0:bp+1], y[bp+1]/x[bp+1]*x[0:bp+1], 'b-', label='Linear fit: y='+str('%0.2f'%(y[54]/x[54]))+'x')       # Linear portion
plt.legend()
s1=lut.DNB2013_std.apply(int)
# plt.scatter(x, y,s = s1*1000, color = 'black', alpha = 0.2, label = 'Standard deviation in DNB')
plt.errorbar(x,y,s1, linestyle='None', ecolor='orange',  alpha=0.7, label = 'Error bar')
plt.xlabel('OLS 2013 ($Watt/cm^2/sr$)', fontsize=18)
plt.ylabel('DNB 2013 ($Watt/cm^2/sr$)', fontsize=18)
plt.title('Calibration of DNB and OLS radiance images', fontsize=20)

plt.show()


# Summary:
# breakpoint is x=430
# exponential 53: a = 0.009178868005602062, b = -77.433852827324472, c= 6.4071776523406472
# linear 0:53: (0.0,0.0) - (10.710000000000001, 430.33999999999997) --> y=10/430x

# checking the R sq. in adjusted fitting curve
# y_c = a*x**2 + b*x + c        # Quad

y_c=[]
for xl in x: y_c.append(0.000008 * a * np.exp(-0.0005 * b * xl) + 1.5 * c if xl > 431.0 else  y[54] / x[54] * xl)
m,b,r_value,p_value,std_err=stats.linregress(y,y_c) # r_value(x,y)=0.89688038458600183--just lienar regression; r_value(y_c,y) = 0.9513256160364868--quadratic regression;

plt.figure()
plt.plot(y_c, y, 'ko', label="R$^2$ value = "+str('%2f'%(r_value)))
plt.plot(range(0,int(max(y))), range(0,int(max(y))), 'b.', label='Identity line' )
# plt.plot(x, func(x, *popt), 'r-', label='line fit')
plt.legend()
plt.xlabel('DNB radiance (2013)', fontsize=16)
plt.ylabel('Calibrated OLS radiance (2013)', fontsize=16)
plt.title('Calibration of DNB and OLS radiance images', fontsize=20)
plt.show()

#Calibration by order two curve
olsrad_arr1 = copy.deepcopy(olsrad_arr)
a = 0.009178868005602062
b = -77.433852827324472
c= 6.4071776523406472
olsrad_arr1[olsrad_arr1<=430] = (10.710000000000001/430.33999999999997)*olsrad_arr1[olsrad_arr1<=430]
olsrad_arr1[olsrad_arr1>430] = 0.000008*a*np.exp(-0.0005*b*olsrad_arr1[olsrad_arr1>430])+1.5*c

ols_arrobj = Image_arr(olsrad_arr1)
ols_arrobj.georef = ols.georef
ols_arrobj.arr_to_raster('Calib1OLS2013.tif')      # This is radiance as well calibrated with DNB

# Calibration 1 by exponential
# olsrad_arr1 = copy.deepcopy(olsrad_arr)
# olsrad_arr1[olsrad_arr1==0] = 0
# olsrad_arr1[olsrad_arr1!=0] = 0.3205397924*exp(0.0704915218*olsrad_arr1[olsrad_arr1!=0])
#
# ols_arrobj = Image_arr(olsrad_arr1)
# ols_arrobj.georef = ols.georef
# ols_arrobj.arr_to_raster('Calib1_OLS2013.tif')      # This is radiance as well calibrated with DNB


# Calibration 2 by parts
# olsrad_arr2 = olsrad_arr
# olsrad_arr2[(olsrad_arr2)<262.53] =(olsrad_arr2[(olsrad_arr2)<262.53]*0.1159872231)
# olsrad_arr2[olsrad_arr2>=262.53] =  4.5*exp(0.0612148962*olsrad_arr2[olsrad_arr2>=262.53])
#
# ols_arrobj = Image_arr(olsrad_arr2)
# ols_arrobj.georef = ols.georef
# ols_arrobj.arr_to_raster('Calib2_OLS2013.tif')


# 3. Next comparing resampled DNB and calibrated radiance from OLS at taluk level by Creating statistic csv file for each shapefile X image . Original in read_India_stats.py
dnb.file = r'/home/prakhar/Research/AQM_research/Data/Data_process/Resampl_img/ResamplDNB_OLS_2013.tif'     #file for which the stat are to be found out
shapefile_path = r'/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/Ind_adm3_splitshp//'  #IND_adm3_ID_3__299.shp  # the place all the split shapefiels are stored
df_shpatt = pd.read_csv( shapefile_path + 'IND_adm3.csv', header=0)  # List of all shape file. making dataframe of shape file attribute list

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


# * * * *  * * # * * * *  * * # * * * *  * *# # *  Task IV OLS-TNL sum for GDP verfication *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Zhang/'
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
input_zone_polygon = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
for year in range(2001, 2014):
    ols.file = glob(os.path.join(ols.path, '*'+str(year)+ '*'+'.tif'))[0]
    print year, srs.zonal_stats(input_zone_polygon, ols.file)[0]

df_nlgdp = pd.read_csv(csv_in_path+'OLS_intercalibration_GDP.csv', header=0 )
df_nlgdp[0:13].corr("pearson")
# 2001 - 2013coorelation:  Wu 0.89, Zhang - 0.87


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

