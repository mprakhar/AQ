#__author__ = 'Prakhar MISRA'
# Created 3/26/2016
#Last edit 9/18/2017

# Purpose: (1) For a given pair of images with different resolutions, this will find a regression equation between the DN values of two images. It will then be used for calibration so as to continue legacy from one sensor to another. This function
#           also first resamples the images at same resolution to be able to compare the pixels
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
satS = 'VIIRS'       # options VIIRS
prodS = 'DNB'        # options DNB refers to radiance values obtained from DN values
# ImageT: target reolsution to be achieved, ImageS: source image to be resampled
raster_pathT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\\'
input_value_rasterT = r'F:\DMSP OLS\F182013.v4\F182013.v4c_web.stable_lights.avg_vis.tif\F182013.v4c_web.stable_lights.avg_vis.tif'
raster_pathS=r'D:\2013 beta\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif'
input_value_rasterS= r'D:\2013 beta\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif\npp_d20130101_d20130131.zero_li_on.sl_off.cloud_iicmofix.75N060E.avg_dnb.tif'
georef_raster = gb_path + r"Data/Data_process/Georef_img/DMSP_georef.tif" #reference image that has the georef info

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

def plot_scatter(satT, prodT, satS, prodS):
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


    dfTT['rad2013'] = dfTT[prodTY]**(3.0/2.0) #radis the radiacne values

    y_label_text= {'ANG': '','AOD':'', 'SO2': '(DU)', 'NO2': '(10E13 molecules/cm2)', 'DNB': '( 10E(-13)nano-Watts/cm2 sr )', }

    mpt.df2Plot2D(df1=dfTT, Xaxis=('rad'+YEARM), df2=dfRR, Yaxis=prodS+YEARM, subt='Pixel by pixel comaprison for '+YEARM, Xlabel=prodT+y_label_text[prodS], Ylabel=prodS+y_label_text[prodS], save_path=plt_save_path+prodS+prodT+YEARM+'.jpg', save=1)

    return dfTT, dfRR




# ##   the following analysis is excusively for calibration between OLS and DNB based on 2013 images   ####

# the following curve fitting gives a very poor regression and intercept.

dfTT, dfRR = plot_scatter()

prodTY=prodT+YEARM #OLS2013
prodRY=prodS+YEARM #DNB2013

np.polyfit(dfTT['rad2013'][1:4032154],dfRR[prodRY][1:4032154], deg=1)
# henxce we need to first clean the data of outliers and then run it again
# to clean the  data of outliers, I. sort DNB2013 on the basis of OLS2013. then for each OLS2013 group find the mean, median , std of DNB2013. for these 64 values then we can then make a plot using excel or whatever

dfG[prodRY].describe()
dfM=dfTT.copy(deep=True) #creating a a new dataframe with merged columns from DFRR and dfTT
dfM[prodRY]=dfRR[prodRY]
dfM.groupby(prodTY).mean()

dfN[prodRY+'_mean']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.mean()) #dfN = Navigate into unknown :)
dfN[prodRY+'_med']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.median())
dfN[prodRY+'_min']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.min())
dfN[prodRY+'_max']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.max())
dfN[prodRY+'_std']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.std())
dfN['rad2013']=pd.DataFrame(dfM.groupby(prodTY).rad2013.mean())
dfN[prodRY+'_count']=pd.DataFrame(dfM.groupby(prodTY).DNB2013.count())
dfN.to_csv(csv_save_path+'df_LUT_stats_'+prodT+prodS+'_'+dor+ver+'.csv', index=True, header=True)    #generating a kind of LUT for calibration from OLS to DNB






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

def resample(prodS, prodT):
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
def DNtorad(ols):
    ols_arr = ols.zone_mask(input_zone_polygon_0)[0]
    olsrad_arr = np.power(np.array(ols_arr, dtype=float), 3.0/2.0)
    ols_arrobj = Image_arr(olsrad_arr)
    ols_arrobj.georef = ols.georef
    ols_arrobj.arr_to_raster('RadOLS2013.tif')
    return olsrad_arr

# finding fitting curve for OLSrad and DNB 2013
def func(x, a, b, c):
    #return a*x**2 + b*x + c        # quad function
    # scaled by .0000008 and 0.0005 for clarity
    return 0.000008*a * np.exp(-0.0005*b * x) + 1.5*c       #exponential fucntion

#LUT orepared manually on excel
lut=pd.read_csv(csv_save_path + 'df_LUT_2013OLSDNB.csv', header=0)


# Performing calibration by parts

# a) Exponential part
# funciton to calulcate coefficent of exponential portion
def plot_exponential(x,y, lut):
    popt, pcov = curve_fit(func, x, y)
    a,b,c = popt #([  8.92266619e-05,  -9.34595854e-03,   1.28252397e+00]) -- for quadratic

    plt.figure()
    plt.plot(x, y, 'ko', label="DNB & OLSrad")

    #l2=('Fitted curve : ' +str('%09f'%a)+'x$^2$+'+str('%09f'%b)+'x+'+str('%09f'%c)) # for quadratic eqn
    l2 = ('Fitted curve : ' +str((0.000008*a))+'e('+str('%05f'%(-0.0005*b) )+'x)+'+str('%05f'% (1.5*c))) # for quadratic eqn
    plt.plot(x, func(x, *popt), 'r-', label=l2)
    s1=lut.DNB2013_std.apply(int)
    plt.scatter(x, y,s = s1*10, color = 'black', alpha = 0.5, label = 'Standard deviation in DNB')
    plt.legend()
    plt.show()
    return a,b,c, popt
#end function

# input stff for plt exponential
DN_brkpt = 47 #DN rbeakpoint
y = lut.DNB2013_mean_corrected[DN_brkpt:] # we first make exponential fitting using 47*. but later to trend fit only the acurate values, we take linear until 54 and exponetial thence 54:
x = lut.rad2013[DN_brkpt:]
#run finction
a,b,c, popt = plot_exponential(x,y, lut)


# b) plotting by parts

def plot_exponential_linear(x,y, lut , bp, a,b,c, popt):

    plt.hold(True)
    plt.figure()
    plt.plot(x, y, 'ko', label="DNB & OLSrad")
    plt.show()

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
# functon end
DN_brkpt = 57 #DN rbeakpoint
bp = DN_brkpt -4
y = lut.DNB2013_mean_corrected[0:]
x = lut.rad2013[0:]
plot_exponential_linear(x,y, lut, bp,  a,b,c, popt)


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


# -- Calibration by order two curve --
def OLS2013calibsave():
    olsrad_arr = DNtorad(ols)
    olsrad_arr1 = copy.deepcopy(olsrad_arr)

    #linear coeffcient
    y2 = 430.33999999999997
    x2 = 10.710000000000001
    #exponential coeffcient
    a = 0.009178868005602062
    b = -77.433852827324472
    c= 6.4071776523406472

    # calibrating
    olsrad_arr1[olsrad_arr1<=430] = (x2/y2)*olsrad_arr1[olsrad_arr1<=430]
    olsrad_arr1[olsrad_arr1>430] = 0.000008*a*np.exp(-0.0005*b*olsrad_arr1[olsrad_arr1>430])+1.5*c

    #save image
    srs.rio_arr_to_raster(olsrad_arr1, ols.georef, 'Calib1OLS2013.tif')

# Calibration 1 by exponential
# olsrad_arr1 = copy.deepcopy(olsrad_arr)
# olsrad_arr1[olsrad_arr1==0] = 0
# olsrad_arr1[olsrad_arr1!=0] = 0.3205397924*exp(0.0704915218*olsrad_arr1[olsrad_arr1!=0])
#
# srs.rio_arr_to_raster(olsrad_arr1, ols.georef, 'Calib1_OLS2013.tif')      # This is radiance as well calibrated with DNB


# Calibration 2 by parts
# olsrad_arr2 = olsrad_arr
# olsrad_arr2[(olsrad_arr2)<262.53] =(olsrad_arr2[(olsrad_arr2)<262.53]*0.1159872231)
# olsrad_arr2[olsrad_arr2>=262.53] =  4.5*exp(0.0612148962*olsrad_arr2[olsrad_arr2>=262.53])
#
# srs.rio_arr_to_raster(olsrad_arr2, ols.georef, 'Calib2_OLS2013.tif')
