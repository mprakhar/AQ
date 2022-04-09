#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar'
# Created 2/29/2016
# Last edit 11/29/2016

# ~ this code is deifferent from pervios ones in the sense that it relies completely on class and bjects

# Purpose: To convert two different sensor rasters at same resolution tocompare their output for a given shapefile
#          To classify AQ - NL images by thresholds and generate images for creating a transition movie
# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used: T -  target resolution to be achieved. usually MODIS, OMI image; S - source of the image to be resampled
'''# output filenames produced
'df_PixVal_'+satT+prodT+'_'+dor+ver+'.csv' :
'df_PixValT_'+satT+prodT+'_'+dor+ver+'.csv' : deprecated; would give trasnposed of df_PixVal_'+satT+prodT+'_'+dor+ver+'.csv'
'df_PixVal_Resampl_'+satT+'_'+satS+prodS+'_'+dor+ver+'.csv' :
'df_PixValDNBAODANG'+YEARM+'_v2.csv' :
'''

import numpy
from numpy import *
import numpy.ma as ma
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import block_reduce

sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from  spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct


from  spatialop import infoFinder as info
from  spatialop import shp_rstr_stat as srs
import pandas as pd
import csv
from  spatialop import my_plot as mpt
import pylab
from  spatialop import my_math as mth
import seaborn as sns
import os.path
import copy
from glob import glob

from  spatialop import classRaster
from classRaster import Raster_file
from classRaster import Image_arr
# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Step0: Initialize     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
#Input
ver = 'v1'  # version number of output
dor = '20161012'  # date of run
YEARM = '200800'  # yearmonth for running and comapring, e.g 201406 for VIIRS or 200800 for OLS; 00 in the end implies annual
START_VRS = 200100  # data from which VIIRS data is avaliable.. a constant value, hence all in caps.showuld be 201400 in case of monthly VRS
END_VRS = 201500
GIS_C = 13  # class number of target feature type. 13 refers to class urban in the LULC map
sat_aq = [('MODIS', 'ANG'), ('MODIS', 'AOD'), ('OMI', 'SO2'), ('OMI', 'NO2')]  # Air quality satellites
sat_nl = [('VIIRS', 'DNB'), ('DMSP', 'OLS')]  # Nightlight satellite


#Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

aod = Raster_file()
aod.path = gb_path + r'/Data/Data_process/Meanannual_MODIS/clean/'
aod.sat = 'MODIS'
aod.prod = 'AOD'
aod.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
aod.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_georef.tif'

ang = Raster_file()
ang.path = gb_path + r'/Data/Data_process/Meanannual_MODIS/clean/'
ang.sat = 'MODIS'
ang.prod = 'ANG'
ang.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'MOD04L2.A201511.AOD.Global'
ang.georef = gb_path + r'/Data/Data_process/Georef_img//MODIS_georef.tif'

so2 = Raster_file()
so2.path = gb_path + r'/Data/Data_process/Meanannual_OMI/clean/'
so2.sat = 'OMI'
so2.prod = 'SO2'
so2.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'OMI.SO2.201412.min.Global'
so2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

no2 = Raster_file()
no2.path = gb_path + r'/Data/Data_process/Meanannual_OMI/clean/'
no2.sat = 'OMI'
no2.prod = 'NO2'
no2.sample = gb_path + 'Data/Data_raw/MOD04L2/L4//' + 'OMI.NO2.201412.min.Global'
no2.georef = gb_path + '/Data/Data_process/Georef_img//OMI_georef.tif'

dnb = Raster_file()
dnb.path = gb_path + r'/Data/Data_raw/VIIRS Composite/75N060E//'
dnb.sat = 'VIIRS'
dnb.prod = 'DNB'
dnb.sample = dnb.path + '/SVDNB_npp_20140601-20140630_75N060E_vcmslcfg_v10_c201502121209.avg_rade9.tif'

ols = Raster_file()
ols.path = gb_path + r'Data/Data_process/DMSPInterannualCalibrated_20160512//'
ols.sat = 'DMSP'
ols.prod = 'OLS'
ols.sample = gb_path + 'Data/Data_raw/DMSP OLS/F162008.v4/F162008.v4b_web.stable_lights.avg_vis.tif'
ols.georef = gb_path + '/Data/Data_process/Georef_img//DMSP_georef.tif'

gis = Raster_file()
gis.path = gb_path + '/Data/Data_raw/GIS data/' + 'GlobalLandCover_tif/'
gis.sat = 'GIS'
gis.prod = 'LCType'
gis.sample = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'
gis.georef = gb_path + '/Data/Data_raw/GIS data/GlobalLandCover_tif/LCType.tif' + 'LCType.tif'




input_zone_polygon = gb_path+'/Data/Data_raw/Shapefiles/IND_adm1/IND_adm0.shp'
# folder or directory path locations of data
# sample name of the file
sample_raster = {'LULC': 'GlobalLandCover_tif\LCType.tif',
                 'POP': 'Ind Population\IND-POP\IND15\popmap15.tif'}

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#     Step I: DATA PREPARATION       * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#
# for low resolution images images
global G_datamask  # G represents gloabl variable


# ------------------------------------------------------     1.1  Pixel resampling function    ------------------------------------------------------

# function begin
def pix_resample(prodS, prodT):
    ## function to read files, mask them  them and resample to same resoltion. also store the pixel coordinates along with pixel values
    ''' G : Function to save pixel level information of source (to be resampled to the Target resolution) and target(provides the resolution which has to be achieved) by applyingthe shapefile masks
    over both the files and then degrading source to target.
    Aimed at resampling DMSP OLS to MODIS and OMI resolution

    Parameters
    ----------
    input_zone_polygon : shapefile complete path
    satS : Source satelltie path
    prodS　：Source product path
    input_value_rasterS　：example of a particular source product
    satT　：target satelltie path
    prodT　：target product path
    input_value_rasterT　：example of a particular target product
    input_value_rasterG：location of GIS file to resampled
    G : boolean to check if any GIS file has to be resampled or not

   '''

    imgarrayS, datamaskS = prodS.zone_mask(input_zone_polygon)  # actually no use of datamask in this step
    imgarrayT, datamaskT = prodT.zone_mask(input_zone_polygon)
    G_datamask = datamaskT
    yfactor = (float((numpy.shape(imgarrayS))[0])) / (float((numpy.shape(imgarrayT))[0]))
    xfactor = (float((numpy.shape(imgarrayS))[1])) / (float((numpy.shape(imgarrayT))[1]))
    print "imgarray S resolution is", float((numpy.shape(imgarrayS))[0]), float((numpy.shape(imgarrayS))[1]), "imgarray T resolution is", float((numpy.shape(imgarrayT))[0]), float((numpy.shape(imgarrayT))[1])

    if prodS.prod in 'OLS':
        imgarrayS = imgarrayS ** (1.5) #to convert DNB to radiance
        imgarrayR = block_reduce(imgarrayS.data, block_size=(int(ceil(yfactor)), int(ceil(xfactor))), func=numpy.mean)  # resampled image R ----- Imp Step
        imgarrayR = numpy.ma.masked_array(imgarrayR, numpy.logical_not( datamaskT))  # necessary to apply the mask again because of the blockreduce the mask gets removed.
        imgarrayR = imgarrayR ** (2.0/3.0)  # to convert DNB to radiance

    if prodS.prod in 'LCType':
        GIS_C = 13       # 13 is the class number for urban
        numpy.place(imgarrayS, imgarrayS.data != GIS_C,0)  # 13 is class name for urban GIS_C; replacing all other classes by value 0
        imgarrayR = block_reduce(imgarrayS.data, block_size=(int(ceil(yfactor)), int(ceil(xfactor))), func=numpy.sum)  # resampled image R ----- Imp Step
        imgarrayR = numpy.ma.masked_array(imgarrayR, numpy.logical_not(G_datamask))  # necessary to apply the mask again because of the blockreduce the mask gets removed.

        '''
        per_urbanpix is the minimum number of pixel in the  rasampling kernel block that should be urban for the whole block to be tagged as urban
        MODIS: performing some classificaiton again for MODIS resmpled image (downscaling ration is 20X20) to resample based on the criteria that if 25% percent of a pixel is urban, make it whole urban to include suburban area as well
        MODIS: so max value of a pixel is 5200. why? because if 20*20 pixels have value 13 i.e. 400*13 =5200. therefore 1300 is 1/4 of 20X20X13
        OMI: performing some classificaiton again for OMI resmpled image (downscaling ration is 60X60) to resample based on the criteria that if 25% percent of a pixel is urban, make it whole urban to include suburban area as well
        OMI: so max value of a pixel is 34169. therefore 8500 is 1/4  roughly 34619. MODIS logic does not work because it is too coarse here. criteria wil become too strict
        '''
        if prodT.sat == 'MODIS':
            URBANPIX = 1300  # hardcoded for the given pair of MODIS and LULC mask
        elif prodT.sat == 'OMI':
            URBANPIX = 6000  # hardcoded for the given pair of MODIS and LULC mask
        # creating urban mask based on our new minimum pixel assumption as mentioned above
        numpy.place(imgarrayR, imgarrayR.data < URBANPIX, 0)
        numpy.place(imgarrayR, imgarrayR.data >= URBANPIX, 1)

    if prodS.prod in 'DNB':
        imgarrayR = block_reduce(imgarrayS.data, block_size=(int(ceil(yfactor)), int(ceil(xfactor))),func=numpy.mean)  # resampled image R ----- Imp Step
        imgarrayR = numpy.ma.masked_array(imgarrayR, numpy.logical_not(datamaskT))  # necessary to apply the mask again because of the blockreduce the mask gets removed.

    prodR = Image_arr(imgarrayR)
    prodR.georef = prodT.georef
    return prodR

# function end



# --------------------------------------------------------------       1.4    LULC/GIS             --------------------------------------------------------------

#

# RUN FUNTION pix_resample
prodT = so2
prodS = ols

prodT.file = glob(os.path.join(prodT.path, '*'+'.tif'))[0]
for prodS.file in glob(os.path.join(prodS.path, '*'+'.tif')):
    year = int(prodS.file[-8:-4])
    # prodS.file = prodS.path + 'SVDNB_npp_20140601-20140630_75N060E_vcmslcfg_v10_c201502121209.avg_rade9.tif'
    # year = 2015
    print year
    prodR = pix_resample(prodS, prodT)
    prodR.arr_to_raster(prodS.sat+prodT.sat+str(year)+'.tif')


# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * *  * *#         Step II:   DATA ANALYSIS : to visualize whatever has been created so far          * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# --------------------------------------------------------------        2.1  Visualize scatter plots        --------------------------------------------------------------

prodS = ols
prodS.path = gb_path + 'Data/Data_process/Resampl_img/'
ls=[]
for prodT in [aod, ang, so2, no2]:
    for prodT.file in glob(os.path.join(prodT.path, '*' + prodT.prod + '*'+ '.tif')):
        year =  prodT.file[-8:-4]
        # print year
        prodS.file = glob(os.path.join(prodS.path + 'DMSP'+prodT.sat + str(year) + '.tif'))[0]
        imgarrayS, datamaskS = prodS.zone_mask(input_zone_polygon)  # actually no use of datamask in this step
        imgarrayT, datamaskT = prodT.zone_mask(input_zone_polygon)
        ls_imgarrayS = pd.DataFrame((imgarrayS[~imgarrayS.mask]).tolist())
        ls_imgarrayT = pd.DataFrame((imgarrayT[~imgarrayT.mask]).tolist())

        # also getting infor for urban pixels
        urbanmask = srs.raster_as_array(gb_path + 'Data/Data_process/Resampl_img/' + 'GIS' + prodT.sat + '2010.tif')
        ls_urbanmask = (pd.DataFrame((urbanmask[~imgarrayT.mask]).tolist()))

        df_attrib = ls_imgarrayS
        df_attrib['x'] = ls_imgarrayS        # NL
        df_attrib['y'] = ls_imgarrayT/1000  # AQ
        df_attrib['z'] = ls_urbanmask  # urbanmask


        # plotting
        mkr = {0: '.', 1: 'o'}
        size = {0: 15, 1: 30}
        color = {0: 'green', 1: 'navy'}
        fig, ax = plt.subplots()
        labeltxt = {0:'non-urban class pixels', 1:'urban class pixels'}

        for kind in mkr:
            d = df_attrib[df_attrib.z == kind]
            plt.scatter(d['x'], d['y'] , color=color[kind], s=size[kind] , marker=mkr[kind], alpha=0.6, label =labeltxt[kind])  # put dfT[prod+YEARM]/1000 for MODIS products

        plt.xlim([0, int((amax(df_attrib.x))) +1.0])
        plt.ylim([0, int((amax(df_attrib.y))) +1.0])
        ax.set_title('Scatter plot for NL pixels and ' + prodT.prod +' : ' + str(year), fontsize=20)

        plt.xlabel(prodS.prod, fontsize=17)
        plt.ylabel(prodT.prod, fontsize=17)

        plt.legend(fontsize=15)
        plt.savefig(plt_save_path + 'AQNL_' + prodT.prod + year + '.png')


        # Thresholding
        MINVAL ={'AOD':20.0, 'ANG':20.0, 'SO2':5.0, 'NO2':5.0}
        thresh = mth.RosinThreshold((imgarrayT[~imgarrayT.mask]), MINVAL[prodT.prod])
        ls.append([prodT.prod, year, thresh])

# make historgams and determine the thresholds to be used in the next steps
df_thresh = pd.DataFrame(ls, columns=['prod', 'year', 'thresh'])
df_thresh.to_csv(csv_save_path + 'df_threshAQ.csv',  index=False, header=False )




# --------------------------------------------------------------       2.3    Classification step             --------------------------------------------------------------

THRESH = {'ANG': 789, 'AOD': 301, 'SO2': 229, 'NO2': 4045, 'DNB': 10, 'OLS': 49**1.5}  # threshold values for the parameters for base year 2005. stored in E:\Acads\Research\AQM research\Docs prepared\Excel\AQ Nightlight\Mean_annual_img_thresh.xlsx
# applying some thresholds: AOD 200, DNB 0.3, ANG250, NO2 2000, SO2 100  >>>>>>> Very important to optimize the thresholding value chosen!!!!
THRESH = {'ANG': 663, 'AOD': 288, 'SO2': 152, 'NO2': 2879, 'DNB': 4.9, 'OLS': 30}  # minimum threshold of threshold

THRESH = {'ANG': 1100, 'AOD': 400, 'SO2': 350, 'NO2': 4000, 'DNB': 4.9, 'OLS': 25}  # minimum threshold of threshold
prodS = ols
prodS.path = gb_path + 'Data/Data_process/Resampl_img/'
ls_o =[]
ls_u =[]
for prodT in [aod, ang, so2, no2]:
    for prodT.file in glob(os.path.join(prodT.path, '*' + prodT.prod + '*'+ '.tif')):
        year =  int(prodT.file[-8:-4])
        if year>= 2014:
            THRESH['OLS']=2.86 # this is obatined by reverse applying DMSP calibration equaiton on OLS2013 to find correspodning DN for 30. the using that DN to find DNB values

        prodS.file = glob(os.path.join(prodS.path + 'DMSP' + prodT.sat + str(year) + '.tif'))[0]
        imgarrayT = prodT.raster_as_array()
        imgarrayS = prodS.raster_as_array()

        #Need to read the datamsk to subset
        datamask = srs.raster_as_array('Maskarray'+prodT.sat+'.tif')

        #Classificaion step
        imgarrayC = copy.copy(imgarrayT)
        imgarrayC[(imgarrayS < THRESH[prodS.prod]) & (imgarrayT >= THRESH[prodT.prod])] = 1  # LLHP
        imgarrayC[(imgarrayS >= THRESH[prodS.prod]) & (imgarrayT >= THRESH[prodT.prod])] = 2  # HLHP
        imgarrayC[(imgarrayS < THRESH[prodS.prod]) & (imgarrayT < THRESH[prodT.prod])] = 3  # LLLP
        imgarrayC[(imgarrayS >= THRESH[prodS.prod]) & (imgarrayT < THRESH[prodT.prod])] = 4  # HLLP

        #Plot the classified data
        fig, ax = plt.subplots()
        imgarraysub = ma.masked_array(imgarrayC, (datamask-1)*(-1))     #just inverting the mask attched with arrayC
        cmap = plt.get_cmap('jet', 4)
        plt.imshow(imgarraysub, cmap=cmap)
        ax.grid(False)
        plt.title(' Pixel Classification by ' + prodT.prod + ' and NL thresholds for ' + str(year), fontsize=20, alpha=0.7)
        nv_patch = mpatches.Patch(color='navy', label='LLHP')
        aq_patch = mpatches.Patch(color='aqua', label='HLHP')
        yl_patch = mpatches.Patch(color='yellow', label='LLLP')
        mr_patch = mpatches.Patch(color='maroon', label='HLLP')
        plt.legend(handles=[nv_patch, aq_patch, yl_patch, mr_patch])
        plt.savefig(plt_save_path + 'AQNLClass_'+prodT.prod+str(year)+'.png')

        # store the overal statistics
        print year, prodT.prod, (imgarraysub==1).sum(), (imgarraysub==2).sum(), (imgarraysub==3).sum(), (imgarraysub == 4).sum()
        ls_o.append([year, prodT.prod, (imgarraysub==1).sum(), (imgarraysub==2).sum(), (imgarraysub==3).sum(), (imgarraysub == 4).sum()])

        # working only for urban pixels
        urbanmask = srs.raster_as_array(gb_path + 'Data/Data_process/Resampl_img/' + 'GIS' + prodT.sat + '2010.tif')
        imgarraysub = ma.masked_array(imgarrayC, (urbanmask - 1) * (-1))

        # store the overal statistics
        print year, prodT.prod, (imgarraysub==1).sum(), (imgarraysub==2).sum(), (imgarraysub==3).sum(), (imgarraysub == 4).sum()
        ls_u.append([year, prodT.prod, (imgarraysub==1).sum(), (imgarraysub==2).sum(), (imgarraysub==3).sum(), (imgarraysub == 4).sum()])

df_overallclass = pd.DataFrame(ls_o, columns=['year','prod', 'LLHP','HLHP','LLLP','HLLP'])
df_overallclass.to_csv(csv_save_path + 'df_AQNLClassif_overall.csv', index=False, header=True)
df_urbanclass = pd.DataFrame(ls_u, columns=['year','prod', 'LLHP','HLHP','LLLP','HLLP'])
df_urbanclass.to_csv(csv_save_path + 'df_AQNLClassif_urban.csv', index=False, header=True)

# Plot the urban trends
df_urbanclass = pd.read_csv(csv_save_path + 'df_AQNLClassif_urban.csv', header=0)
for aq in ['AOD', 'ANG', 'SO2', 'NO2']:
    x = df_urbanclass[df_urbanclass['prod']==aq]['year']

    plt.figure(figsize=(10,4))
    plt.plot(x, df_urbanclass[df_urbanclass['prod']==aq]['HLHP'], 'bo-' , label='HLHP')
    plt.plot(x, df_urbanclass[df_urbanclass['prod']==aq]['LLHP'],'g>-' ,label='LLHP')
    plt.plot(x, df_urbanclass[df_urbanclass['prod']==aq]['HLLP'],'r^-', label='HLLP')
    plt.plot(x, df_urbanclass[df_urbanclass['prod']==aq]['LLLP'], 'k<-' , label='LLLP')
    plt.title('AQ&NL trend for '+aq, fontsize=20)
    plt.legend(fontsize=15)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Urban pixel count', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    plt.tight_layout()
    plt.savefig(plt_save_path+'AQNLClassurban'+aq+'.png')



# =========================# =========================# =========================# =========================# =========================





# to check the colormapp produced and corresponding values
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plot = ax.pcolor(imgarraysubset)
fig.colorbar(plot)

# to get counts len(result2[result2['SO2filter']==2])





# --------------------------------------------------------------        2.4   plotting scatter plots with attribute        --------------------------------------------------------------

y_label_text = {'ANG': '', 'AOD': '', 'SO2': '(X1000 DU)', 'NO2': '(microgram/m2)', 'DNB': '(nano-Watts/cm2 sr)'}
prodS = ols
for prodT in sat_aq:
    dfT = pd.read_csv(
        'E:\Acads\Research\AQM research\Docs prepared\Excel files\AQ-DNB Classification thresh counts ' + YEARM + prodT.sat + ' 20160302.csv',
        header=0)

    if prodT.sat in 'MODIS':
        pylab.xlim([0, max(dfT[prodS.prod + YEARM])])
        pylab.ylim([0, max(dfT[prodT.prod + YEARM]) / 1000])

    if satT in 'OMI':
        pylab.xlim([0, max(dfT[prodS.prod + YEARM])])
        if prodT.prod in 'SO2':
            pylab.ylim([0, 0.5])  # 0.5 for SO2 and 7 for NO2
        if prodT.prod in 'NO2':
            pylab.ylim([0, 7])

    # fig=plt.figure()
    mkr = {0: '.', 1: '+'}
    for kind in mkr:
        d = dfT[dfT.LULC_dummy == kind]
        plt.scatter(d[prodS.prod + YEARM], d[prodT.prod + YEARM] / 1000, s=d['Area'], cmap='OrRd',
                    marker=mkr[kind])  # put dfT[prod+YEARM]/1000 for MODIS products
        # fig.suptitle('Comparison of '+ prodT+' and '+prodS+' for Urban and Non-urban pixels '+YEARM, fontsize=15)
        plt.ylabel(prodT.prod + y_label_text[prodT.prod], fontsize=18)  # DU for SO2  prodcust, 10E13 molecules/cm2
        plt.xlabel(prodS.prod + y_label_text[prodS.prod], fontsize=18)
        # fig.savefig(plt_save_path+prodS+prodT+YEARM+'_v3.jpg')  # ???? connect pylab and figure to enable saving


# --------------------------------------------------------------       taking screenshots for ppt        --------------------------------------------------------------

prodS =ols

for prodS.file in glob(os.path.join(prodS.path, '*' + prodS.prod + '*'+ '.tif')):
    year = int(prodS.file[-8:-4])
    imgarrayS = prodS.raster_as_array()
