#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created11/25/2017
# Last edit 12/15/2017

#

# Purpose: make toy AQ socio-economoc model

#changes:
# in UM R +18 removed
# aster considerd for 2001
# IWD now as 1/d**2
# Location of output: E:\Acads\Research\AQM\Data process\CSVOut

# terminology used:
'''# output filenames produced

'''

import rasterio as rio
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import block_reduce
from sklearn import linear_model
import seaborn as sns
import sys
import os

# pvt imports
sys.path.append('/home/prakhar/Research/AQM_research/Codes/')
import spatialop
from  spatialop import shp_rstr_stat as srs
from spatialop.classRaster import Raster_file
import spatialop.coord_translate as ct




# declare
gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path

city20shppath = gb_path + r'/Data/Data_process/Shapefiles/20city_big_shapefiles/'

# Output location
output_value_raster_path = gb_path + r'/Data/Data_process/AW3D/India/20city/'
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut/DSM/20city/'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data\Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'
#get present dirctory
pwd = os.getcwd()

#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
#-----------------Function--------------------------------------------#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def getpixcoord(citychk, arr_path, citylist_path):
    # get the pixel llocation coordinates
    # open the csv
    df_citylist = pd.read_csv(citylist_path, header = 0)

    # read the lat lon
    lat = df_citylist[df_citylist.city_coord ==citychk]['lat'].values[0]
    lon = df_citylist[df_citylist.city_coord ==citychk]['lon'].values[0]

    # get the pixel coordinate
    [pix1] = ct.latLonToPixel(arr_path ,[[float(lat) ,float(lon)]] ) #ch

    return pix1

# create array distance from city centre array
def distarray_residential(citychk, arr_path, citylist_path):

    #get the pix coordinate
    pix1 =  getpixcoord(citychk, arr_path, citylist_path)

    # fill array with distacen
    distarr = np.fromfunction(lambda i ,j: np.sqrt(( i -pix1[1] ) **2 + ( j -pix1[0] ) **2) , rio.open(arr_path).read(1).shape, dtype = np.float32)

    # multiply by pixel size - 30m/1000m to convert to weights to km
    distarr = distarr*30/1000



    return distarr

# create array distance from city centre array
def distarray_industrial(citychk, arr_path, citylist_path):

    #get the pix coordinate
    pix1 =  getpixcoord(citychk, arr_path, citylist_path)

    # fill array with distacen
    distarr = np.fromfunction(lambda i ,j: np.sqrt(( i -pix1[1] ) **2 + ( j -pix1[0] ) **2) , rio.open(arr_path).read(1).shape, dtype = np.float32)


    # multiply by pixel size - 30m/1000m to convert to weights to km
    distarr = distarr*30/1000

    #remove zero distance; assuming effect of fatory is equally spread upto 1km radius from the factory
    distarr = np.where(distarr <= 1, 1, distarr)

    return distarr



def weighted_distance(featurearr, distarr):
    # get weighted distance

    #feature array is a boolean array of locaiton pf the featurees

    #inverted values in array ()as per Habibi, Alesheikh 2017
    invdist = 1/(distarr**2)

    # get weighted sum by multiply by featur array
    weightedsum = np.nansum(featurearr*invdist)

    return weightedsum

def resample (sourcepath, targetpath, destinationpath ):
    #fucntion to resample road network raster to MODI level
    #sourcepath = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/Ind_road/road_raster_1000m_clip.tif'
    #targetpath = '/mnt/usr1/home/prakhar/Research/AQM_research/Data/Data_process/AirRGB/AirRGB20171202Imasu/India/Month/MonthRGB200101.tif'

    #rotate
    source = rio.open(sourcepath).read(1)
    target = rio.open(targetpath).read(1)

    xfactor = int(np.shape(source)[1]/np.shape(target)[1])
    yfactor = int(np.shape(source)[0] / np.shape(target)[0])

    resampsource = block_reduce(source, block_size = (yfactor, xfactor), func = np.sum)

    srs.rio_arr_to_raster(resampsource, targetpath, destinationpath)

def get_betaRH(RH):
    # beta refers to mass extinction efficiency to correct AOD by RH
    # based on Tropospheric AOT from the GOCART model and comaparisons with satelluite and sun photometer measurements (chin, ginoux, kinne 2002)
    #returning beta values for hydrophiolic OC only
    RH = int(RH)
    if RH < 0:
        beta = 11/11.

    if (RH >=20) & (RH <40):
        beta = 12/11.

    if (RH >=40) & (RH <60):
        beta = 13/11.

    if (RH >=60) & (RH <80):
        beta = 14/11.

    if RH>=80:
        beta = np.e**(0.0001376*RH**3 - 0.0355741*RH**2 + 3.0742882*RH - 85.9946270)/11.

    return beta


#--------------------------------------------------------------------------
#-----------------Function--------------------------------------------#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------



class AQmodel():

    #'class to find beta coefficients and rtun ti over annual data to comapre estimated and observed '

    #UM estimates obtained by area expansion function
    UMestimates = pwd +  r'/lib/area_growth_estimates_clean_20180204.csv'

    #vehicle populaiton estimate
    vehiclestimate = pwd + r'/lib/Vehicleestimates.csv'

    # funxtion to generate distance image from city centre
    citylist_path = pwd + r'/lib/20citycoordforAQmodel.csv'

    #agrod fire df path
    df_agrfirepath = pwd + r'/lib/df_20cityFRPcount_300_0.csv'

    # location of sevtor wise emission intensity
    EIpath = pwd + r'/lib/EmissionIntensityratio.csv'

    #roadnetwork rsater, already resampled to MODIS
    roadpath = gb_path + r'/Data/Data_process/Shapefiles/Ind_road/road_raster_10km_clip.tif'

    # BK estimate by ity
    BKestimatepath = pwd + r'/lib/BKweight20city.csv'

    #estimated VKT from patch indx and Builtup density
    df_VKTpath  = pwd + r'/lib/VKTcorrected.csv'

    #define the monthly activity
    # they are defined horizontally by  in the order
    # residentia, commericial , industrial, agriculture, brick kiln, vehicle, constant
    EA = np.array([
        [1, .8, .8, .8, .8, .8, .8, .8, .8, .8, .8, 1 ],
        [.5, .5, .5, .5, 1, 1, 1, 1, .5, .5, .5, .5 ],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, .8, .8, .8, .8, .8, .8, .8, .8, 1, 1 ],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
    ])

    #Vehilce annual in Kaour , startign from 2001


    def __init__(self, Count=0, citychk=0, citychk_loc=0, year=0, daily = False, EA=0, ASTER = True ):

        #define all the concerned infor for a city

        #given id to city
        self.Count = Count

        # e.g. Kanpur; central location
        self.citychk = citychk

        #e,g. Kanpur00, Kanpur12; surrounding ixels of central location
        self.citychk_loc = citychk_loc
        #self.df_agrfirepath = df_agrfirepath

        #year for which analysis performed
        self.year = year

        #open the RGB csvfile. This csv file contais daily level AOD, ANG and RGB values.
        self.df_rgb=0

        #set the shapefile fot he district
        self.shppath = gb_path + r'/Data/Data_process/Shapefiles/India_20citydistrict/2011_Dist_DISTRICT_'+citychk+'.shp'

        #set the UM AW3Dpath so that weighted distances can boud
        # set the AW3D UM ath
        self.UMpath = gb_path+ r'/Data/Data_process/AW3D/India/20city/UMpostprocessgeorefshp/' + self.citychk + 'UM.tif'

        if ASTER==True & self.year==2001:
            self.UMpath = gb_path + r'/Data/Data_process/ASTER/India/20city/UMpostprocessgeorefshp/' + self.citychk + 'UM.tif'

        #set the lcaiton for meterological parameter RH
        #self.RHpath = gb_path + r'/Data/Data_process/Meteorology/Weather/'+self.citychk+ 'RH.csv'
        self.RHpath = gb_path + r'/Data/Data_process/Meteorology/Weather/20citymeanRH.csv'

        #timestep??
        self.daily = daily

        #oprion to specify user specidefi speasonal emiasonal activity coeffocnt
        self.EA = EA



    def get_R(self, RHcorrect=True):
        #get R values corresponding to the city

        #set the R values as well
        if self.daily:
            self.df_rgb = pd.read_csv(pwd + r'/lib/observedAQImasumodII/df_AQRGB_daily' + self.citychk_loc + '.csv', header=0)
        else:
            self.df_rgb = pd.read_csv(pwd + r'/lib/observedAQImasumodII/df_AQRGB_' + self.citychk_loc + '.csv', header=0)

        #set time index
        self.df_rgb['date'] = pd.to_datetime(self.df_rgb['date'], format = "%Y%m")
        self.df_rgb['date1'] = pd.to_datetime(self.df_rgb['date'], format = "%Y%m")
        self.df_rgb['year'] = pd.DatetimeIndex(self.df_rgb.date).year
        self.df_rgb['month'] = pd.DatetimeIndex(self.df_rgb.date).month

        #make date as index
        self.df_rgb.set_index('date1', inplace=True)

        if self.daily:
            #monthly dataplot
            y_monthly = self.df_rgb[self.df_rgb.year==self.year].groupby(pd.TimeGrouper(freq='M')).mean()
        else:
            y_monthly = self.df_rgb[self.df_rgb.year == self.year]

            if RHcorrect:

                y_monthly = self.correctRH(y_monthly[['R', 'date', 'month']])

        return y_monthly

    # old method of gettingRH that considered daily RH from wunderground.com
    def correctRHold(self, y_monthly):
        # correct R values for seaosnal change by dividing it by relative humidity

        #open the RH values
        df_RH = pd.read_csv(self.RHpath, header = 0)


        #set datetime index
        df_RH['date'] = pd.to_datetime(df_RH['year'], format = "%Y%m")
        df_RH['date1'] = pd.to_datetime(df_RH['date'], format = "%Y%m")
        df_RH['year'] = pd.DatetimeIndex(df_RH.date).year
        df_RH['month'] = pd.DatetimeIndex(df_RH.date).month

        #merge values by common date
        df_y_monthly = pd.merge(y_monthly, df_RH, how = 'left', on = ['date'])
        df_y_monthly['AvgHumidity'] = df_y_monthly['AvgHumidity'] / 100.0

        df_y_monthly['beta'] = df_y_monthly['AvgHumidity']

        #calling coefficient factor
        df_y_monthly['beta'] = df_y_monthly['beta'].apply(lambda x: get_betaRH(x*100))
        #df_y_monthly['beta'] = np.max(1, df_y_monthly['AvgHumidity']/.6)

        #now divide R by RH
        #simple method
        df_y_monthly['R'] = df_y_monthly['R']/(df_y_monthly['beta'])

        #method by Lowenthal (1995), Laulainen(1993)
        #df_y_monthly['R'] = df_y_monthly['R'] * (1+.25*(df_y_monthly['AvgHumidity'])**2/(1-df_y_monthly['AvgHumidity']))

        return df_y_monthly


    def correctRH(self, y_monthly):
        # correct R values for seaosnal change by dividing it by relative humidity

        #open the RH values
        df_RH = pd.read_csv(self.RHpath, header = 0)
        df_RH = df_RH[['month', self.citychk]]
        df_RH['AvgHumidity'] = df_RH[self.citychk]
        df_RH = df_RH[['month', 'AvgHumidity']]

        #merge values by common date
        df_y_monthly = pd.merge(y_monthly, df_RH, how = 'left', on = ['month'])
        df_y_monthly['AvgHumidity'] = df_y_monthly['AvgHumidity'] / 100.0

        df_y_monthly['beta'] = df_y_monthly['AvgHumidity']

        #calling coefficient factor
        df_y_monthly['beta'] = df_y_monthly['beta'].apply(lambda x: get_betaRH(x*100))
        #df_y_monthly['beta'] = np.max(1, df_y_monthly['AvgHumidity']/.6)

        #now divide R by RH
        #simple method
        df_y_monthly['R'] = df_y_monthly['R']/(df_y_monthly['beta'])

        #method by Lowenthal (1995), Laulainen(1993)
        #df_y_monthly['R'] = df_y_monthly['R'] * (1+.25*(df_y_monthly['AvgHumidity'])**2/(1-df_y_monthly['AvgHumidity']))

        return df_y_monthly


    def getEI(self):
        #get the EI emission intensty of various sectors involved with respect to the year concerened

        #open the EI
        df_EI = pd.read_csv(self.EIpath, header = 0)

        #check the converned year
        return df_EI[df_EI.year == self.year].values.tolist()[0]



    def agro_firemonthyl(self):
        #count fire
        # open the csv
        df_agro = pd.read_csv(self.df_agrfirepath, header = 0)

        #set the time axis
        df_agro['date1'] = pd.to_datetime(df_agro['date'], format="%Y%m%d")
        df_agro['datei'] = pd.DatetimeIndex(df_agro.date1)
        df_agro['year'] = pd.DatetimeIndex(df_agro.date1).year
        #make date as index
        df_agro.set_index('date1', inplace=True)

        #mean by month
        fire_mean = df_agro[df_agro.year==self.year].groupby(pd.TimeGrouper(freq='M')).mean()

        #return the foire count for  specific city
        return fire_mean[self.Count].tolist()


    def proportionalVKT(self):
        #get proportional VKT corresponding to the pioxel in consideration

        #subset the road raster
        srs.rio_zone_mask( self.shppath , self.roadpath, 'temp.tif')

        #get coordinates
        # get the pix coordinate
        #u-- cancel--sing the ful road roaster because sometimes the shapefile cuts the necessary pixels on edge
        pix1 = getpixcoord(self.citychk_loc, 'temp.tif', self.citylist_path)

        #get the pixel value unde rthe given coordinates
        VKTsubsetarr = rio.open('temp.tif').read(1)

        #total length of road
        totalroad = np.sum(VKTsubsetarr)

        #length of road in that pixel
        pixroad = VKTsubsetarr[pix1[1], pix1[0]]

        #get the VKT for citychk
        df_VKT = pd.read_csv(self.df_VKTpath)

        #correponsing VKT calue
        VKT = df_VKT[df_VKT.city==self.citychk].values.tolist()[0][1]

        #return rpoportional value (divide 12 to make it monthyl)
        VKTprop = VKT*pixroad/totalroad/12

        return VKTprop


    def get_UMvalue(self, distthresh=1000):

        # open the image as array
        Umarr = rio.open(self.UMpath).read(1)

        # get distance array for resindetial/commercial type sources and industria sources
        distarrRC = distarray_residential(self.citychk_loc, self.UMpath, self.citylist_path)
        #distarrI = distarray_industrial(self.citychk_loc, self.UMpath, self.citylist_path)
        Umarr[distarrRC >= distthresh] = np.nan
        #distarrRC[distarrRC>=distthresh]= np.nan
        #distarrI[distarrI >= distthresh] = np.nan


        # get weighted distance,
        # remove zero distance, ie.e UM pixels within MODIS pixel as well
        # we chose pixels location in the center of modis pixel. so now we weill consider all pixels within 5km radius
        # however this will exclude pixels beyond 5km lying on the diagonal of the square piel. hence we shall consider pxixel within 7km
        # there 7km= 7000/30 pixels as resolution = 30m

        distarrRC = np.where(distarrRC <= 7, 1, distarrRC - 7 + 1)

        # residential
        dAUM1 = weighted_distance(np.ma.masked_where(Umarr == 1, Umarr).mask, distarrRC)/np.ma.masked_where(Umarr == 1, Umarr).mask.sum()
        # commercial
        dAUM2 = weighted_distance(np.ma.masked_where(Umarr == 2, Umarr).mask, distarrRC)/np.ma.masked_where(Umarr == 2, Umarr).mask.sum()
        # industrial
        dAUM3 = weighted_distance(np.ma.masked_where(Umarr == 3, Umarr).mask, distarrRC)/np.ma.masked_where(Umarr == 3, Umarr).mask.sum()

        #funciton to getthe estimated UM areas

        #brute force funxtion for Knaour
        df_UMestimate = pd.read_csv(self.UMestimates, header = 0)


        #open the image as array
        #Umarr = df_UMestimate[self.year-2000]

        #get distances
        #distarr = distarray(citychk, UMpath)

        if self.year in [2001,2011]:
            Umarr0=np.array([np.sum(Umarr==1), np.sum(Umarr==2), np.sum(Umarr==3)])*900.0/1000000.0
        else:
            Umarr0 = df_UMestimate[(df_UMestimate.year == self.year) & (df_UMestimate.city == self.citychk)][
                ['R', 'C', 'I']].values.tolist()[0]

        # get weighted distance,
        #  historically - (dAUM1 - 0.1957926696851979, dAUM2 - 0.34863134932844608, dAUM3 - 0.18259306489751581)
        AUM1 = dAUM1*(Umarr0[0])/900*1000000 # 18 caould be added because for tier 2 cities model made on R10
        AUM2 = dAUM2*Umarr0[1]/900*1000000
        AUM3 = dAUM3*Umarr0[2]/900*1000000

        return [AUM1, AUM2, AUM3]

    def getBK(self):
        #function to assign the BK weight (includes count BK inversely weighted by distance) depending on the city
        df_BK = pd.read_csv(self.BKestimatepath, header = 0)
        return df_BK[df_BK.city==self.citychk]['BKweight'].values[0]

    def get_designmatrix(self, distthresh=1000):
        #get the design matrix A (activity multiplied by values)

        #get the weighted distance of different land sue class pixels
        [AUM1, AUM2, AUM3] = self.get_UMvalue(distthresh=distthresh)

        #remove negative AUM values. e.g. in case of gwalior , dehradun etc where are results were poor.
        if AUM1<=0:
            AUM1=0
        if AUM2<=0:
            AUM2=0
        if AUM3<=0:
            AUM3=0

        #for agro fire
        Fire = self.agro_firemonthyl()

        #for brick kiln. assuming 1300 BK near delhi, where 50% at distance of 23 km from centr, 25% AT DIST OF 36KM AND 25% t dist of 42km at distance of
        #ABK = BK[citychk]
        ABK= self.getBK()

        #get VKTprop
        VKTprop = self.proportionalVKT()

        #for vehicle
        df_vehicle = pd.read_csv(self.vehiclestimate, header = 0)
        AV = VKTprop* df_vehicle[df_vehicle.year==self.year][self.citychk].values.tolist()[0]

        #get the emission intensity
        EI = self.getEI()
        #EI = [1,1,1,1,1,1,1,1]

        #generate design matrix
        A = np.ones([7,12])

        #res
        A[0] = self.EA[0]*AUM1*EI[3]
        #comm
        A[1] = self.EA[1] * AUM2*EI[3]
        #indus
        A[2] = self.EA[2] * AUM3*EI[2]
        #fire
        A[3] = self.EA[3] * Fire
        #BK
        A[4] = self.EA[4] * ABK
        #Vehicle
        A[5] = self.EA[5] * AV*EI[4]

        #get transpose to make compatible
        A = A.transpose()

        return A

    def get_designmatrixUM(self,distthresh=1000):
        #get the design matrix A (activity multiplied by values) for only R, C, I and vehicle. to get their contribution in AQ

        #get the weighted distance of different land sue class pixels
        [AUM1, AUM2, AUM3] = self.get_UMvalue(distthresh=distthresh)

        #get VKTprop
        VKTprop = self.proportionalVKT()

        #for vehicle
        df_vehicle = pd.read_csv(self.vehiclestimate, header = 0)
        AV = VKTprop* df_vehicle[df_vehicle.year==self.year][self.citychk].values.tolist()[0]

        #get the emission intensity
        EI = self.getEI()
        #EI = [1,1,1,1,1,1,1,1]

        #generate design matrix
        A = np.ones([7,12])

        A[0] = self.EA[0]*AUM1*EI[3]
        #comm
        A[1] = self.EA[1] * AUM2*EI[3]
        #indus
        A[2] = self.EA[2] * AUM3*EI[2]
        #fire
        A[3] = self.EA[3] * 0.0
        #BK
        A[4] = self.EA[4] * 0.0
        #Vehicle
        A[5] = self.EA[5] * AV*EI[4]

        #get transpose to make compatible
        A = A.transpose()

        return A

    def get_dfAY(self):
        # combine the Y and A into into a same dataframe

        #set the design matrix
        A = self.get_designmatrix()
        df_A = pd.DataFrame(data=A, columns=['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const'])

        #set the observed dependent R
        Y = self.get_R(RHcorrect=True)

        #merge the dataframes
        AY = Y.join(df_A)

        AY['Count'] = self.Count
        AY['city'] = self.citychk
        AY['citychk_loc'] = self.citychk_loc

        self.AY = AY

        return AY




#run the class


# trying the beta on another city Kanpur
# sample run

#objkpr00 = AQmodel('C09', 'Kanpur', 'Kanpur', year, daily=False)
#A1 = objkpr00.get_designmatrix()
#Y1 = objkpr00.get_R()
#lrkpr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
#lrkpr.fit(A1, Y1)
#lrkpr.predict(A1)
#lrkpr.coef_
#lrkpr.score()










