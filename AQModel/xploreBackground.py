#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created 02/10/2017
# Last edit 02/10/2017

#

# Purpose: to investigate the effect of radius upto which UM types are considered on their contribution to urban AirRGB R wrt background
# absed on toy_model_run_v2



import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import pandas as pd
import os
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
#pvt import
import toy_modelv5
#reload(toy_modelv5)
from toy_modelv5xploreBackground import AQmodel
import SEAmodifications as SEA
#get present dirctory
pwd = os.getcwd()



# ----------------------------------------------------------------------------------------

#---------------------------------------------------
# -------   MODEL Trials over KanpurLucknow ----------------
#---------------------------------------------------
gb_path = r'/home/prakhar/Research/AQM_research//'  # global path tp be appended to each path

#prepare beta over Kanpur
def get_Countid(citychk):
    df_citylist = pd.read_csv(pwd+r'/lib/20citycoordforAQmodel.csv', header=0)
    return df_citylist[df_citylist.city_coord==citychk]['Count'].values[0]

def get_neighborpixels(citychk):
    df_citylist = pd.read_csv(pwd + r'/lib/20citycoordforAQmodel.csv', header=0)
    neighbors = df_citylist[df_citylist.city == citychk]['city_coord'].values.tolist()
    return neighbors

def remove_monsoon(AY):
    return AY[-AY['month'].isin([7,8])]

def replace_monsoon(AY):

    replacementvalue = AY[AY['month'].isin([6,9])]['R'].mean()
    #AY[AY['month'].isin([7,8,9])]['R']=replacementvalue
    #set by index. = month -1
    AY.at[6, 'R'] = replacementvalue
    AY.at[7, 'R'] = replacementvalue
    #AY.at[8, 'R'] = replacementvalue
    return AY


#prepare deisgn matric and depebedndet
def getAY(year, citychk, RHcorrect = True, onlyone = True, UM = False, EA = 0, replacemonsoon = False):

    #function to generate dataframe A, Y for a given year in Kanpur and it s4 pixels

    if onlyone:
        objkpr00 = AQmodel(get_Countid(citychk), citychk, citychk, year, daily=False, EA = EA, ASTER = True)
        AY = objkpr00.get_dfAY()

        if replacemonsoon:
            AY = replace_monsoon(AY)
        return AY


    else:
        neighbors = get_neighborpixels(citychk)
        ls_AY = []
        for citychk_loc in neighbors:
            #try:
            objkpr00 = AQmodel(get_Countid(citychk), citychk, citychk_loc, year, daily = False, EA = EA, ASTER = True )
            AY1 = objkpr00.get_dfAY()

            if replacemonsoon:
                AY1 = replace_monsoon(AY1)

            ls_AY.append(AY1)
            #except TypeError:
            #    print 'coord conversion error in ', citychk_loc

        # append all the df
        AY = pd.concat(ls_AY, ignore_index=True)

        return AY


#define the dependent and indepenednt variables
ind_var = ['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const']
dep_var = ['R']

#set seasonal emission activity to be sued
#EAlist = [SEA.EA, SEA.EA_1_1, SEA.EA_1_2, SEA.EA_1_3 ,  SEA.EA_2_1 , SEA.EA_2_2 , SEA.EA_2_2 , SEA.EA_3_1 , SEA.EA_123_1 ,SEA.EA_123_2, SEA.EA_123_3]
#EA = EAlist[9]
EAlist = SEA.EAall
EA = np.array(EAlist[13])

EA = SEA.EA_templateff(s=0.6, r=0.4)

# --------------------------------------------------------------------------

# R U N
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Making AY UM and vehicle for the for the KanpurLucknow
def getA(citychk='Kanpur', year= 2011):

    savefilepathASTERUM = r"/home/prakhar/Research/AQM_research/Data/Data_process/ASTER/India/KanpurLucknow/KanpurLucknowDSM_UM.tif"
    savefilepathAW3DUM = r"/home/prakhar/Research/AQM_research/Data/Data_process/AW3D/India/KanpurLucknow/KanpurLucknow_UMth4.tif"

    objkpr00 = AQmodel(get_Countid(citychk), citychk, citychk, year, daily=False, EA = EA, ASTER = True)

    objkpr00.shppath = gb_path + r'/Data/Data_process/Shapefiles/India_20citydistrict/2011_Dist_DISTRICT_' + citychk + '.shp'
    objkpr00.UMpath = savefilepathAW3DUM
    if year == 2001:
        objkpr00.UMpath = savefilepathASTERUM

    df_A10 = pd.DataFrame(data=objkpr00.get_designmatrix(distthresh=10), columns=['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const'])
    df_A10['dist']=10
    df_A30 = pd.DataFrame(data=objkpr00.get_designmatrix(distthresh=30), columns=['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const'])
    df_A30['dist']=30
    df_A50 = pd.DataFrame(data=objkpr00.get_designmatrix(distthresh=50), columns=['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const'])
    df_A50['dist']=50
    df_A100 = pd.DataFrame(data=objkpr00.get_designmatrix(distthresh=100), columns=['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const'])
    df_A100['dist']=100
    A = pd.concat([df_A10,df_A30, df_A50, df_A100])  # , A04, A07, A09))#, A11a, A01a))
    A['year']=year

    return A


A2011=getA(citychk='Kanpur', year= 2011)
A2001=getA(citychk='Kanpur', year= 2001)

A=pd.concat([A2011, A2001])
A.to_csv('AKanpurLucknow20180210.csv', header = True)



#generate design matrix
A = np.ones([4,12])
#get the emission intensity
EI = objkpr00.getEI()
# get VKTprop
VKTprop = objkpr00.proportionalVKT()
# for vehicle
df_vehicle = pd.read_csv(objkpr00.vehiclestimate, header=0)
AV = VKTprop * df_vehicle[df_vehicle.year == year][citychk].values.tolist()[0]

[AUM1, AUM2, AUM3] = objkpr00.get_UMvalue()
# res
A[0] = objkpr00.EA[0] * AUM1 * EI[3]
# comm
A[1] = objkpr00.EA[1] * AUM2 * EI[3]
# indus
A[2] = objkpr00.EA[2] * AUM3 * EI[2]
# Vehicle
A[3] = objkpr00.EA[5] * AV * EI[4]

# ------------------------------------------------------------------------------------------------
# Experiment 1 - combine with infor of BK and crop fire

# make model using the 2001 and 2011 year, cityid, citychk
AY11 = getAY(2011, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True,)
AY01 = getAY(2001, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True ,)


AYl11 = getAY(2011, 'Lucknow', onlyone=False, EA=EA, replacemonsoon = True, )
AYl01 = getAY(2001, 'Lucknow', onlyone=False, EA=EA, replacemonsoon = True, )


AY = pd.concat([AY01,AY11, AYl11, AYl01], ignore_index=True)  # , A04, A07, A09))#, A11a, A01a))



