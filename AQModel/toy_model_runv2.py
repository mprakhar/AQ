#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'Prakhar MISRA'
# Created12/15/2017
# Last edit 12/16/2017

#

# Purpose: to run the code toy_modev3

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
from toy_modelv5 import AQmodel
import SEAmodifications as SEA
#get present dirctory
pwd = os.getcwd()

# ----------------------------------------------------------------------------------------

#---------------------------------------------------
# -------   MODEL Trials over Kanpur ----------------
#---------------------------------------------------
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

def makemodel(AY,removemonsoon = False):
    #function to make the model

    if removemonsoon:
        AY = remove_monsoon(AY)

        remove_monsoon(AY)

    lrkpr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
    lrkpr.fit(AY[ind_var], AY[dep_var])
    res = sm.OLS(AY[dep_var], AY[ind_var]).fit()
    #print "SCORE ", lrkpr.score(AY[ind_var], AY[dep_var])
    #print res.summary()
    lrkpr.predict(AY[ind_var])
    # get R2
    #

    #return lrkpr
    return lrkpr.score(AY[ind_var], AY[dep_var])

def makemodelp(AY):
    #function to make the model with positive coeff
    lrkpr = linear_model.Lasso(fit_intercept=False, normalize=True, positive=True)
    lrkpr.fit(AY[ind_var], AY[dep_var])
    print lrkpr.coef_

    lrkpr.predict(AY[ind_var])
    # get R2
    print "SCORE ", lrkpr.score(AY[ind_var], AY[dep_var])

    return lrkpr

#lrkpr = linear_model.Lasso(fit_intercept=False, normalize=True,positive = False)

def plotmodel(lrkpr, AY01):

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m\n%y'))
    #loc = plticker.MultipleLocator(base=1.0)
    #ax.xaxis.set_major_locator(loc)

    #set the datetime#
    #AY01['date'] = pd.to_datetime(AY01['date'], format="%Y-%m-%d")
    #AY01.set_index('date', inplace=True)
    AY01['pred'] = lrkpr.predict(AY01[ind_var])

    ax.plot(AY01.date, AY01[dep_var], label=' AirRGB-R observed')
    ax.plot(AY01.date, AY01['pred'], label=' AirRGB-R estimated')
    plt.legend()
    plt.show()

    # get correlation
    print 'correlation', AY01['pred'].corr(AY01['R'])

def fitall(lrkpr, EA = 0,  citychk='Kanpur', onlyone=True, plot= False ):
    # run the model over Kanpur from years 2002 to 2010

    # empty list for all df
    ls_AY = []

    for year in range(2001, 2016):
        AY05 = getAY(year, citychk, RHcorrect=True, onlyone=onlyone, UM=False, EA=EA)
        ls_AY.append(AY05)

    # append all the df
    AY0 = pd.concat(ls_AY, ignore_index=True)
    # AY0 = remove_monsoon(AY0)

    AY0 = replace_monsoon(AY0)

    if plot:
        # plot the result
        plotmodel(lrkpr, AY0)
    return AY0

def modelall(citylist, onlyone=False, EA=0):
    # empty list for all df
    ls_AY = []
    for citychk in citylist:
        for year in range(2001, 2016):
            AY05 = getAY(year, citychk, RHcorrect=True, onlyone=onlyone, UM=False, EA=EA)
            ls_AY.append(AY05)
    AY = pd.concat(ls_AY, ignore_index=True)  # , A04, A07, A09))#, A11a, A01a))
    lrkpr = makemodel(AY, removemonsoon=False)
    return lrkpr


# --------------------------------------------------------------------------

# run the get AY function

#define the dependent and indepenednt variables
ind_var = ['AR', 'AC', 'AI', 'AF', 'ABK', 'AV', 'const']
dep_var = ['R']

#set seasonal emission activity to be sued
#EAlist = [SEA.EA, SEA.EA_1_1, SEA.EA_1_2, SEA.EA_1_3 ,  SEA.EA_2_1 , SEA.EA_2_2 , SEA.EA_2_2 , SEA.EA_3_1 , SEA.EA_123_1 ,SEA.EA_123_2, SEA.EA_123_3]
#EA = EAlist[9]
EAlist = SEA.EAall
EA = np.array(EAlist[13])


# --------------------------------------------------------------------------

# R U N
# --------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Experiment 0 - General model buiding

# make model using the 2001 and 2011 year, cityid, citychk,
AY11 = getAY(2011, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True,)
AY01 = getAY(2001, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True,)
AY05 = getAY(2005, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True,)
AY10 = getAY(2010, 'Kanpur', onlyone=False, EA=EA, replacemonsoon = True,)


AYl11 = getAY(2011, 'Lucknow', onlyone=False, EA=EA, replacemonsoon = True, )
AYl01 = getAY(2001, 'Lucknow', onlyone=False, EA=EA, replacemonsoon = True, )
AYl05 = getAY(2005, 'Lucknow', onlyone=False, EA=EA, replacemonsoon = True, )


AYb11 = getAY(2011, 'Ahmedabad', onlyone=True, EA=EA, replacemonsoon = True)
AYb01 = getAY(2001, 'Bangalore', onlyone=True, EA=EA, replacemonsoon = True)
AYb05 = getAY(2006, 'Bangalore', onlyone=True, EA=EA, replacemonsoon = True)

AY = pd.concat([AYb01,AYb05, AYb11], ignore_index=True)  # , A04, A07, A09))#, A11a, A01a))
lrkpr = makemodel(AY, removemonsoon= False)


#makemodel using all years
lrkpr = modelall(['Kanpur'], onlyone=False, EA=EA)
#plotmodel(lrkpr, AY01)

AY0 = fitall(lrkpr, EA= EA, citychk='Bangalore')

# ------------------------------------------------------------------------------------------------
#get all data from cities
ls = []
for citychk in ind_city[]:
    EA = SEA.EA_template(s, r)
    AY0 = fitall(lrkpr, EA=EA, citychk=citychk)
    ls.append(AY0)
    print citychk
AYall = pd.concat(ls, ignore_index=True)
AYall.to_csv('allAY20180115.csv', header = True)
# ------------------------------------------------------------------------------------------------
# Experiment 1 - trying out different EA
i=0
for EA in EAlist:
    i += 1
    EA = np.array(EA)
    print " THIS IS EA NUMBER ", i
    #make model using the 2001 and 2011 year, cityid, citychk,
    AY11 = getAY(2011, 'Kanpur', onlyone = False, EA = EA )
    AY01 = getAY(2001,  'Kanpur', onlyone = False, EA = EA )
    #AY05 = getAY(2005,  'Kanpur',  onlyone = False, EA = EA )
    AYl11= getAY(2011, 'Chennai', onlyone = False, EA = EA )
    AYl01 = getAY(2001, 'Chennai',  onlyone = False, EA = EA )

    #combining all data
    AY = pd.concat([AYl11, AYl01], ignore_index=True)#, A04, A07, A09))#, A11a, A01a))

    #removing cloudmonths 7, 8,9
    #AY = remove_monsoon(AY)

    #replace monsoon months bu constant value of 30
    AY = replace_monsoon(AY)

    #make model
    lrkpr = makemodel(AY)

    #plotmodel(lrkpr, AY01)



#---------------------------------------------------

def SEAsimulation(citychk):
    # simulaiton doen with both fire underestimation correction off and on
    #[.5, .6, .7, .8, .9, .92, .94, .96, .98,  1]
    #fire correction off
    i = 0
    for s in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        for r in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            EA = SEA.EA_template (s, r)


            AY11 = getAY(2011, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY01 = getAY(2001, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY05 = getAY(2005, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY10 = getAY(2010, citychk, onlyone=False, EA=EA, replacemonsoon = True,)



            AY = pd.concat([AY01,AY05, AY10, AY11], ignore_index=True)  # , A04, A07, A09))#, A11a, A01a))
            print " MODEL ------  ", i, s, r, makemodel(AY, removemonsoon=False)
            #lrkpr = makemodel(AY, removemonsoon=False)
            i +=1

    #fire correction on (Venkataramn results)
    print "template 2"
    i = 0
    for s in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        for r in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            EA = SEA.EA_template2 (s, r)


            AY11 = getAY(2011, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY01 = getAY(2001, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY05 = getAY(2005, citychk, onlyone=False, EA=EA, replacemonsoon = True,)
            AY10 = getAY(2010, citychk, onlyone=False, EA=EA, replacemonsoon = True,)



            AY = pd.concat([AY01,AY05, AY10, AY11], ignore_index=True)  # , A04, A07, A09))#, A11a, A01a))
            print " MODEL ------  ", i, s, r, makemodel(AY, removemonsoon=False)
            #lrkpr = makemodel(AY, removemonsoon=False)
            i +=1

ind_city = ['Agra', 'Ahmedabad', 'Allahabad', 'Amritsar', 'Chennai', 'Firozabad', 'Gwalior', 'Jodhpur',
            'Kanpur', 'Lucknow', 'Ludhiana', 'Patna', 'Raipur', 'Hyderabad', 'Jaipur',
            'Bangalore', 'Kolkata', 'NewDelhi', 'Mumbai',]
            #'Dehradun',]

for citychk in ind_city:
    print  '  --------adfCITYNAMEsdf --------',  citychk
    SEAsimulation(citychk)





# ------------------------------------------------------------------------------------------------
#get all data from cities
ls = []

for citychk in ind_city[:]:

    if citychk=="Chennai":
        s = .7
        r = .6
    if citychk=="Bangalore":
        s = .9
        r = .9
    if citychk=="Kolkata":
        s = .8
        r = .8
    if citychk=="Hyderabad":
        s = .9
        r = .6
    if citychk=="Mumbai":
        s = .8
        r = .4
    if citychk=="Ahemdabad":
        s = .6
        r = .9

    #inclucidng Joshpur and Jaiupur for the new run # 20180122
    if citychk=="Jaipur":
        s = .9
        r = .5
    if citychk=="Jodhpur":
        s = .2
        r = .7
    else:
        s = .6
        r = .4

    #EA = SEA.EA_template(s, r)
    EA = SEA.EA_templateff(s, r)
    AY0 = fitall(lrkpr, EA=EA, citychk=citychk,  onlyone=False, plot= False )
    ls.append(AY0)
    print citychk

AYall = pd.concat(ls, ignore_index=True)
AYall.to_csv('allAY20180124ff1500km.csv', header = True)



