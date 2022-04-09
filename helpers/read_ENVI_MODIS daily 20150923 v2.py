#__author__ = 'Prakhar'
# Created 9/23/2015
#Last edit 9/23/2015

# Purpose: This code will read the MODIS files that have aready been processed at daily level  *AOD AND ANG in this case) in batch mode. also it will give the values for select pixel coodridinates for the time series data.
# the purpose os not only to read the daily level provcessed files but also to find outliers in the data acquisition
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



# INPUTS
path = "F:\Data OMI MODIS\L3 MODIS\L3\\"
f_type = 'daily'

# CITY functions pertaining to city pixel locations
city_loc = (0,0)

def C1_Agra():
    city_loc_f = ( 755,3097)
    return city_loc_f
def C2_Ahmedabad():
    city_loc_f = ( 805,3031)
    return city_loc_f
def C3_Allahabad():
    city_loc_f = ( 776,3142)
    return city_loc_f
def C4_Amritsar():
    city_loc_f = ( 701,3059)
    return city_loc_f
def C5_Chennai():
    city_loc_f = ( 924,3123)
    return city_loc_f
def C6_Firozabad():
    city_loc_f = ( 754,3101)
    return city_loc_f
def C7_Gwalior():
    city_loc_f = ( 766,3099)
    return city_loc_f
def C8_Jodhpur():
    city_loc_f = ( 764,3037)
    return city_loc_f
def C9_Kanpur():
    city_loc_f = ( 764,3125)
    return city_loc_f
def C10_Kolkata():
    city_loc_f = ( 810,3221)
    return city_loc_f
def C11_Lucknow():
    city_loc_f = ( 759,3131)
    return city_loc_f
def C12_Ludhiana():
    city_loc_f = ( 710,3070)
    return city_loc_f
def C13_Mumbai():
    city_loc_f = ( 850,3035)
    return city_loc_f
def C14_New_Delhi():
    city_loc_f = ( 735,3085)
    return city_loc_f
def C15_Patna():
    city_loc_f = ( 773,3182)
    return city_loc_f
def C16_Raipur():
    city_loc_f = ( 826,3141)
    return city_loc_f


# map the inputs to the function blocks
options = { 1 : C1_Agra,
            2 : C2_Ahmedabad ,
            3 : C3_Allahabad ,
            4 : C4_Amritsar ,
            5 : C5_Chennai ,
            6 : C6_Firozabad ,
            7 : C7_Gwalior ,
            8 : C8_Jodhpur ,
            9 : C9_Kanpur ,
            10 : C10_Kolkata ,
            11 : C11_Lucknow ,
            12 : C12_Ludhiana ,
            13 : C13_Mumbai ,
            14 : C14_New_Delhi ,
            15 : C15_Patna ,
            16 : C16_Raipur
            }

'''
A code system to be used later on

1 : AGR: C1_Agra ,
2 : AMD: C2_Ahmedabad ,
3 : ALD: C3_Allahabad ,
4 : AMR: C4_Amritsar ,
5 : CHN: C5_Chennai ,
6 : FRZ: C6_Firozabad ,
7 : GWL: C7_Gwalior ,
8 : JDP: C8_Jodhpur ,
9 : KNP: C9_Kanpur ,
10 : KOL: C10_Kolkata ,
11 : LKO: C11_Lucknow ,
12 : LDH:C12_Ludhiana ,
13 : MUM: C13_Mumbai ,
14 : NDL: C14_New_Delhi ,
15 : PTN: C15_Patna ,
16 : RPR: C16_Raipur

'''

# all the AOD and ANG files names are read as follows
filename_AOD = glob(os.path.join(path, '*.AOD.Global'))
filename_ANG = glob(os.path.join(path, '*.ANG.Global'))

#total files of each kind
num_files = len(filename_AOD)


# AOD, ANG processing outer body begin dl = data list
dl_AOD = [['yearjd', 'year', 'jd',  'C1_Agra', 'C2_Ahmedabad' , 'C3_Allahabad'  , 'C4_Amritsar' , 'C5_Chennai' , 'C6_Firozabad' , 'C7_Gwalior' , 'C8_Jodhpur' , 'C9_Kanpur' , 'C10_Kolkata' , 'C11_Lucknow' , 'C12_Ludhiana' ,
          'C13_Mumbai' , 'C14_New_Delhi' , 'C15_Patna' , 'C16_Raipur']]

dl_ANG = [['yearjd', 'year', 'jd',  'C1_Agra', 'C2_Ahmedabad' , 'C3_Allahabad'  , 'C4_Amritsar' , 'C5_Chennai' , 'C6_Firozabad' , 'C7_Gwalior' , 'C8_Jodhpur' , 'C9_Kanpur' , 'C10_Kolkata' , 'C11_Lucknow' , 'C12_Ludhiana' ,
          'C13_Mumbai' , 'C14_New_Delhi' , 'C15_Patna' , 'C16_Raipur']]

for i in range (0, num_files):
    print i
    # opening the files
    img_AOD = envi.open(filename_AOD[i]+".hdr", filename_AOD[i])
    img_ANG = envi.open(filename_ANG[i]+".hdr", filename_ANG[i])
    # exctracting the dates from the filenames: for MODIS filenames:
    # at daily level it should be- yearjd -18:-11, year -18:-14, jd -14:-11
    # at monthly level it should be- yearjd -17:-11, year -17:-13, jd -13:-11
    yearjd = int(filename_AOD[i][-18:-11])
    year = int(filename_AOD[i][-18:-14])
    jd = int(filename_AOD[i][-14:-11])    #jd is the julian date

    # VALUE MATRIX start reading values of the city into the matrix
    city_AODval=[yearjd , year, jd]
    city_ANGval=[yearjd , year, jd]

    for j in range(1,17):
        city_loc = options[j]()
        # -1 in subsequent statements because in ENVI coordinates start form 1,1 while in python they start from 0,0
        # the coordinates stored in functions above are those of ENVi images

        AODval = int(img_AOD[city_loc[0]-1,city_loc[1]-1])
        # print AODval, j
        city_AODval.insert(j+2,AODval)

        ANGval = int(img_ANG[city_loc[0]-1,city_loc[1]-1])
        city_ANGval.insert(j+2,ANGval)

    dl_AOD.append(city_AODval)
    dl_ANG.append(city_ANGval)

# converting the datalist to dataframe
df_AOD = pd.DataFrame(dl_AOD)
df_ANG = pd.DataFrame(dl_ANG)

df_AOD.to_csv('CSVOut\df_AODmaster_'+f_type+'.csv', index=False, header=False)
df_ANG.to_csv('CSVOut\df_ANGmaster_'+f_type+'.csv', index=False, header=False)

    # val=[[]]
    # for j in range(0,3):
    #     val.append(j)


#________________Performing operations on the data read to CSV _________________________________________________________

# reading from the CSV as a dataframe
dfaod = pd.read_csv('CSVOut\df_AODmaster_daily.csv', header=0)
dfang = pd.read_csv('CSVOut\df_ANGmaster_daily.csv', header=0)

dfaod['ymd'] = pd.to_datetime(dfaod['yearjd'], format='%Y%j')
dfaod['mon'] = dfaod['ymd'].dt.month

dfaod['season0'] = ['W' if x==1 or x==2 or x==11 or x==12 else
                    'S' if x==3 or x==4 or x==5 or x==6 else
                    'R' for x in dfaod['mon'] ]


dfaod[dfaod['season0']=='W']['C1_Agra']




