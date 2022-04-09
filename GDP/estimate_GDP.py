#__author__ = 'Prakhar'
# Created 8/18/2016
#Last edit 8/18/2016

# Purpose:
# (1) - to extrapolate GDP for missing years for cities using TNL
# (2) - to estimate sector wise GDP for district using state level sector wise estimates - NA
# (3) - to estimate GDP from 1999Current values to 2004CurrentValues - NA
# (4) - to normalize GDP each state sector and find PCA i All, II GCT


import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from glob import glob

#Pvt imports
import shp_rstr_stat as srs
import pandas as pd
from scipy import stats
import my_plot as mpt
from classRaster import Raster_file
from classRaster import Image_arr
import infoFinder as info


# OUTPUT Info
plt_save_path = r'/home/prakhar/Research/AQM_research/Codes/PlotOut//'  # fig plot output path
csv_save_path = r'/home/prakhar/Research/AQM_research/Codes/CSVOut//'  # cas output path
exl_path = r'/home/prakhar/Research/AQM_research/Docs/Excel//'  # excel saved files read path

# Input info
# Task 1
shapefile_path = r'/home/prakhar/Research/AQM_research/Data/Data_process/Shapefiles/Ind_adm2_splitshp//'  #IND_adm3_ID_3__299.shp  # the place all the split shapefiels are stored
df_shpatt = pd.read_csv( shapefile_path + 'IND_adm2.csv', header=0)  # List of all shape file. making dataframe of shape file attribute list
eco_dst = r'/home/prakhar/Research/AQM_research/Data/Data_process/Economics/District_level_GDP//'


# Objects
ols = Raster_file()
ols.path = r'/home/prakhar/Research/AQM_research/Data/Data_process/DMSPInterannualCalibrated_20160512//'
ols.sat = 'DMSP'
ols.prod = 'OLS'
ols.sample = ols.path + 'F182011.tif'
ols.georef = '/home/prakhar/Research/AQM_research/Data/Data_process/Georef_img//DMSP_georef.tif'

# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * *  Task I Extrapolate missing GDPs  *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# (1) Creating database of TNL for all years at district level

# function to calculate stats for a given admin level for all shapefi;es for a gieven image
def all_stt(input_value_raster, shapefile_path,df_shpatt, out_file, ID, varsfx):  # full india scorecard :)
    # input_value_raster
    # shapefile_path - dir of all the shape file
    # df_shpatt - List of all shape file. making dataframe of shape file attribute list
    # out_file - output name
    # ID -  shapefile feature which needs to statisticed
    # varsfx - any suffix after new variable added

    ind2 = []
    dl_sum=[]
    dl_mean=[]
    dl_med=[]
    dl_std=[]
    sd=os.path.normpath(input_value_raster)

    #opening each vector and storing their info
    for vector in info.findVectors(shapefile_path, '*.shp'):
        (vinfilepath, vinfilename)= os.path.split (vector)
        input_zone_polygon = vinfilepath+ '/' + vinfilename
        sd=os.path.normpath(vector)
        id, (sum, mean, med, std)  = list(srs.loop_zonal_stats(input_zone_polygon, input_value_raster, ID))

        ind2.append(id)
        dl_sum.append(sum)
        dl_mean.append(mean)
        dl_med.append(med)
        dl_std.append(std)


    d = {ID:ind2, 'sum'+varsfx:dl_sum,'mean'+varsfx:dl_mean}#,'med'+varsfx:dl_med,'std'+varsfx:dl_std }
    df = pd.DataFrame(d)

    return df
    # df.to_csv(csv_save_path+'df_ShpStat_'+out_file, index=False, header=False)
# function end


for ols.file in glob(os.path.join(ols.path, 'F'+'*'+'.tif')) :
    year = ols.file[-8:-4]
    print year
    df_tnl = all_stt(ols.file, shapefile_path,df_shpatt,out_file= 'TNL_OLS', ID='ID_2', varsfx=year)
    # Checking if the cooresponding csv file fror city exists or not and creating it if not
    fname = csv_save_path + 'df_TNL_district.csv'
    if os.path.isfile(fname) == False:
        df_tnl.to_csv(fname, index=False, header=True)
        print 'opening csv '
        print 'processing  ' + year

    else:
        print 'appending '
        print 'processing  ' + year
        f = pd.read_csv(fname, header=0)
        df_tnl = pd.concat([f,df_tnl], axis =1)
        df_tnl.to_csv(fname, index=False, header=True)


# (2) Reading files of economic data and TNL

arr_state=df_shpatt['NAME_1'].unique()

state = 'Uttar Pradesh'

arr_cities = df_shpatt[df_shpatt['NAME_1']==state].NAME_2.tolist()
df_stateld = pd.read_csv(eco_dst + 'Districtwise_GDP_and_growth_rate_based_at_current_price_2004-05_' + state + '.csv', header=0) # state at level district GDP
df_disttnl = pd.read_csv(csv_save_path + 'df_TNL_district.csv', header=0)       # database of all city all TNL
df_stateld['yyyy'] = df_stateld['Year'].str.split('-').str.get(0).apply(int)

# (2.1) Creating regression eqn
ls=[['m','b', 'rvalue','pvalue', 'stderr' ]]
ls=[]

for city in arr_cities:
    city_id = int(df_shpatt[(df_shpatt['NAME_1']==state)&(df_shpatt['NAME_2']==city)]['ID_2'])
    city_gdp = df_stateld[city]
    city_tnl = []
    print city
    for year in df_stateld['yyyy']:
        city_tnl.append(float(df_disttnl[df_disttnl['ID_2']==int(city_id)]['sum'+str(year)]))
    x=(city_tnl)
    y=(city_gdp)
    [m, b, r_value, p_value, std_err] = stats.linregress(x,y)
    ls.append([city_id, city, mean(x), mean(y), m, b, r_value, p_value, std_err])



    # plt.figure()
    # plt.plot((city_tnl), (city_gdp, 'ko', label='')
    # plt.plot(city_tnl,map(lambda x: x*m+b, city_tnl) , 'b-', label='Best fit: y='+str('%2f'%m)+'x+'+str('%2f'%b) ,  alpha=.5)
    # # plt.plot(x, func(x, *popt), 'r-', label='line fit')
    # plt.legend()
    # plt.xlabel('TNL (unit ???)', fontsize=16)
    # plt.ylabel('GDP (in INR crore)', fontsize=16)
    # plt.title('TNL and GDP regression for '+city+' ('+str(min(df_stateld['yyyy']))+':'+str(max(df_stateld['yyyy']))+')', fontsize=20)
    # plt.show()

    # Plotting
    fig, ax =plt.subplots()
    ax.plot(x, y, 'ko', label='R$^2$ value = '+str('%.2f'%r_value), alpha=.8)
    ax.plot(x,map(lambda z: z*m+b, x) , 'r--', label='Best fit: y='+str('%.2f'%m)+'x+'+str('%.2f'%b)  ,  alpha=.8)
    # plt.plot(x, func(x, *popt), 'r-', label='line fit')
    ax.legend()
    plt.xlabel('TNL (unit DN)', fontsize=16)
    plt.ylabel('GDP (in INR crore)', fontsize=16)
    plt.title('TNL and GDP regression for '+city+' ('+str(min(df_stateld['yyyy']))+':'+str(max(df_stateld['yyyy']))+')', fontsize=20)
    for i,txt in enumerate(df_stateld['yyyy']):
        ax.annotate(txt, (x[i], y[i]) , alpha=.8 , xytext=(x[i]+200, y[i]-100))
    # plt.show()
    plt.savefig(plt_save_path+'GDP_TNL_'+city+'.png')

df_cityGDP_reg=pd.DataFrame(ls, columns=['ID_2', 'city', 'meanTNL', 'meanGDP', 'm','b', 'rvalue','pvalue', 'stderr'])         #strores regression coefficients for each city
df_cityGDP_reg.to_csv(csv_save_path+'df_UttarPradeshcityGDP_TNL.csv')

#some more plotting
plt.scatter(df_cityGDP_reg['meanTNL'], df_cityGDP_reg['meanGDP'] )
plt.xlabel('mean TNL', fontsize=16)
plt.ylabel('meanGDP', fontsize=16)
plt.title('TNL cannot represent high GDP districts accurately', fontsize=20)

# plt.ylabel('R$^2$', fontsize=16)
# plt.title('With high GDP, R$^2$ of TNL&GDP decreases (Uttar Pradesh)', fontsize=20)

# (3) Extrapolating missing GDP from 2001 to 2012 will amke it 2015 after completing TNLintra calibration
df_cityreg = pd.read_csv(csv_save_path+'df_cityGDP_reg.csv', header=0)
yr_mn = min(df_stateld['yyyy'])     # starting year of actual district GDP available
yr_mx = max(df_stateld['yyyy'])     # last year of actual district GDP available
yr_stmn = 2001      # starting year of NLimage available
yr_stmx = 2012      # ending year of NLimage available
fname = csv_save_path + 'df_UttarPradesh_city_extGDP.csv'
df_city_gdp = pd.DataFrame(range(yr_stmn, yr_stmx+1), columns=['year'])
for city in arr_cities:
    city_id = int(df_shpatt[(df_shpatt['NAME_1'] == state) & (df_shpatt['NAME_2'] == city)]['ID_2'])
    ls_city_gdp=[]
    for yr in range(int(yr_stmn),int(yr_stmx)+1):
        if yr in range(int(yr_mn), int(yr_mx)+1):
            city_gdp = df_stateld[df_stateld['yyyy']==yr][city].iloc[0]
        else:
            city_gdp = df_cityreg[df_cityreg['ID_2']==city_id]['m'].iloc[0]* df_disttnl[df_disttnl['ID_2']==int(city_id)]['sum'+str(yr)].iloc[0] + df_cityreg[df_cityreg['ID_2']==city_id]['b'].iloc[0]
        print city_gdp
        ls_city_gdp.append(city_gdp)
    df_city_gdp[city] = ls_city_gdp
    df_city_gdp.to_csv(fname, index=False, header=True)


 # strores regression coefficients for each city
df_cityGDP_reg.to_csv(csv_save_path + 'df_cityGDP_reg.csv')



# (3) Least square coefficient estimation
b = np.genfromtxt(r'D:\Obsmtrx.csv',delimiter=',')
A = np.genfromtxt(r'D:\Dzinmtrx1.csv',delimiter=',')

res = stats.lsq_linear(A, b, bounds=(0.01, 0.99), lsmr_tol='auto', verbose=1)




# * * * *  * * # * * * *  * * # * * * *  * *# # * * * *  Task III stimate GDP from 1999Current values to 2004CurrentValues  # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#




