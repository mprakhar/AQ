#__author__ = 'Prakhar MISRA'
# Created 9/18/2017
#Last edit 9/18/2017

# Purpose:
# (1)  -  get TNL for 20 cities in 1999-2013
# (2)  -  get TNL for all states in 1999 - 2013








#improt
import fiona as fio
import rasterio as rio
import numpy as np
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
from rasterstats import zonal_stats

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

city_shape_path = gb_path + '/Data/Data_raw/Shapefiles/maps-master/Districts/Census_2011/2011_Dist.shp'
state_shape_path = gb_path + '/Data/Data_raw/Shapefiles/maps-master/States/Admin2.shp'

ols = Raster_file()
ols.sat = 'DMSP'
ols.prod = 'OLS'
ols.path = gb_path + '/Data/Data_process/DMSPInterannualCalibrated_20160512/Wu/'
ols.sample = gb_path + r'/Data/Data_raw/DMSP OLS//' + 'F182010.v4/F182010.v4d_web.stable_lights.avg_vis.tif'
ols.georef = gb_path + 'Data/Data_process/Georef_img//DMSP_georef.tif'

dict_cityid = {
    'C01': 1,
    'C02': 2,
    'C03': 10,
    'C04': 17,
    'C17': 46,
    'C05': 110,
    'C20': 139,
    'C06': 184,
    'C07': 213,
    'C18': 231,
    'C19': 239,
    'C08': 261,
    'C09': 279,
    'C10': 313,
    'C11': 344,
    'C12': 345,
    'C13': 381,
    'C14': 409,
    'C15': 438,
    'C16': 462
}

# for all the states in india
dict_stateid = np.arange(36)

#Output
plt_save_path = gb_path + r'/Codes/PlotOut//'  # fig plot output path
csv_save_path = gb_path + r'Codes/CSVOut//AQ_NL/'  # cas output path
exl_path = gb_path + r'/Docs prepared/Excel files//'  # excel saved files read path
img_save_path = gb_path + r'/Data/Data_process//'
csv_in_path = gb_path + '/Codes/CSVIn/'
# ----------------------------------------------------------





polygon_id = []

def get_imgstat(polygon_path, image_path, dict_id, property = 'DISTRICT'):
    # function to get zonal stat for all polygons in the imagepath

    #open shapefile
    polygons = fio.open(polygon_path)
    #features = [feature['geometry'] for feature in polygons]

    ls = []
    id = []

    if property == 'DISTRICT':
        fulllist = dict_id.values()
    if property == 'ST_NM':
        fulllist = dict_stateid

    for shapeid in fulllist:

        city = polygons[shapeid]['properties'][property]
        id.append(city)
        print ' doing ', city

        #get its shapefiles
        features = polygons[shapeid]

        # the zonal stat
        ls.append(zonal_stats(features, image_path, stats = ['sum'])[0]['sum'])

    return id, ls


def get_TNL(ols):

    #get data for cities and state
    df_city = pd.DataFrame()
    df_state = pd.DataFrame()

    #run all images
    for file in glob(os.path.join(ols.path, '*'+'.tif')):
        # get year name
        year = file[-8:-4]

        #run fucntion city
        idc, listc = get_imgstat(city_shape_path, file, dict_cityid, property = 'DISTRICT' )
        df_city[year] = listc

        #run function state
        ids, lists = get_imgstat(state_shape_path, file, dict_stateid, property = 'ST_NM' )
        df_state[year] = lists

    # also assign id
    df_city['id'] = idc
    df_state['id'] = ids

    #save the df
    df_city.to_csv(csv_save_path+'20cityTNL.csv', header = True)
    df_state.to_csv(csv_save_path + 'allstateTNL.csv', header=True)

