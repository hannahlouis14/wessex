import cf_xarray as cfxr
import shapely
import xarray as xr
import xesmf as xe
import numpy as np
import glob
import netCDF4 as nc
import os.path
import pandas as pd
import datetime



def get_lonlat_file(data_path, name):
    lonslats = nc.Dataset(data_path+'air_2m_'+name+'_y2010.nc')
    lons = lonslats['air'][:] #['LON']
    print(lonslats)
    print(lonslats)
    lats = lonslats['LAT']

    lat = np.array(np.meshgrid(lats, lons)[0])
    lon = np.meshgrid(lons, lats)[0]
    lat = np.transpose(lat)

    df = xr.Dataset({'Latitudes': xr.DataArray(data=lat, dims=['lat', 'lon']), 'Longitudes': xr.DataArray(data=lon, dims=['lat', 'lon'])},)
    df.to_netcdf('NCEP_R2_LatitudesLongitudesGrid.nc',format='NETCDF4')


#get_lonlat_file(data_path='/mnt/storage5/hlouis/NCEP_R2/', name='gauss')


def regrid_data(year, var, var_name, leapyear=False):
    output_path = '/mnt/storage6/hlouis/DATA/AtmForcing/NCEP_R2/' 
    data_path = '/mnt/storage5/hlouis/NCEP_R2/'
    output_grid = '/mnt/storage4/tahya/model_files/ANHA4_mesh_mask.nc'
    #CGRF_data_path = '/mnt/storage3/xhu/ANHA_INPUT/CGRF/'
    #CGRF_data_path = '/mnt/storage5/hlouis/'
    '''
    lonslats = nc.Dataset('CORE2_LatitudesLongitudesGrid.nc')
    lat = lonslats['Latitudes']
    lon = lonslats['Longitudes']
    '''
    ds_out = xr.open_dataset(output_grid)
    ds_out = ds_out.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})
    ds_out = ds_out[['lon','lat']]

    ds_in = xr.open_dataset(data_path+var+'_gauss_y'+str(year)+'.nc')  # change to data_path for 2002-2016
    #ds_in = ds_in.rename({'LON': 'lon', 'LAT': 'lat', 'TIME': 'time_counter'})
    ds_in = ds_in.rename({'time': 'time_counter'})
    ds_in = ds_in.isel(time_counter=slice(None, -4))
    print(ds_in) 
    time = ds_in.time_counter[:-4].values
    #ds_in['time'] = time 

    if leapyear==False:
        rng = pd.date_range("01 Jan "+ str(year)+" 00:00", "30 Dec "+str(year)+" 23:00", freq='6H')
    
    else:
        rng = pd.date_range("01 Jan "+ str(year)+" 00:00", "30 Dec "+str(year)+" 23:00", freq='6H')
    
    df = pd.DataFrame({'a':ds_in['time_counter'].values}, index=rng)
    
    ds_in['time_counter'] = rng
    ds_in = ds_in.resample(time_counter='120h', label='right').mean()


    variable = ds_in[var_name]

    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', extrap_method='inverse_dist', reuse_weights=True, filename='NCEP_bilinear_94x192_800x544.nc')
    var_out = regridder(ds_in[var_name])
    var_out = var_out.rename(var_name)
    
    var_out.to_netcdf(output_path+var+'_gauss_ANHA4_y'+str(year)+'.nc')


def run_regrid(forcing, leap=False):

    if forcing == 'CORE':
        years = np.arange(1958, 1960, 1)
        nemo_path = '/mnt/storage6/pmyers/'
        data_path = '/mnt/storage3/xhu/ANHA_INPUT/ANHA4-I/CORE2-IA/'

        
        AirHeight=10
        WindHeight=10
        precipmult=1
        SnowFlag=0
        
        precip_file='prc_core2_y'
        precip_var='prc'

        temp_file = 't10_core2_y'
        temp_var = 't10'

        humid_file = 'q10_core2_y'
        humid_var = 'q10'

        u_wind_file = 'u10_core2_y'
        u_wind_var = 'u10'

        v_wind_file = 'v10_core2_y'
        v_wind_var = 'v10'

    if forcing == 'NCEP':
        years = np.arange(2002, 2010, 1)
        nemo_path = '/mnt/storage6/myers/'
        data_path = '/mnt/storage5/hlouis/NCEP_R2/'
        
        AirHeight = 2
        WindHeight = 10 

        temp_file = 'air_2m_gauss_y'
        temp_var = 'air'

        humid_file = 'shum_2m_gauss_y'
        humid_var = 'shum'
        
        u_wind_file = 'uwnd_10m_gauss_y'
        u_wind_var = 'uwnd'

        v_wind_file = 'vwnd_10m_gauss_y'
        v_wind_var = 'vwnd'


    leap_arr = []
    year_arr = []

    for element in years:
        if element % 4 == 0:
            leap_arr.append(element)
        else:
            year_arr.append(element)
    
    #leaps = [year for year in range(1960, 2010) if year % 4 == 0]
    #years = [x for x in years if not any(word in x for word in str(leaps))]
    #years = np.delete(years, leaps) 
    #leap_arr.pop(0)
    #leap_arr.pop(0)
    print(leap_arr)
    # non leap years first
    if leap==False:
        for yr in year_arr:
            # temp regrid
            regrid_data(year=yr, var=temp_var+'_2m', var_name=temp_var, leapyear=False) 
            print('regridded temp for year '+str(yr))
            '''            
            # spec. humidity regrid
            regrid_data(year=yr, var=humid_var+'_2m', var_name=humid_var, leapyear=False)
            print('regridded q for year '+str(yr))        
        
            # u10 regrid
            regrid_data(year=yr, var=u_wind_var+'_10m', var_name=u_wind_var, leapyear=False)
            print('regridded u10 for year '+str(yr))        
            
            # v10 regrid
            regrid_data(year=yr, var=v_wind_var+'_10m', var_name=v_wind_var, leapyear=False)
            print('regridded v10 for year '+str(yr))        
            '''
    # now regrid leap years
    else:
        for yr in leap_arr:
            # temp regrid
            regrid_data(year=yr, var=temp_var+'_2m', var_name=temp_var, leapyear=True) 
            print('regridded temp for leap year '+str(yr))
            ''' 
            # spec. humidity regrid
            regrid_data(year=yr, var=humid_var+'_2m', var_name=humid_var, leapyear=True)
            print('regridded q for leap year '+str(yr))        
        
            # u10 regrid
            regrid_data(year=yr, var=u_wind_var+'_10m', var_name=u_wind_var, leapyear=True)
            print('regridded u10 for leap year '+str(yr))        
            
            # v10 regrid
            regrid_data(year=yr, var=v_wind_var+'_10m', var_name=v_wind_var, leapyear=True)
            print('regridded v10 for leap year '+str(yr))        
            '''

  
run_regrid(forcing='NCEP', leap=False)
#run_regrid(forcing='CORE', leap=True)

#regrid_data(year=2022, var='vwnd_10m', var_name='vwnd', leapyear=False)
