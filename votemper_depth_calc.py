'''
depth_calc.py
author: Tahya Weiss-Gibbons
Calculate the freshwater content for ANHA4 down to a set depth level, relative to 34.8
Output calculated values as a netCDF file
'''
import numpy as np
import netCDF4 as nc
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
def votemper_depth_calc(runid, endyear, endmonth, endday, startyear=2002, startmonth=1, startday=5):
    path = '/mnt/storage6/myers/NEMO/ANHA4-EPM151/'
    output_path = '/mnt/storage6/hlouis/data_files/temperature_salinity/' 

    start_time = datetime.date(startyear, startmonth, startday)

    end_time = datetime.date(endyear, endmonth, endday)

    #figure out all the dates we have model files
    delta = end_time - start_time
    times = []

    i = 0
    while i < delta.days+1:
        t = start_time + datetime.timedelta(days=i)
        if t.month == 2 and t.day == 29:
            t = datetime.date(t.year, 3, 1)
            i = i+6
        else:
            i = i+5
        times.append(t)

    #and now make a list of model files to read
    mdl_files = []
    for t in times:
        mdl_files.append(path+"ANHA4-"+runid+"_y"+str(t.year)+"m"+str(t.month).zfill(2)+"d"+str(t.day).zfill(2)+"_gridT.nc")

    d = xr.open_mfdataset(mdl_files, concat_dim='time_counter', data_vars='minimal', coords='minimal', compat='override')
    #also want to read in the mesh grid info
    grid_file = '/mnt/storage4/tahya/runoff/runoff_temp_regions_mask.nc'
    mesh = nc.Dataset(grid_file)
    mask = np.array(mesh.variables['hb_mask'])
    mesh.close()

    d.coords['mask'] = (('deptht', 'y_grid_T', 'x_grid_T'), mask[0,:,:,:])
    d = d.where(d.mask == 2)
    
    #want to do annual averages
    annual_average = d  #d['votemper'].resample(time_counter='M').mean() #wouldn't this resample and give the monthly average?
    d1 = annual_average.where(annual_average['deptht'] < 50, drop=True)
    #and now we want to average over the top 200m of the water column
    d = xr.where(annual_average['deptht'] < 50, annual_average['votemper'], np.nan)  # drop=True)
    #temp_data = d.values
    times = d1['time_counter'].values
    #t = d.dims['time_counter']
    
    d1.close()
    
    #calculate the weights
    n = len(d['deptht'])
    weight = np.zeros(n)
    dz = np.zeros(n)
    dd = d['deptht'][n-1]
    for i in range(n):
        if i == 0:
            weight[i] = d['deptht'][i]/dd
            dz[i] = d['deptht'][i]
        else:
            weight[i] = (d['deptht'][i] - d['deptht'][i-1])/dd
            dz[i] = d['deptht'][i] - d['deptht'][i-1]
    '''
    weights = xr.DataArray(weight, coords=[d['deptht']], dims=['deptht'])

    #and take the average
    d_weighted = d.weighted(weights)
    surface_temp= d_weighted.mean(dim='deptht', skipna=True)
    '''
    x = d.sizes['x_grid_T']
    y = d.sizes['y_grid_T']

    #get dz on the grid
    dz_grid = np.tile(dz[:,np.newaxis,np.newaxis], (1,y,x))
    tmp = d
    
    t1 = tmp.values*dz_grid
    temp  = t1.sum(dim='deptht', skipna=True)
    temp = temp.values
    times = [dt.strptime(str(k), '%Y-%m-%d %H:%M:%S') for k in list(times)]
    
    plt.plot(times, temp)
    plt.title('Hudson Bay Temperature in the Top 50m')
    plt.ylabel('Temperature ($^o$C)')
    plt.show()
    plt.clf()       
    
if __name__ == "__main__":
    votemper_depth_calc(runid='EPM151', endyear=2002, endmonth=12, endday=31)
