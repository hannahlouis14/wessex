import cf_xarray as cfxr
import shapely
import xarray as xr
import xesmf as xe
import numpy as np
import os
import glob
import copy
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib


output_path = '/mnt/storage6/hlouis/DATA/AMSR2/'
data_path = '/mnt/storage2/myers/DATA/AMSR2/'
output_grid = '/mnt/storage4/tahya/model_files/ANHA4_mesh_mask.nc'
lonlat_file = '/mnt/storage6/hlouis/scripts/LongitudeLatitudeGrid-n6250-Arctic.nc'
lonslats = nc.Dataset(lonlat_file)
#lonslats = xr.open_dataset(lonlat_file)
lon = lonslats['Longitudes']
lat = lonslats['Latitudes']
print(lonslats)
print(type(lat))
ds_out = xr.open_dataset(output_grid)
ds_out = ds_out.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})
ds_out = ds_out[['lon','lat']]

years = [2014]  #np.arange(2022,2023,1)

'''
for year in years:
    directory = data_path+str(year)+'/'

    for filename in sorted(glob.glob(directory+'*.nc')):
        ds_in = xr.open_dataset(filename)
        
        #ds_in = ds_in.rename({'x':'lon', 'y':'lat'})
        ice_in = ds_in.z / 100
        
        file_name = filename.replace(directory, '')
        file_name = file_name.replace('asi-AMSR2-n6250-','')
        file_name = file_name.replace('-v5.4.nc', '')
        print(file_name)
        #ds_in = ds_in.isel(y=slice(None,None, -1))  #reindex(y=list(reversed(ds_in.y)))

        ########################## 
        x = ds_in.x / 1000 #['x']
        y = ds_in.y /1000  #['y']
        #y = y.isel(y=slice(None,None, -1))
        #print(polar_ij_to_lonlat(1, 1, grid_size=6.25,hemisphere='north'))
        
        lon, lat = np.meshgrid(len(x), len(y), indexing='ij') 
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                longitude, latitude = polar_ij_to_lonlat(i, j, grid_size=6.25, hemisphere='north')
                lon.append(longitude)
                lat.append(latitude)
                print(np.shape(lat))
        
        lon, lat = polar_xy_to_lonlat(x, y, true_scale_lat, re, e, hemisphere='north')
      
        ############################ 
        #ds = xr.Dataset(coords=dict(lon=(['y','x'], lon.values), lat=(['y','x'], lat.values)), data_vars=dict(sic=(['y','x'],ice_in.data.transpose())), attrs=dict(description='jerry rigged input dataset'),)
        ds = xr.Dataset(coords=dict(lon=(['y','x'], lon), lat=(['y','x'], lat)), data_vars=dict(sic=(['y','x'],ice_in.data)),)
        #ds =ds.rename({'x':'lon', 'y': 'lat'}) 
        
        regridder = xe.Regridder(ds, ds_out, 'bilinear')
        
        #fn = regridder.to_netcdf()  # this will sace the weights calculated here so you can reuse them later
        ice_out = regridder(ds.sic)
        ice_out = ice_out.rename('sic')        
        ice_out.to_netcdf(output_path+str(year)+'/'+'asi-AMSR2-ANHA4-'+file_name+'.nc')
       
        ice_out.close()
        ice_in.close()
        ds.close()
'''
