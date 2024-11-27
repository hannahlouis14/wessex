import datetime
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import pandas as pd
import matplotlib.path as mpath
import matplotlib.dates as mdates
import glob
import numpy.ma as ma


def top50_temp(runid, endyear, endmonth, endday, startyear=2002, startmonth=1, startday=5):
    path = '/mnt/storage6/myers/NEMO/ANHA4-EPM151/'
    output_path = '/mnt/storage6/hlouis/plots/fwc/'
    #mdl_files = glob.glob(path+'ANHA4-'+runid+'*_gridT.nc')
    mdl_files = []
    start_time = datetime.date(startyear, startmonth, startday)
    end_time = datetime.date(endyear, endmonth, endday)
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
    mdl_files = []

    for t in times:
        mdl_files.append(path+"ANHA4-EPM151_y"+str(t.year)+"m"+str(t.month).zfill(2)+"d"+str(t.day).zfill(2)+"_gridT.nc")

    grid_file = '/mnt/storage4/tahya/model_files/ANHA4_mesh_mask.nc'
    mesh = nc.Dataset(grid_file)
    mask = np.array(mesh.variables['tmask'])
    lons = np.array(mesh.variables['nav_lon'])
    lats = np.array(mesh.variables['nav_lat'])
    
    mesh.close()

    ds = xr.open_mfdataset(mdl_files, concat_dim='time_counter', data_vars='minimal', coords='minimal', compat='override')
    ds.coords['mask'] = (('deptht', 'y_grid_T', 'x_grid_T'), mask[0,:,:,:])
    ds = ds.where(ds.mask == 1)  # drops data out of mask
    
    average = ds.groupby('time_counter.year').mean('time_counter')  #ds['vosaline'].resample(time_counter='M').mean()
    d = average.where(average['deptht'] < 50, drop=True)
    #print(d['vosaline'])
    
    zero_start = True
    n = len(d['deptht'])
    weight = np.zeros(n)
    dd = d['deptht'][n-1] #SHOULD THIS BE THE FROM FULL DEPTH??
    for i in range(n):
        if zero_start:
            if i == 0:
                weight[i] = d['deptht'][i]/dd
            else:
                weight[i] = (d['deptht'][i] - d['deptht'][i-1])/dd
        else:
            if i == 0:
                k = full_depth.index(d['deptht'][i])
                weight[i] = (d['deptht'][i] - full_depth[k-1])/dd
            else:
                weight[i] = (d['deptht'][i] - d['deptht'][i-1])/dd

    weights = xr.DataArray(weight, coords=[d['deptht']], dims=['deptht'])
    d_weighted = d.weighted(weights)
    surface_temp = d_weighted.mean(dim='deptht', skipna=True)

    years = surface_temp['year'].values

    for y in range(surface_temp.dims['year']):
        temp = surface_temp['votemper'].isel(year=y).values
        
        land_50m = feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='gray', linewidth=0.5)
        projection=ccrs.Mercator(central_longitude=-80)

        fig = plt.figure(figsize=(10, 9))
        ax = plt.subplot(1, 1, 1, projection=projection)
        #ax = plt.axes(projection=projection)
        ax.set_extent([-96,-68,50,67], crs=ccrs.PlateCarree())
        ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        ax.coastlines(resolution='50m')

        p1 = ax.pcolormesh(lons, lats, temp, transform=ccrs.PlateCarree(), cmap='gist_ncar', vmax=4)
        ax_cb = plt.axes([0.85, 0.25, 0.015, 0.5])
        cb = plt.colorbar(p1, cax=ax_cb, orientation='vertical')
        cb.ax.set_ylabel('Weighted Temperature in the top 50m')
        plt.savefig(output_path+'Temperature_weighted_total_0_50_'+runid+'_'+str(years[y])+'.png')
        #plt.show()
        plt.clf()


def top50_timeseries(runid, endyear, endmonth, endday, startyear=2002, startmonth=1, startday=5):
    path = '/mnt/storage6/myers/NEMO/ANHA4-EPM151/'
    output_path = '/mnt/storage6/hlouis/plots/fwc/'
    mdl_files = []
    start_time = datetime.date(startyear, startmonth, startday)
    end_time = datetime.date(endyear, endmonth, endday)
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
    mdl_files = []

    for t in times:
        mdl_files.append(path+"ANHA4-EPM151_y"+str(t.year)+"m"+str(t.month).zfill(2)+"d"+str(t.day).zfill(2)+"_gridT.nc")
 
    grid_file = '/mnt/storage4/tahya/model_files/ANHA4_mesh_mask.nc'
    mesh_file = '/mnt/storage4/tahya/model_file/regions_mask.nc'

    mesh = nc.Dataset(grid_file)
    mask = np.array(mesh.variables['tmask'])
    mesh.close()

    ds = xr.open_mfdataset(mdl_files, concat_dim='time_counter', data_vars='minimal', coords='minimal', compat='override')
    ds.coords['mask'] = (('deptht', 'y_grid_T', 'x_grid_T'), mask[0,:,:,:])
    ds = ds.where(ds.mask == 1)

    average = ds  # ds.groupby('time_counter.year').mean('time_counter')
    d = average.where(average['deptht'] < 50, drop=True)
    times = d['time_counter'].values
    t = d.dims['time_counter']


    n = len(d['deptht'])
    weight = np.zeros(n)
    dd = d['deptht'][n-1]
    for i in range(n):
        if i == 0:
            weight[i] = d['deptht'][i]/dd
        else:
            weight[i] = (d['deptht'][i] - d['deptht'][i-1])/dd

    weights = xr.DataArray(weight, coords=[d['deptht']], dims=['deptht'])
    d_weighted = d.weighted(weights)
    surface_variables = d_weighted.mean(dim='deptht', skipna=True)
    votemp = surface_variables['votemper'].values
    d.close()
    mf = nc.Dataset(mask_file)
    date = []
    votemper = []
    regions = {'hb_mask': 'Hudson Bay Complex'}
    region = []
    times = [datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') for x in list(times)]
    for r in regions.keys():
        rmask = mf[r][0,0]
        rmask_broadcast_to(rmask,(t,)+rmask.shape)

        #masked_votemper = votemp.where(rmask == 2)
        masked_votemper = ma.masked_where(rmask==1, votemp)
        regional_votemper_ts = masked_votemper.mean(axis=(1,2))
        print(regional_votemper_ts.shape)
        for i in range(t):
            region.append(r)
        #votemp_ts = masked_votemper.mean(('x_grid_T', 'y_grid_T'))
        #votemp_ts.to_netcdf(output_path+runid+'_votemper_'+r+'.nc')
        date.extend(times)
        votemper.extend(list(regional_votemper_ts)

    experiment='EPM151'
    all_data = {'experiment': experiment,'region':region, 'Temp': votemper, 'date':date}
    df = pd.DataFrame(all_data)
    for r in regions:
        rd = df.loc[df['region']==r]
        rd = rd.pivot(index='date', columns='experiment', values='votemper')
        rd.plot()
        plt.grid(True)
        plt.title(regions[r]+' Temperature')
        plt.ylabel('Temperature ($^o$C)')
        plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.05),ncol=2, fancybox=True, shadow=True)
        plt.legend()
        plt.tight_layout()
        plt.show() 
        #plt.savefig(output_path+'temp_top50_timeseries'+runid+'.png')



if __name__ == "__main__":
    #top50_temp(runid='EPM151', endyear=2021, endmonth=12, endday=31)
    top50_timeseries(runid='EPM151', endyear=2021, endmonth=12, endday=31)



