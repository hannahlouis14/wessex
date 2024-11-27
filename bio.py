import datetime
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt

def dissO2(runid, year):  #, endyear, endmonth, endday, startyear=2021, startmonth=8, startday=3):
    data_path = '/project/6007519/hlouis/data_files/obs/'
    mod_path = '/project/6007519/pmyers/ANHA4/ANHA4-'+runid+'-S/'
    path = '/home/hlouis/projects/rrg-pmyers-ad/hlouis/plotting/data_files/surf_bio/'
    grid_file = '/project/6007519/weissgib/plotting/data_files/anha4_files/ANHA4_mesh_mask.nc'
    fig_path = '/project/6007519/hlouis/plotting/figures/bio/'
    mask_file = '/project/6007519/hlouis/scripts/HBC_mask.nc'
    data_file = path+'ctd_data-'+year+'.csv'
    
    variable = 'vooxy'    

    #jb_oxy  = path+runid+'_vooxy_jb.nc'
    #hb_oxy  = path+runid+'_vooxy_hbc.nc'
    regions = {'jb': 'James Bay', 'hbc': 'Hudson Bay'}
    long_name = {'vooxy': 'Dissolved Oxygen'}

    experiment = []
    date = []
    var = []
    region = []

    for r in regions.keys():
        d = xr.open_mfdataset(path+runid+'_'+variable+'_'+r+'.nc')

        v = d[variable].values

        datetimeindex = d.indexes['time_counter'].to_datetimeindex()
        times = datetimeindex.values
        l = d.dims['time_counter']

        if runid == 'EPM101':
            runid = 'HYPE,CGRF'

        for i in range(l):
           region.append(r)
           experiment.append(runid)
        var.extend(list(v))
        date.extend(list(times))

        d.close()

    #now make a pandas dataframe for easy plotting
    all_data = {'experiment': experiment, 'region': region, variable: var, 'date': date}
    df = pd.DataFrame(all_data)

    for r in regions:
        rd = df.loc[df['region'] == r]
        rd = rd.pivot(index='date', columns='experiment', values=variable)
        rd.plot()
        plt.grid(True)
        plt.title(regions[r]+' '+long_name[variable])
        plt.ylabel('mol/m$^3$')
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=2, fancybox=True, shadow=True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(fig_path+variable+'_'+r+'.png')
        plt.clf()
    
dissO2(runid='EPM151', year='2021')  #, endyear=2021, endmonth=8, endday=28)
