import numpy as np
import xarray as xr


def shift_to_center(og_var, dimension):
    if dimension == 'x':
        var = (og_var[:,:, 0:-2] + og_var[:,:, 1:-1]) * 0.5
        amountToPad = og_var.x.size - var.x.size
        var = var.pad(x=(0,amountToPad), constant_values=np.nan)
        var = var.roll(x=1, roll_coords=True)

    if dimension == 'y':
        var = (og_var[:,0:-2,:] - og_var[:,1:-1,:]) * 0.5
        amountToPad = og_var.y.size - var.y.size
        var = var.pad(y=(0,amountToPad), constant_values=np.nan)
        var = var.roll(y=1, roll_coords=True)
    
    return var

def rotate(atmos, year):
    angle_file = '/mnt/storage1/xhu/ANHA4-I/RotatedAngle_ANHA4.nc'
    angles = xr.open_dataset(angle_file)

    gsinu = angles['gsinu']
    gsinv = angles['gsinv']
    gcosu = angles['gcosu']
    gcosv = angles['gcosv']
    
    # move the u and v rotated grids to the t point
    gcosV = (gcosv[0:-2,:] + gcosv[1:-1,:]) * 0.5
    gsinV = (gsinv[0:-2,:] + gsinv[1:-1,:]) * 0.5
    gcosU = (gcosu[:, 0:-2] + gcosu[:, 1:-1]) * 0.5
    gsinU = (gsinu[:, 0:-2] + gsinu[:, 1:-1]) * 0.5

    gcosV = gcosV.pad(y=(0,2), constant_values = np.nan) 
    gsinV = gsinV.pad(y=(0,2), constant_values = np.nan) 
    gcosU = gcosU.pad(x=(0,2), constant_values = np.nan) 
    gsinU = gsinU.pad(x=(0,2), constant_values = np.nan) 
    
    gcosV = gcosV.roll(y=1, roll_coords=True)
    gsinV = gsinV.roll(y=1, roll_coords=True)
    gcosU = gcosU.roll(x=1, roll_coords=True)
    gsinU = gsinU.roll(x=1, roll_coords=True)
    
    ''' 
    # need to load in gridU and gridV files for all years...
    if runid == 'EPM151':
        data_path = '/mnt/storage6/myers/NEMO/ANHA4-EPM151/'

    if runid == 'ETW151':
        data_path = '/mnt/storage6/tahya/model_files/ANHA4-ETW161/' 

    if runid == 'ETW152':
        data_path = '/mnt/storage6/tahya/model_files/ANHA4-ETW162/'
    '''
    path = '/mnt/storage6/hlouis/DATA/AtmForcing/'+atmos+'/'

    u_wind_data = xr.open_dataset(path+'u10_core2_ANHA4_y'+str(year)+'.nc')
    u_wind = u_wind_data['t10']
    u = shift_to_center(u_wind, 'x')
    
    v_wind_data = xr.open_dataset(path+'v10_core2_ANHA4_y'+str(year)+'.nc')
    v_wind = v_wind_data['t10']
    v = shift_to_center(v_wind, 'y')

    # rotate angles to NEMO angles?
    u_rot = u*gcosU - v*gsinU
    v_rot = v*gcosV + u*gsinV

    v_rot.to_netcdf(path+'v10_rotated_core2_ANHA4_y'+str(year)+'.nc')    
    u_rot.to_netcdf(path+'u10_rotated_core2_ANHA4_y'+str(year)+'.nc')    


def run_rotate(year_start, year_end):
    years = np.arange(year_start, year_end+1, 1)
    for yr in years:
        rotate(atmos='CORE2-IA', year=yr)

run_rotate(year_start=1958, year_end=2009)



#rotate(atmos='CORE2-IA')
