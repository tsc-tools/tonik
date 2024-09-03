from datetime import datetime
import logging
import os
from warnings import filterwarnings

from cftime import num2date, date2num
import h5netcdf
import numpy as np


def xarray2hdf5(xArray, fdir, rootGroupName="original", timedim="datetime",
                archive_starttime=datetime(2000, 1, 1), resolution=None):
    """
    Store an xarray dataset as an HDF5 file.

    Parameters
    ----------
    xArray : xarray.Dataset
        Data to store.
    fdir : str
        Directory to store data under.
    rootGroupName : str
        Hdf5 group name.
    timedim : str
        Name of time dimension.
    archive_starttime : datetime
        Start time of archive. If the start time of the data is before this
        time, the data start time is used.
    resolution : float
        Time resolution of the data in hours. If None, the resolution is
        determined from the data.
    """
    filterwarnings(action='ignore', category=DeprecationWarning,
               message='`np.bool` is a deprecated alias')

    starttime = xArray[timedim].values[0].astype('datetime64[us]').astype(datetime)
    starttime = min(starttime, archive_starttime)
    if resolution is None:
        resolution = (np.diff(xArray[timedim])/np.timedelta64(1, 'h'))[0]

    for featureName in list(xArray.data_vars.keys()):
        h5file = os.path.join(fdir, featureName +'.nc')

        mode = 'a' if os.path.isfile(h5file) else 'w'
        
        with h5netcdf.File(h5file, mode) as h5f:
            try:
                rootGrp = _create_h5_Structure(rootGroupName, featureName,
                                               h5f, xArray, starttime, timedim)
            except ValueError: # group already exists, append
                rootGrp = h5f[rootGroupName]

            # determine indices
            new_time = date2num(xArray[timedim].values.astype('datetime64[us]').astype(datetime),
                                units=rootGrp[timedim].attrs['units'],
                                calendar=rootGrp[timedim].attrs['calendar'])
            t0 = date2num(starttime,
                          units=rootGrp[timedim].attrs['units'],
                          calendar=rootGrp[timedim].attrs['calendar'])
            indices = np.rint((new_time - t0)/resolution).astype(int)
            if not np.all(indices >= 0):
                raise ValueError("Data starts before the archive start time")
            times = rootGrp[timedim]
            newsize = indices[-1] + 1
            if newsize > times.shape[0]:
                rootGrp.resize_dimension(timedim, newsize)
            times[:] = t0 + np.arange(times.shape[0]) * resolution
            data = rootGrp[featureName]
            if len(data.shape) > 1:
                data[:, indices] = xArray[featureName].values
            else:
                data[indices] = xArray[featureName].values
            rootGrp.attrs['endtime'] = str(num2date(times[-1], units=rootGrp[timedim].attrs['units'],
                                                    calendar=rootGrp[timedim].attrs['calendar']))
            try:
                _setMetaInfo(featureName, h5f, xArray)
            except KeyError as e:
                logging.warning(f"Could not set all meta info for {featureName}: {e}")


def _create_h5_Structure(defaultGroupName, featureName, h5f, xArray, starttime, timedim):
    rootGrp = h5f.create_group(defaultGroupName)
    rootGrp.dimensions[timedim] = None
    coordinates = rootGrp.create_variable(timedim, (timedim,), float)
    coordinates.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
    coordinates.attrs['calendar'] = 'gregorian'
    rootGrp.attrs['starttime'] = str(starttime)
    for label, size in xArray.dims.items(): 
        if not np.issubdtype(xArray[label].dtype, np.datetime64):
            rootGrp.dimensions[label] = size 
            coordinates = rootGrp.create_variable(label, (label,), float)
            coordinates[:] = xArray[label].values
    # Note: xArray.dims returns a dictionary of dimensions that are not necesarily
    # in the right order; xArray[featureName].dims returns a tuple with dimension
    # names in the correct order
    rootGrp.create_variable(featureName, tuple(xArray[featureName].dims), dtype=float, fillvalue=0.)
    return rootGrp


def _setMetaInfo(featureName, h5f, xArray):
    h5f.attrs['station'] = xArray.attrs['station']
    h5f.attrs['latitude'] = -42
    h5f.attrs['longitude'] = 168
    h5f.attrs['datatype'] = featureName
 