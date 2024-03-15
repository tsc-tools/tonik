from datetime import datetime
import logging
import os
from warnings import filterwarnings

from cftime import num2date, date2num, date2index
import h5netcdf
import numpy as np
import xarray as xr


def xarray2hdf5(xArray, fdir, rootGroupName="original", timedim="datetime"):
    """
    Store an xarray dataset as an HDF5 file.

    :param xArray: Data to store. 
    :type xArray: :class:`xarray.Dataset`
    :param fdir: Directory to store data under.
    :type fdir: str
    :param rootGroupName: Hdf5 group name.
    :type rootGroupName: str
    :param timedim: Name of time dimension.
    :type timedim: str
    """
    filterwarnings(action='ignore', category=DeprecationWarning,
               message='`np.bool` is a deprecated alias')

    for featureName in list(xArray.data_vars.keys()):
        h5file = os.path.join(fdir, featureName +'.nc')

        mode = 'a' if os.path.isfile(h5file) else 'w'
        
        with h5netcdf.File(h5file, mode) as h5f:
            try:
                rootGrp = _create_h5_Structure(rootGroupName, featureName, h5f, xArray)
            except ValueError: # group already exists, append
                rootGrp = h5f[rootGroupName]

            # determine indices
            new_time = date2num(xArray[timedim].values.astype('datetime64[us]').astype(datetime),
                                units=rootGrp[timedim].attrs['units'],
                                calendar=rootGrp[timedim].attrs['calendar'])
            dt = (np.diff(xArray['datetime'])/np.timedelta64(1, 'h'))[0]
            t0 = date2num(np.datetime64(rootGrp.attrs['starttime']).astype('datetime64[us]').astype(datetime),
                          units=rootGrp[timedim].attrs['units'],
                          calendar=rootGrp[timedim].attrs['calendar'])
            indices = np.rint((new_time - t0)/dt).astype(int)
            assert np.all(indices >= 0)
            times = rootGrp[timedim]
            newsize = indices[-1] + 1
            if newsize > times.shape[0]:
                rootGrp.resize_dimension(timedim, newsize)
            times[:] = t0 + np.arange(times.shape[0])*dt
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


def _create_h5_Structure(defaultGroupName, featureName, h5f, xArray):
    rootGrp = h5f.create_group(defaultGroupName)
    for label, size in xArray.dims.items(): 
        _setAttributes(label, size, rootGrp, xArray)
    # Note: xArray.dims returns a dictionary of dimensions that are not necesarily
    # in the right order; xArray[featureName].dims returns a tuple with dimension
    # names in the correct order
    rootGrp.create_variable(featureName, tuple(xArray[featureName].dims), dtype=float, fillvalue=0.)
    return rootGrp


def _setAttributes(label, size, rootGrp, xArray):
    if np.issubdtype(xArray[label].dtype, np.datetime64):
        starttime = str(xArray[label].values[0].astype('datetime64[us]').astype(datetime))
        rootGrp.dimensions[label] = None
        coordinates = rootGrp.create_variable(label, (label,), float)
        coordinates.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
        coordinates.attrs['calendar'] = 'gregorian'
        rootGrp.attrs['starttime'] = starttime
    else:
        rootGrp.dimensions[label] = size 
        coordinates = rootGrp.create_variable(label, (label,), float)
        coordinates[:] = xArray[label].values
        

def _setMetaInfo(featureName, h5f, xArray):
    h5f.attrs['station'] = xArray.attrs['station']
    h5f.attrs['latitude'] = -42
    h5f.attrs['longitude'] = 168
    h5f.attrs['datatype'] = featureName
 