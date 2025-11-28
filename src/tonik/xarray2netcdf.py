import logging
import os
from datetime import datetime, timezone
from warnings import filterwarnings

import h5netcdf
import numpy as np
import xarray as xr
from cftime import date2num, num2date

from .utils import merge_arrays


def xarray2netcdf(xArray, fdir, group="original", timedim="datetime",
                  archive_starttime=datetime(2000, 1, 1), resolution=None,
                  mode='a'):
    """
    Store an xarray dataset as an HDF5 file.

    Parameters
    ----------
    xArray : xarray.Dataset
        Data to store.
    fdir : str
        Directory to store data under.
    group : str
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

    data_starttime = xArray[timedim].values[0].astype(
        'datetime64[us]').astype(datetime)
    starttime = min(data_starttime, archive_starttime)
    now = datetime.now(tz=timezone.utc)
    if resolution is None:
        resolution = (np.diff(xArray[timedim])/np.timedelta64(1, 'h'))[0]

    for featureName in list(xArray.data_vars.keys()):
        h5file = os.path.join(fdir, featureName + '.nc')
        _mode = 'w'
        if os.path.isfile(h5file) and mode == 'a':
            if archive_starttime > data_starttime:
                xds_existing = xr.open_dataset(
                    h5file, group=group, engine='h5netcdf')
                xda_new = merge_arrays(
                    xds_existing[featureName], xArray[featureName],
                    resolution=resolution)
                xds_existing.close()
                xda_new.to_netcdf(h5file, group=group,
                                  mode='w', engine='h5netcdf')
                continue
            _mode = 'a'

        with h5netcdf.File(h5file, _mode) as h5f:
            try:
                rootGrp = _create_root_group(group, featureName,
                                             h5f, xArray, starttime, timedim)
            except ValueError:  # group already exists, append
                rootGrp = h5f[group]

            try:
                metaGrp = _create_meta_group(h5f, resolution)
            except ValueError:  # group already exists, append
                metaGrp = h5f['meta']

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
            now_time = date2num(now, units=metaGrp['update_log'].attrs['units'],
                                calendar=metaGrp['update_log'].attrs['calendar'])
            ulog = metaGrp['update_log']
            ldata = metaGrp['last_datapoint']
            metaGrp.resize_dimension('update', ulog.shape[0] + 1)
            ulog[-1] = now_time
            metaGrp.resize_dimension('endtime', ldata.shape[0] + 1)
            ldata[-1] = times[-1]
            old_resolution = metaGrp['resolution'][()]
            if abs(old_resolution - resolution) > 1e-5:
                raise ValueError(f"Resolution mismatch for {featureName}: "
                                 f"{old_resolution} != {resolution}")


def _create_root_group(defaultGroupName, featureName, h5f, xArray, starttime, timedim):
    rootGrp = h5f.create_group(defaultGroupName)
    rootGrp.dimensions[timedim] = None
    coordinates = rootGrp.create_variable(timedim, (timedim,), float)
    coordinates.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
    coordinates.attrs['calendar'] = 'gregorian'
    rootGrp.attrs['archive_starttime'] = str(starttime)
    for label, size in xArray.sizes.items():
        if not np.issubdtype(xArray[label].dtype, np.datetime64):
            rootGrp.dimensions[label] = size
            coordinates = rootGrp.create_variable(label, (label,), float)
            coordinates[:] = xArray[label].values
    # Note: xArray.dims returns a dictionary of dimensions that are not necesarily
    # in the right order; xArray[featureName].dims returns a tuple with dimension
    # names in the correct order
    rootGrp.create_variable(featureName, tuple(
        xArray[featureName].dims), dtype=float, fillvalue=0.)
    _set_attributes(featureName, rootGrp, xArray)
    return rootGrp


def _set_attributes(featureName, rootGrp, xArray):
    """
    Set attributes for the root group. Attributes are assumed to not change
    over time. If they do, they should be stored in the 'meta' group.
    """
    for key, value in xArray.attrs.items():
        rootGrp.attrs[key] = value
    rootGrp.attrs['feature'] = featureName


def _create_meta_group(h5f, resolution):
    """
    Create meta group to track processing history.
    """
    metaGrp = h5f.create_group('meta')
    metaGrp.dimensions['update'] = None
    ulog = metaGrp.create_variable('update_log', ('update',), float)
    ulog.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
    ulog.attrs['calendar'] = 'gregorian'
    metaGrp.dimensions['endtime'] = None
    ldata = metaGrp.create_variable('last_datapoint', ('endtime',), float)
    ldata.attrs['units'] = 'hours since 1970-01-01 00:00:00.0'
    ldata.attrs['calendar'] = 'gregorian'
    res = metaGrp.create_variable("resolution", (), dtype=float)
    res[()] = resolution
    return metaGrp
