from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tonik import generate_test_data, StorageGroup
from tonik.xarray2hdf5 import xarray2hdf5


def test_xarray2hdf5(tmp_path_factory):
    """
    Test writing xarray data to hdf5.
    """
    xdf = generate_test_data(dim=2, ndays=3)
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    g = StorageGroup('test_experiment', rootdir=temp_dir,
                     starttime=datetime.fromisoformat(xdf.attrs['starttime']),
                     endtime=datetime.fromisoformat(xdf.attrs['endtime']))
    c = g.get_store('MDR', '00', 'HHZ')
    c.save(xdf)

    xdf_test = c('ssam') 
    np.testing.assert_array_equal(xdf['ssam'].values,
                                  xdf_test.values)
    np.testing.assert_array_equal(xdf['frequency'].values,
                                  np.squeeze(xdf_test['frequency'].values))
    # minor differences can occur on the level of nanoseconds; ensure
    # differences are less than 1 microsecond
    dt = np.abs((xdf_test['datetime'].values - xdf['datetime'].values)).max()
    assert dt < np.timedelta64(1, 'us')


def test_xarray2hdf5_archive_starttime(tmp_path_factory):
    xdf = generate_test_data(dim=1, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    g = StorageGroup('test_experiment', rootdir=temp_dir,
                     starttime=datetime(2000,1,1),
                     endtime=datetime.fromisoformat(xdf.attrs['endtime']))
    c = g.get_store('MDR', '00', 'HHZ')
    c.save(xdf, archive_starttime=datetime(2022, 1, 1))

    xdf_test = c('rsam') 
    assert np.all(np.isnan(xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].data))
    nitems = int((datetime(2022, 7, 18, 0, 0, 0) - datetime(2022, 1, 1))/timedelta(minutes=10))
    assert xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].shape[0] == nitems

def test_xarray2hdf5_resolution(tmp_path_factory):
    xdf = generate_test_data(dim=1, ndays=1, tstart=datetime(2022, 7, 18, 0, 0, 0),
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    g = StorageGroup('test_experiment', rootdir=temp_dir,
                     starttime=datetime(2000,1,1),
                     endtime=datetime.fromisoformat(xdf.attrs['endtime']))
    c = g.get_store('MDR', '00', 'HHZ')
    c.save(xdf, resolution=0.1, archive_starttime=datetime(2022, 7, 18))

    xdf_test = c('rsam') 
    assert xdf_test.loc['2022-07-18T00:12:00'] == xdf['rsam'].loc['2022-07-18T00:10:00']
    assert np.isnan(xdf_test.loc['2022-07-18T00:06:00'].data)

    

def test_xarray2hdf5_with_gaps(tmp_path_factory):
    """
    Test writing xarray data to hdf5 with gaps.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end)
    g = StorageGroup('test_experiment', rootdir=temp_dir,
                     starttime=start, endtime=end + timedelta(days=1))
    c = g.get_store('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam') 
    assert xdf_test.isnull().sum() == 21 


@pytest.mark.xfail(raises=OSError)
def test_xarray2hdf5_multi_access(tmp_path_factory):
    """
    Test writing xarray data to hdf5 while the file is open. This is currently
    not working with NetCDF4. See the following discussions for reference:
    https://github.com/pydata/xarray/issues/2887
    https://stackoverflow.com/questions/49701623/is-there-a-way-to-release-the-file-lock-for-a-xarray-dataset
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=datetime(2022, 7, 18, 8, 0, 0))
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=datetime(2022, 7, 19, 12, 0, 0))

    xarray2hdf5(xdf1, temp_dir)
    xdf_dummy = xr.open_dataset(os.path.join(temp_dir, 'rsam.nc'),
                                group='original', engine='h5netcdf')
    xarray2hdf5(xdf2, temp_dir)
 