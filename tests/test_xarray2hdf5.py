from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsctools import xarray2hdf5


def generate_test_data(dim=1, nhours=30, nfreqs=10,
                       nans=True, tstart=datetime.utcnow(),
                       feature_name=None, freq_name=None):
    """
    Generate a 1D or 2D feature for testing.
    """
    assert dim < 3
    assert dim > 0

    nints = nhours * 6
    dates = pd.date_range(tstart, freq='10min', periods=nints)
    rs = np.random.default_rng(42)
    # Random walk as test signal
    data = np.abs(np.cumsum(rs.normal(0, 8., len(dates))))
    if dim == 2:
        data = np.tile(data, (nfreqs, 1))
    # Add 10% NaNs
    idx_nan = rs.integers(0, nints-1, int(0.1*nints))
    if dim == 1:
        if nans:
            data[idx_nan] = np.nan
        if feature_name is None:
            feature_name = 'rsam'
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[dates], dims=['datetime'])})
    if dim == 2:
        data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_name is None:
            feature_name = 'ssam'
        if freq_name is None:
            freq_name = 'frequency'
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[freqs, dates], dims=[freq_name, 'datetime'])})
    xrd.attrs['starttime'] = str(dates[0])
    xrd.attrs['endtime'] = str(dates[-1])
    xrd.attrs['station'] = 'MDR'
    return xrd


def test_xarray2hdf5(tmp_path_factory):
    """
    Test writing xarray data to hdf5.
    """
    linfreqs = np.arange(0, 25.1, 0.1)
    xdf = generate_test_data(dim=2, nhours=3*24)
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    xarray2hdf5(xdf, temp_dir)

    xdf_test = xr.open_dataset(os.path.join(temp_dir, 'ssam.nc'), group="original")
    np.testing.assert_array_equal(xdf['ssam'].values,
                                  xdf_test['ssam'].values)
    np.testing.assert_array_equal(xdf['frequency'].values,
                                  np.squeeze(xdf_test['frequency'].values))
    # minor differences can occur on the level of nanoseconds; ensure
    # differences are less than 1 microsecond
    dt = np.abs((xdf_test['datetime'].values - xdf['datetime'].values)).max()
    assert dt < np.timedelta64(1, 'us')
    

def test_xarray2hdf5_with_gaps(tmp_path_factory):
    """
    Test writing xarray data to hdf5 with gaps.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    xdf1 = generate_test_data(dim=1, nhours=1, tstart=datetime(2022, 7, 18, 8, 0, 0))
    xdf2 = generate_test_data(dim=1, nhours=1, tstart=datetime(2022, 7, 18, 12, 0, 0))

    xarray2hdf5(xdf1, temp_dir)
    xarray2hdf5(xdf2, temp_dir)
    xdf_test = xr.open_dataset(os.path.join(temp_dir, 'rsam.nc'),
                                group='original')
    assert xdf_test.rsam.isnull().sum() ==  18
        
@pytest.mark.xfail(raises=OSError)
def test_xarray2hdf5_multi_access(tmp_path_factory):
    """
    Test writing xarray data to hdf5 while the file is open. This is currently
    not working with NetCDF4. See the following discussions for reference:
    https://github.com/pydata/xarray/issues/2887
    https://stackoverflow.com/questions/49701623/is-there-a-way-to-release-the-file-lock-for-a-xarray-dataset
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2hdf5')
    xdf1 = generate_test_data(dim=1, nhours=1, tstart=datetime(2022, 7, 18, 8, 0, 0))
    xdf2 = generate_test_data(dim=1, nhours=1, tstart=datetime(2022, 7, 18, 12, 0, 0))

    xarray2hdf5(xdf1, temp_dir)
    xdf_dummy = xr.open_dataset(os.path.join(temp_dir, 'rsam.nc'),
                                group='original', engine='h5netcdf')
    xarray2hdf5(xdf2, temp_dir)
 