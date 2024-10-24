import os
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from tonik import Storage, generate_test_data
from tonik.xarray2netcdf import xarray2netcdf


def test_xarray2netcdf(tmp_path_factory):
    """
    Test writing xarray data to hdf5.
    """
    xdf = generate_test_data(
        dim=2, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime.fromisoformat(xdf.attrs['starttime']),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
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


def test_xarray2netcdf_archive_starttime(tmp_path_factory):
    xdf = generate_test_data(
        dim=1, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf, archive_starttime=datetime(2022, 1, 1))

    xdf_test = c('rsam')
    assert np.all(
        np.isnan(xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].data))
    nitems = int((datetime(2022, 7, 18, 0, 0, 0) -
                 datetime(2022, 1, 1))/timedelta(minutes=10))
    assert xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].shape[0] == nitems


def test_xarray2netcdf_merge_arrays(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf2, archive_starttime=datetime(2022, 8, 1))
    c.save(xdf1, archive_starttime=datetime(2022, 8, 1))
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == 24
    assert xdf_test.loc['2022-07-18T08:00:00'] == xdf1['rsam'].loc['2022-07-18T08:00:00']
    assert xdf_test.loc['2022-07-20T11:50:00'] == xdf2['rsam'].loc['2022-07-20T11:50:00']


def test_xarray2netcdf_resolution(tmp_path_factory):
    xdf = generate_test_data(dim=1, ndays=1, tstart=datetime(2022, 7, 18, 0, 0, 0),
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf, resolution=0.1, archive_starttime=datetime(2022, 7, 18))

    xdf_test = c('rsam')
    assert xdf_test.loc['2022-07-18T00:12:00'] == xdf['rsam'].loc['2022-07-18T00:10:00']
    assert np.isnan(xdf_test.loc['2022-07-18T00:06:00'].data)
    assert xdf_test.attrs['resolution'] == 0.1
    assert xdf_test.attrs['resolution_units'] == 'h'


def test_xarray2netcdf_attributes(tmp_path_factory):
    starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf = generate_test_data(dim=1, ndays=1, tstart=starttime,
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf, archive_starttime=starttime)
    xdf_test = c('rsam')
    assert xdf_test.attrs['station'] == xdf.attrs['station']
    assert xdf_test.attrs['feature'] == 'rsam'


def test_xarray2netcdf_with_gaps(tmp_path_factory):
    """
    Test writing xarray data to hdf5 with gaps.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == 24


@pytest.mark.xfail(raises=OSError)
def test_xarray2netcdf_multi_access(tmp_path_factory):
    """
    Test writing xarray data to hdf5 while the file is open. This is currently
    not working with NetCDF4. See the following discussions for reference:
    https://github.com/pydata/xarray/issues/2887
    https://stackoverflow.com/questions/49701623/is-there-a-way-to-release-the-file-lock-for-a-xarray-dataset
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    xdf1 = generate_test_data(
        dim=1, ndays=1, tstart=datetime(2022, 7, 18, 8, 0, 0))
    xdf2 = generate_test_data(
        dim=1, ndays=1, tstart=datetime(2022, 7, 19, 12, 0, 0))

    xarray2netcdf(xdf1, temp_dir)
    xdf_dummy = xr.open_dataset(os.path.join(temp_dir, 'rsam.nc'),
                                group='original', engine='h5netcdf')
    xarray2netcdf(xdf2, temp_dir)


def test_xarray2zarr(tmp_path_factory):
    xdf = generate_test_data(
        dim=2, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime.fromisoformat(xdf.attrs['starttime']),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf)

    xdf_test_ssam = c('ssam')
    xdf_test_fb = c('filterbank')
    np.testing.assert_array_equal(xdf['ssam'].values,
                                  xdf_test_ssam.values)
    np.testing.assert_array_equal(xdf['frequency'].values,
                                  np.squeeze(xdf_test_ssam['frequency'].values))
    np.testing.assert_array_equal(xdf['filterbank'].values,
                                  xdf_test_fb.values)
    np.testing.assert_array_equal(xdf['fbfrequency'].values,
                                  np.squeeze(xdf_test_fb['fbfrequency'].values))
    # minor differences can occur on the level of nanoseconds; ensure
    # differences are less than 1 microsecond
    dt = np.abs((xdf_test_ssam['datetime'].values -
                xdf['datetime'].values)).max()
    assert dt < np.timedelta64(1, 'us')


def test_xarray2zarr_with_gaps(tmp_path_factory):
    """
    Test writing xarray data to zarr with gaps.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == 0


def test_xarray2zarr_outofsequence(tmp_path_factory):
    """
    Test writing xarray data to zarr where the later part is written first.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    middle = datetime(2022, 7, 18, 12, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, tstart=middle)
    xdf3 = generate_test_data(dim=1, intervals=3, tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf3)
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    np.testing.assert_array_equal(
        xdf_test.datetime.values, xr.merge([xdf1, xdf2, xdf3]).datetime.values)


def test_xarray2zarr_duplicates(tmp_path_factory):
    """
    Test writing xarray data to zarr where the later part is written first.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start)
    duplicate_data = xdf1.isel(datetime=-1)
    xdf1 = xr.concat([xdf1, duplicate_data], dim='datetime')
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    np.testing.assert_array_equal(
        xdf_test.datetime.values, xdf1.drop_duplicates('datetime', keep='first').merge(xdf2).datetime.values)


def test_xarray2zarr_with_overlaps_1D(tmp_path_factory):
    """
    Test writing xarray data to zarr with overlaps for 1D features.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 18, 9, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert xdf_test.isel(datetime=0).values == xdf1.rsam.isel(
        datetime=0).values
    assert xdf_test.isel(datetime=1).values == xdf2.rsam.isel(
        datetime=0).values
    assert xdf_test.datetime.values[-1] == xdf2.datetime.values[-1]


def test_xarray2zarr_with_overlaps_2D(tmp_path_factory):
    """
    Test writing xarray data to zarr with overlaps for 2D features.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 18, 9, 0, 0)
    xdf1 = generate_test_data(dim=2, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=2, intervals=3, freq='1h', tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('ssam')
    np.testing.assert_array_equal(xdf_test.isel(datetime=0).values, xdf1.ssam.isel(
        datetime=0).values)
    np.testing.assert_array_equal(xdf_test.isel(datetime=1).values, xdf2.ssam.isel(
        datetime=0).values)
    assert xdf_test.datetime.values[-1] == xdf2.datetime.values[-1]


def test_xarray2zarr_overwrite(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start,
                              seed=43)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=start + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    np.testing.assert_array_equal(
        xdf_test.values, xdf2.rsam.values)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            xdf_test.values, xdf1.rsam.values)
    np.testing.assert_array_equal(
        xdf_test.datetime.values, xdf1.datetime.values)
