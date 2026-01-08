import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
import pandas as pd

from tonik import Storage, generate_test_data
from tonik.xarray2zarr import (xarray2zarr,
                               _init_timeseries_store,
                               _fill_time_gaps_between_datasets)
from tonik.utils import get_dt
import zarr


def test_initialize_zarr_storage(setup_multi_dimensional):
    """
    Test initializing a zarr storage backend. 
    """
    setup_multi_dimensional
    tempdir, xds, xds2 = setup_multi_dimensional
    fout = os.path.join(tempdir, 'test_initialize.zarr')
    feature = 'order2'
    timedim = 'datetime'
    archive_starttime = datetime(2022, 7, 1, 1, 12, 56)
    shape_list = list(xds[feature].shape)
    dims_list = list(xds[feature].dims)
    shape_list.pop(dims_list.index(timedim))
    dims_list.pop(dims_list.index(timedim))
    xds_empty = _init_timeseries_store(
        fout,
        start=np.datetime64(archive_starttime),
        stop=xds[feature][timedim].values[-1],
        interval=get_dt(xds[timedim]),
        data_vars={
            feature: (tuple(dims_list), xds.coords, tuple(shape_list), xds[feature].dtype)},
        group='original',
        chunk_size=144,
        timedim=timedim
    )
    root = zarr.open(fout, mode='r')
    arr = root['original/order2']
    assert arr.shape == (3, 24, 6, 2592)
    assert arr.chunks == (3, 24, 6, 144)
    assert xds_empty.isnull().sum() == 2592 * 3 * 24 * 6


def test_fill_time_gaps_between_datasets():
    """
    Test filling time gaps between two xarray datasets.
    """
    tstart_existing = datetime(2022, 7, 18, 0, 0, 0)
    tstart_new = datetime(2022, 8, 18, 0, 0, 0)
    xds_existing = generate_test_data(
        dim=1, freq='1D', intervals=10,
        tstart=tstart_existing)
    xds_new = generate_test_data(
        dim=1, freq='1D', intervals=5, tstart=tstart_new)
    xds_filled = _fill_time_gaps_between_datasets(
        xds_existing['rsam'].isel({'datetime': -1}), xds_new['rsam'],
        get_dt(xds_new['datetime']), timedim='datetime')
    print(xds_filled)
    assert xds_existing.datetime.values[-1] + \
        np.timedelta64(1, 'D') == xds_filled.datetime.values[0]
    assert xds_filled.loc[dict(datetime=xds_new.datetime)] == xds_new
    assert xds_filled.isnull().sum() == 25
    assert xds_filled.shape[0] % 10 == 0


def test_xarray2zarr(tmp_path_factory):
    xdf = generate_test_data(
        dim=2, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    archive_starttime = datetime(2022, 7, 1, 0, 0, 0)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime.fromisoformat(xdf.attrs['starttime']),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf, archive_starttime=archive_starttime,
           chunk_size=144)
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
    archive_starttime = datetime(2022, 7, 1, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == int(
        (xdf2.datetime[0] - xdf1.datetime[-1])/pd.Timedelta('10min'))


def test_xarray2zarr_outofsequence(tmp_path_factory):
    """
    Test writing xarray data to zarr where the later part is written first.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    middle = datetime(2022, 7, 18, 12, 0, 0)
    end = datetime(2022, 7, 18, 14, 0, 0)
    archive_starttime = datetime(2022, 7, 1, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, tstart=start, seed=42)
    xdf2 = generate_test_data(dim=1, intervals=3, tstart=middle, seed=43)
    xdf3 = generate_test_data(dim=1, intervals=3, tstart=end, seed=44)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf3, chunk_size=10, archive_starttime=archive_starttime)
    c.save(xdf1, chunk_size=10)
    c.save(xdf2, chunk_size=10)
    xdf_test = c('rsam')
    assert np.all(xdf_test.loc[dict(datetime=xdf3.datetime)] == xdf3['rsam'])
    assert np.all(xdf_test.loc[dict(datetime=xdf2.datetime)] == xdf2['rsam'])
    assert np.all(xdf_test.loc[dict(datetime=xdf1.datetime)] == xdf1['rsam'])
    assert xdf_test.sizes['datetime'] == int(
        (xdf3.datetime.values[-1] - np.datetime64(start))/np.timedelta64(600, 's') + 1)


def test_xarray2zarr_duplicates(tmp_path_factory):
    """
    Test writing xarray data to zarr where the later part is written first.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    archive_starttime = datetime(2022, 7, 1, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start)
    duplicate_data = xdf1.isel(datetime=-1)
    xdf1 = xr.concat([xdf1, duplicate_data], dim='datetime')
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, chunk_size=10, archive_starttime=archive_starttime)
    c.save(xdf2, chunk_size=10)
    xdf_test = c('rsam')
    assert np.all(xdf_test.loc[dict(datetime=xdf1.datetime)].dropna(
        'datetime') == xdf1['rsam'].dropna('datetime'))
    assert np.all(xdf_test.loc[dict(datetime=xdf2.datetime)].dropna(
        'datetime') == xdf2['rsam'].dropna('datetime'))


def test_xarray2zarr_with_overlaps_1D(tmp_path_factory):
    """
    Test writing xarray data to zarr with overlaps for 1D features.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 18, 10, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert (~xdf_test.isnull()).sum() == 5
    assert xdf_test.isel(datetime=0).values == xdf1.rsam.isel(
        datetime=0).values
    assert xdf_test.isel(datetime=2).values == xdf2.rsam.isel(
        datetime=0).values


def test_xarray2zarr_with_overlaps_2D(tmp_path_factory):
    """
    Test writing xarray data to zarr with overlaps for 2D features.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 18, 9, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf1 = generate_test_data(dim=2, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=2, intervals=3, freq='1h', tstart=end)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    c.save(xdf2)
    xdf_test = c('ssam')
    assert np.all(~np.isnan(xdf_test.values[:, 0:4]))
    assert np.all(np.isnan(xdf_test.values[:, 4:]))
    np.testing.assert_array_equal(xdf_test.isel(datetime=0).values, xdf1.ssam.isel(
        datetime=0).values)
    np.testing.assert_array_equal(xdf_test.isel(datetime=1).values, xdf2.ssam.isel(
        datetime=0).values)


def test_xarray2zarr_overwrite(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start,
                              seed=43)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=start + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert np.all(xdf_test.loc[dict(datetime=xdf1.datetime)] == xdf2['rsam'])


def test_xarray2zarr_errors(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    start_false = datetime(2022, 7, 1, 0, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(
        dim=1, intervals=3, freq='1h', tstart=start_false)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=start + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    with pytest.raises(ValueError):
        c.save(xdf2)


def test_xarray2zarr_high_dimensionality(setup_multi_dimensional):
    """
    Test writing xarray data to zarr with more than 2 dimensions.
    """
    tempdir, xds, xds2 = setup_multi_dimensional
    start = datetime(2022, 7, 18, 0, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    g = Storage('test_experiment', rootdir=tempdir,
                starttime=start, endtime=start + timedelta(days=10),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xds, chunk_size=144, archive_starttime=archive_starttime)
    c.save(xds2, chunk_size=144)
    xds_test = c('order2')
    assert np.all(xds_test.loc[dict(datetime=xds.datetime)] == xds['order2'])
    assert np.all(xds_test.loc[dict(datetime=xds2.datetime)] == xds2['order2'])
    assert xds_test.sizes['datetime'] == 432


def test_xarray2zarr_metadata(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2zarr')
    start = datetime(2022, 7, 18, 8, 0, 0)
    archive_starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf1 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start)
    xdf2 = generate_test_data(dim=1, intervals=3, freq='1h', tstart=start,
                              seed=43)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=start + timedelta(days=1),
                backend='zarr')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1, archive_starttime=archive_starttime)
    c.save(xdf2)
    xdf_test = c('rsam', metadata=True)
    assert len(xdf_test['update_log'].values) == 2
    assert len(xdf_test['last_datapoint'].values) == 2
    assert xdf_test['last_datapoint'].values[-1] == xdf2.datetime.values[-1]
    assert xdf_test['update_log'].values[-1] <= np.datetime64(
        datetime.now(timezone.utc))
    assert xdf_test['resolution'][()] == 1.0
