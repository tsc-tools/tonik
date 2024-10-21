import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tonik import Storage
from tonik.utils import generate_test_data


def test_group(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir, backend='netcdf')
    c = g.get_substore('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(
        rootdir, 'test_experiment/site1/sensor1/channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.get_substore('MDR1', '00', 'HHZ')


def test_subgroup(tmp_path_factory):
    """
    Test storing data in different subgroups in netcdf and zarr.
    """

    startdate = datetime(2016, 1, 1)
    enddate = datetime(2016, 1, 2, 12)
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir,
                starttime=startdate, endtime=enddate)
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    xdf.rsam.values += 100.
    g.save(xdf, group='modified')
    rsam_original = g('rsam')
    rsam_modified = g('rsam', group='modified')
    assert int(rsam_modified.mean()) == (int(rsam_original.mean()) + 100)
    g = Storage('volcanoes', rootdir=rootdir,
                starttime=startdate, endtime=enddate,
                backend='zarr')
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    xdf.rsam.values += 100.
    g.save(xdf, group='modified')
    rsam_original = g('rsam')
    rsam_modified = g('rsam', group='modified')
    assert int(rsam_modified.mean()) == (int(rsam_original.mean()) + 100)


def test_non_existant_feature(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir)
    c = g.get_substore('site1', 'sensor1', 'channel1')
    g.starttime = datetime(2016, 1, 1)
    g.endtime = datetime(2016, 1, 2)
    with pytest.raises(FileNotFoundError):
        c('non_existant_feature')


def test_from_directory(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir, backend='netcdf')
    c = g.get_substore('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(rootdir, 'test_experiment', 'site1',
                                  'sensor1', 'channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.get_substore('MDR1', '00', 'HHZ')


def test_to_dict(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir)
    c = g.get_substore('site1', 'sensor1', 'channel1')
    c1 = g.get_substore('MDR1', '00', 'HHZ')
    c2 = g.get_substore('MDR2', '10', 'BHZ')
    assert len(g.children) == 3
    for _s in [c1, c2]:
        with open(os.path.join(_s.path, 'feature.nc'), 'w') as f:
            f.write('test')
    _j = g.to_dict()
    assert _j['test_experiment'][0]['MDR1'][0]['00'][0]['HHZ'][0] == 'feature'


def test_call_multiple_days(tmp_path_factory):
    startdate = datetime(2016, 1, 1)
    enddate = datetime(2016, 1, 2, 12)

    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir).get_substore()
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    g.starttime = startdate
    g.endtime = enddate
    rsam = g('rsam')
    xd_index = dict(datetime=slice(startdate, enddate))
    np.testing.assert_array_almost_equal(
        rsam.loc[startdate:enddate], xdf.rsam.loc[xd_index], 5)
    # Check datetime range is correct
    first_time = pd.to_datetime(rsam.datetime.values[0])
    last_time = pd.to_datetime(rsam.datetime.values[-1])
    assert pd.to_datetime(startdate) == first_time
    assert pd.to_datetime(enddate) == last_time


def test_call_single_day(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir).get_substore()
    startdate = datetime(2016, 1, 2, 1)
    enddate = datetime(2016, 1, 2, 12)
    xdf = generate_test_data(dim=1, tstart=startdate)
    g.save(xdf)
    g1 = Storage('volcanoes', rootdir=rootdir)
    g1.starttime = startdate
    g1.endtime = enddate
    rsam = g1.get_substore()('rsam')
    # Check datetime range is correct
    first_time = pd.to_datetime(rsam.datetime.values[0])
    last_time = pd.to_datetime(rsam.datetime.values[-1])
    assert pd.to_datetime(startdate) == first_time
    assert pd.to_datetime(enddate) == last_time


def test_call_single_datapoint(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2016, 1, 2, 1)
    enddate = startdate
    g.starttime = startdate
    g.endtime = enddate
    xdf = generate_test_data(dim=1, tstart=startdate)
    g.save(xdf)
    rsam = g('rsam')
    assert float(
        xdf.rsam.loc[dict(datetime='2016-01-02T01:00:00')]) == float(rsam)


def test_shape(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    tstart = datetime(2016, 1, 1)
    xdf = generate_test_data(dim=2, intervals=20, tstart=tstart)
    g.save(xdf, mode='w', archive_starttime=tstart)
    rsam_shape = g.shape('ssam')
    assert rsam_shape['datetime'] == 20
    assert rsam_shape['frequency'] == 10

    g1 = Storage('volcanoes', rootdir=rootdir, backend='zarr')
    g1.save(xdf, mode='w')
    rsam_shape = g1.shape('ssam')
    assert rsam_shape['datetime'] == 20
    assert rsam_shape['frequency'] == 10
