from datetime import datetime
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tonik import StorageGroup
from tonik.utils import generate_test_data


def test_group(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data') 
    g = StorageGroup('test_experiment', rootdir)
    c = g.get_store('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(rootdir, 'test_experiment/site1/sensor1/channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.get_store('MDR1', '00', 'HHZ')

def test_non_existant_feature(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data') 
    g = StorageGroup('test_experiment', rootdir)
    c = g.get_store('site1', 'sensor1', 'channel1')
    g.starttime = datetime(2016, 1, 1)
    g.endtime = datetime(2016, 1, 2)
    with pytest.raises(FileNotFoundError):
        c('non_existant_feature')


def test_from_directory(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data') 
    g = StorageGroup('test_experiment', rootdir)
    c = g.get_store('site1', 'sensor1', 'channel1')
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
    c = g.get_store('MDR1', '00', 'HHZ')



def test_to_dict(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = StorageGroup('test_experiment', rootdir)
    c = g.get_store('site1', 'sensor1', 'channel1')
    c1 = g.get_store('MDR1', '00', 'HHZ')
    c2 = g.get_store('MDR2', '10', 'BHZ')
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
    g = StorageGroup('volcanoes', rootdir=rootdir).get_store()
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
    g = StorageGroup('volcanoes', rootdir=rootdir).get_store()
    startdate = datetime(2016, 1, 2, 1)
    enddate = datetime(2016, 1, 2, 12)
    xdf = generate_test_data(dim=1, tstart=startdate)
    g.save(xdf)
    g1 = StorageGroup('volcanoes', rootdir=rootdir)
    g1.starttime = startdate
    g1.endtime = enddate
    rsam = g1.get_store()('rsam')
    # Check datetime range is correct
    first_time = pd.to_datetime(rsam.datetime.values[0])
    last_time = pd.to_datetime(rsam.datetime.values[-1])
    assert pd.to_datetime(startdate) == first_time
    assert pd.to_datetime(enddate) == last_time 


def test_rolling_window(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    startdate = datetime(2016, 1, 1)
    enddate = datetime(2016, 1, 2, 12)
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g = StorageGroup('volcanoes', rootdir=rootdir)
    g.save(xdf)

    stack_len_seconds = 3600
    stack_len_string = '1h'

    num_windows = int(stack_len_seconds / pd.Timedelta(xdf.interval).seconds)
    g.starttime = startdate
    g.endtime = enddate
    rsam = g('rsam')
    rsam_rolling = g('rsam', stack_length=stack_len_string)

    # Check correct datetime array
    np.testing.assert_array_equal(rsam.datetime.values,
                                    rsam_rolling.datetime.values)
    # Check correct values
    rolling_mean = [
        np.nanmean(rsam.data[(ind-num_windows+1):ind+1])
        for ind in np.arange(num_windows, len(rsam_rolling.data))
    ]
    np.testing.assert_array_almost_equal(
        np.array(rolling_mean), rsam_rolling.values[num_windows:], 6
        )