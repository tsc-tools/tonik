from datetime import datetime, timedelta
import inspect
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import xarray as xr

from tonik import StorageGroup, generate_test_data


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


tstart = datetime(2023,1,1,0,0,0)
ndays = 10

@pytest.fixture(scope='package')
def setup(tmp_path_factory):
    features1D = ['rsam',
                  'dsar',
                  'central_freq',
                  'predom_freq',
                  'bandwidth',
                  'rsam_energy_prop']
    features2D = [('sonogram', 'sonofrequency'),
                  ('ssam', 'frequency'),
                  ('filterbank', 'fbfrequency')]

    savedir = tmp_path_factory.mktemp('vumt_test_tmp', numbered=True)
    g = StorageGroup('volcanoes', rootdir=savedir)
    c1 = g.get_store('WIZ', '00', 'HHZ')
    c2 = g.get_store('MDR', '00', 'BHZ')
    c3 = g.get_store('MAVZ', '10', 'EHZ')
    c4 = g.get_store('MMS', '66', 'BHZ')
    # Generate some fake data
    for _f in features1D:
        feat = generate_test_data(tstart=tstart,
                                    feature_name=_f,
                                    ndays=ndays)
        for _c in g.stores:
            _c.save(feat)
    for _n, _f in features2D:
        feat = generate_test_data(tstart=tstart,
                                  feature_name=_n,
                                  ndays=ndays,
                                  nfreqs=8,
                                  freq_name=_f,
                                  dim=2)
        for _c in g.stores:
            _c.save(feat)

    alg = generate_test_data(tstart=tstart,
                             feature_name='autoencoder',
                             ndays=ndays,
                             nfreqs=5,
                             freq_name='cluster',
                             dim=2)
    c2.save(alg)
    return savedir, g 


@pytest.fixture(scope='module')
def setup_api(setup):
    savedir, g = setup
    from tonik.api import TonikAPI
    ta = TonikAPI(str(savedir)) 
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    return client, g.get_store('MDR', '00', 'BHZ') 