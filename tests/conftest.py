from datetime import datetime


from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tonik import Storage, generate_test_data, get_labels


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


tstart = datetime(2023, 1, 1, 0, 0, 0)
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
    g = Storage('volcanoes', rootdir=savedir)
    c1 = g.get_substore('WIZ', '00', 'HHZ')
    c2 = g.get_substore('MDR', '00', 'BHZ')
    c3 = g.get_substore('MAVZ', '10', 'EHZ')
    c4 = g.get_substore('MMS', '66', 'BHZ')
    # Generate some fake data
    for _f in features1D:
        feat = generate_test_data(tstart=tstart,
                                  feature_names=[_f],
                                  ndays=ndays)
        for _c in g.stores:
            _c.save(feat)
            if _f == 'dsar':
                _c.save_labels(get_labels(feat.dsar,
                                          float(feat.dsar.quantile(0.85))))
    for _n, _f in features2D:
        feat = generate_test_data(tstart=tstart,
                                  feature_names=[_n],
                                  ndays=ndays,
                                  nfreqs=8,
                                  freq_names=[_f],
                                  dim=2)
        for _c in g.stores:
            _c.save(feat)

    alg = generate_test_data(tstart=tstart,
                             feature_names=['autoencoder'],
                             ndays=ndays,
                             nfreqs=5,
                             freq_names=['cluster'],
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
    return client, g.get_substore('MDR', '00', 'BHZ')


@pytest.fixture(scope='module')
def setup_api_zarr(tmp_path_factory):
    savedir = tmp_path_factory.mktemp('vumt_test_tmp_zarr', numbered=True)
    g = Storage('volcanoes', rootdir=savedir, backend='zarr')
    from tonik.api import TonikAPI
    ta = TonikAPI(str(savedir), backend='zarr')
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    feat = generate_test_data(tstart=tstart,
                              feature_names=['ssam'],
                              ndays=ndays,
                              nfreqs=8,
                              freq_names=['frequency'],
                              dim=2)
    c = g.get_substore('MDR', '00', 'BHZ')
    c.save(feat)
    return client, c


@pytest.fixture(scope='module')
def setup_multi_dimensional(tmp_path_factory):
    tempdir = tmp_path_factory.mktemp('test_xarray2zarr_high_dimensionality')
    test_data = xr.DataArray(
        np.random.rand(143, 3, 24, 6),
        dims=['datetime', 'channel', 'order_1', 'order_2'],
        coords={
            'datetime': pd.date_range(start='2022-07-18', periods=143, freq='10min'),
            'channel': np.arange(3),
            'order_1': np.arange(24),
            'order_2': np.arange(6)
        },
    )
    test_data_2 = xr.DataArray(
        np.random.rand(10, 3, 24, 6),
        dims=['datetime', 'channel', 'order_1', 'order_2'],
        coords={
            'datetime': pd.date_range(start='2022-07-20', periods=10, freq='10min'),
            'channel': np.arange(3),
            'order_1': np.arange(24),
            'order_2': np.arange(6)
        },
    )
    return tempdir, test_data, test_data_2
