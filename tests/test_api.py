import configparser
from io import StringIO
import json
import os

import numpy as np
import pandas as pd
import pytest


def test_errors(setup_api):
    client, l = setup_api
    params = dict(name='rsam',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime))
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    assert r.status_code == 422

    params = dict(group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime))
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    assert r.status_code == 422

    params = dict(group='volcanoes',
                  subdir=['this', 'doesnt', 'exist'])
    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    assert r.status_code == 404


def test_read_1Dfeature(setup_api):
    client, l = setup_api
    params = dict(name='rsam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime))
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('rsam').values)


def test_html_tags(setup_api):
    client, l = setup_api
    params = dict(name='rsam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime='2023-01-01T00%3A00%3A00.000Z',
                  endtime='2023-01-06T00%3A00%3A00.927Z')
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('rsam').values)


def test_read_ssam(setup_api):
    client, l = setup_api
    params = dict(name='ssam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('ssam').values.ravel(order='C'))

    params = dict(name='ssam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='1D')
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert len(np.unique(df.index)) == 5
    assert len(np.unique(df['freqs'])) == 8


def test_read_ssam_from_zarr(setup_api_zarr):
    client, l = setup_api_zarr
    params = dict(name='ssam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('ssam').values.ravel(order='C'))

    params = dict(name='ssam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='1D')
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert len(np.unique(df.index)) == 5
    assert len(np.unique(df['freqs'])) == 8


def test_read_filterbank(setup_api):
    client, l = setup_api
    params = dict(name='filterbank',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('filterbank').values.ravel(order='C'))


def test_log(setup_api):
    client, l = setup_api
    params = dict(name='filterbank',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=True)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         10*np.log10(l('filterbank').values.ravel(order='C')))


def test_autoencoder(setup_api):
    client, l = setup_api
    params = dict(name='autoencoder',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         l('autoencoder').values.ravel(order='C'))


def test_normalise(setup_api):
    client, l = setup_api
    params = dict(name='sonogram',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=True,
                  normalise=True)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert np.nanmax(df['feature'].values) == 1.
    assert np.nanmin(df['feature'].values) == 0.


def test_aggregate1DFeature(setup_api):
    client, fq = setup_api
    params = dict(name='rsam',
                  group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(fq.starttime),
                  endtime=str(fq.endtime),
                  # given in ms seconds by Grafana (here 1 hr)
                  resolution='1D',
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text

    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert pd.Timedelta(df.index.diff().mean()) > pd.Timedelta('10min')
    assert pd.Timedelta(df.index.diff().mean()) <= pd.Timedelta('1D')


def test_inventory(setup_api):
    client, fq = setup_api
    expected_feature_names = sorted(["sonogram", "predom_freq", "ssam", "bandwidth",
                                     "filterbank", "central_freq", "rsam", "dsar",
                                     "rsam_energy_prop", "autoencoder"])
    required_keys = {'name', 'recordCount',
                     'earliestRecord', 'latestRecord', 'url'}

    # --- tree view (default) ---
    params = dict(group='volcanoes')
    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    result_test = json.loads(txt)
    bhz_features = result_test['volcanoes']['MDR']['00']['BHZ']
    # each feature is now a dict keyed by feature name
    assert all(isinstance(f, dict) for f in bhz_features.values())
    assert sorted(bhz_features.keys()) == expected_feature_names
    for feat_name, feat in bhz_features.items():
        assert required_keys.issubset(feat.keys())
        assert feat['recordCount'] > 0
        assert 'feature' in feat['url']
        assert feat_name in feat['url']

    # --- tree=False returns top-level subdirs as plain strings ---
    params = dict(group='volcanoes', tree=False)
    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    result_test = json.loads(txt)
    assert sorted(result_test) == sorted(['MAVZ', 'WIZ', 'MDR', 'MMS'])

    # --- explicit subdir returns feature info dicts ---
    params = dict(group='volcanoes', subdir=['MDR', '00', 'BHZ'])
    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    result_test = json.loads(txt)
    assert all(isinstance(f, dict) for f in result_test)
    assert sorted(f['name'] for f in result_test) == expected_feature_names
    for feat in result_test:
        assert required_keys.issubset(feat.keys())
        assert feat['recordCount'] > 0
        assert 'MDR' in feat['url'] and '00' in feat['url'] and 'BHZ' in feat['url']


def test_labels(setup_api):
    client, fq = setup_api
    params = dict(group='volcanoes',
                  subdir=['MDR', '00', 'BHZ'],
                  starttime=str(fq.starttime),
                  endtime=str(fq.endtime))
    with client.stream("GET", "/labels", params=params) as r:
        r.read()
        txt = r.text
    result = json.loads(txt)
    assert 'time' in result['dsar'][0]
    assert 'timeEnd' in result['dsar'][0]
    assert 'title' in result['dsar'][0]
    assert 'description' in result['dsar'][0]
    assert 'tags' in result['dsar'][0]
    assert 'id' in result['dsar'][0]
