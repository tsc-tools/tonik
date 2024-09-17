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
    params = dict(group='volcanoes')
    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    features = sorted(["sonogram", "predom_freq", "ssam", "bandwidth",
                       "filterbank", "central_freq", "rsam", "dsar",
                       "rsam_energy_prop", "autoencoder"])
    result_expected = {"volcanoes": [
        {"MDR": [
            {"00": [
                {"BHZ": features}
            ]
            }
        ]
        }
    ]
    }
    result_test = json.loads(txt)
    assert result_test['volcanoes'][1] == result_expected['volcanoes'][0]

    with client.stream("GET", "/inventory", params=params) as r:
        r.read()
        txt = r.text
    result_test = json.loads(txt)
    test_features = result_test['volcanoes'][1]['MDR'][0]['00'][0]['BHZ']
    assert sorted(test_features) == features
