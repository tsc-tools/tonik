import configparser
from io import StringIO
import json
import os

import numpy as np
import pandas as pd
import pytest


def test_read_1Dfeature(setup_api):
    client, l = setup_api
    params = dict(name='rsam',
                  group='volcanoes',
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='1D')
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert len(np.unique(df.index)) ==  5
    assert len(np.unique(df['freqs'])) == 8 

def test_read_filterbank(setup_api):
    client, l = setup_api
    params = dict(name='filterbank',
                  group='volcanoes',
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
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
                  site='MDR',
                  sensor='00',
                  channel='BHZ',
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  resolution='full',
                  log=True,
                  normalise=True)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert np.nanmax(df['feature'].values) ==  1.
    assert np.nanmin(df['feature'].values) == 0.

@pytest.mark.xfail
def test_aggregate1DFeature(setup_api):
    client, fq = setup_api
    params = dict(name='rsam',
                  volcano='Mt Doom',
                  site='MDR',
                  channel='BHZ',
                  starttime=str(fq.starttime),
                  endtime=str(fq.endtime),
                  resolution=3600000, #given in ms seconds by Grafana (here 1 hr)
                  log=False)
    with client.stream("GET", "/feature", params=params) as r:
        r.read()
        txt = r.text

    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    assert df.index[1].value == 1448933100000000000
    assert df.index[2].value == 1448936700000000000

@pytest.mark.xfail
def test_featureEndpoint(setup_api):
    client, fq = setup_api
    with client.stream("GET", "/featureEndpoint") as r:
        r.read()
        txt = r.text 
    features = sorted(["sonogram", "predom_freq", "ssam", "bandwidth",
                       "filterbank", "central_freq", "rsam", "dsar",
                       "rsam_energy_prop", "autoencoder"])
    result_expected = {"scenario": "VUMT", 
                       "volcanoes": [
                            {"volcano": "Mt Doom",
                             "stations":[
                                 {"name":"MDR",
                                  "lat":-39.15,
                                  "lon":175.63,
                                  "channels":[
                                      {"name":"BHZ",
                                       "features": features
                                       }
                                    ]
                                }]
                            }
                        ]
                    }        
    result_test = json.loads(txt)
    assert result_test['volcanoes'][0] == result_expected['volcanoes'][0]
    assert len(result_test['volcanoes']) == 2
    params = dict(volcano="Mt Doom")
    with client.stream("GET", "/featureEndpoint", params=params) as r:
        r.read()
        txt = json.loads(r.text) 
    assert txt['stations'][0]['name'] == "MDR"
 
    params = dict(volcano="Mt Doom", station="MDR")
    with client.stream("GET", "/featureEndpoint", params=params) as r:
        r.read()
        txt = json.loads(r.text) 
    assert txt['channels'][0]['name'] == "BHZ"

    params = dict(volcano="Mt Doom", station="MDR", channel="BHZ")
    with client.stream("GET", "/featureEndpoint", params=params) as r:
        r.read()
        txt = json.loads(r.text) 
    assert sorted(txt['features']) == sorted(features)
 
    params = dict(volcano="Mt Doom", station="MDR", channel="BHZ", type='2D')
    with client.stream("GET", "/featureEndpoint", params=params) as r:
        r.read()
        txt = json.loads(r.text) 
    assert sorted(txt['features']) == sorted(['ssam', 'filterbank', 'sonogram']) 

