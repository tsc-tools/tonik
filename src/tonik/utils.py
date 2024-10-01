from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def generate_test_data(dim=1, ndays=30, nfreqs=10,
                       tstart=datetime.now(),
                       freq='10min', intervals=None,
                       feature_names=None, seed=42,
                       freq_names=None, add_nans=True):
    """
    Generate a 1D or 2D feature for testing.
    """
    assert dim < 3
    assert dim > 0

    if intervals is None:
        nints = ndays * 6 * 24
    else:
        nints = intervals
    dates = pd.date_range(tstart, freq=freq, periods=nints)
    rs = np.random.default_rng(seed)
    # Random walk as test signal
    data = np.abs(np.cumsum(rs.normal(0, 8., len(dates))))
    if dim == 2:
        data = np.tile(data, (nfreqs, 1))
    # Add 10% NaNs
    idx_nan = rs.integers(0, nints-1, int(0.1*nints))

    xds_dict = {}
    if dim == 1:
        if add_nans:
            data[idx_nan] = np.nan
        if feature_names is None:
            feature_names = ['rsam', 'dsar']
        for feature in feature_names:
            xds_dict[feature] = xr.DataArray(
                data, coords=[dates], dims=['datetime'])
    if dim == 2:
        if add_nans:
            data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_names is None:
            feature_names = ['ssam', 'filterbank']
        if freq_names is None:
            freq_names = ['frequency', 'fbfrequency']

        for feature_name, freq_name in zip(feature_names, freq_names):
            xds_dict[feature_name] = xr.DataArray(
                data, coords=[freqs, dates], dims=[freq_name, 'datetime'])
    xds = xr.Dataset(xds_dict)
    xds.attrs['starttime'] = dates[0].isoformat()
    xds.attrs['endtime'] = dates[-1].isoformat()
    xds.attrs['station'] = 'MDR'
    xds.attrs['interval'] = '10min'
    return xds
