from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr


def generate_test_data(dim=1, ndays=30, nfreqs=10,
                       tstart=datetime.now(timezone.utc),
                       freq='10min', intervals=None,
                       feature_name=None, seed=42,
                       freq_name=None, add_nans=True):
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
    if dim == 1:
        if add_nans:
            data[idx_nan] = np.nan
        if feature_name is None:
            feature_name = 'rsam'
        xrd = xr.Dataset({feature_name: xr.DataArray(
            data, coords=[dates], dims=['datetime'])})
    if dim == 2:
        if add_nans:
            data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_name is None:
            feature_name = 'ssam'
        if freq_name is None:
            freq_name = 'frequency'
        xrd = xr.Dataset({feature_name: xr.DataArray(
            data, coords=[freqs, dates], dims=[freq_name, 'datetime'])})
    xrd.attrs['starttime'] = dates[0].isoformat()
    xrd.attrs['endtime'] = dates[-1].isoformat()
    xrd.attrs['station'] = 'MDR'
    xrd.attrs['interval'] = '10min'
    return xrd
