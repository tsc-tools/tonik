from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def generate_test_data(dim=1, ndays=30, nfreqs=10,
                       tstart=datetime.utcnow(),
                       feature_name=None,
                       freq_name=None, add_nans=True):
    """
    Generate a 1D or 2D feature for testing.
    """
    assert dim < 3
    assert dim > 0

    nints = ndays * 6 * 24
    dates = pd.date_range(tstart.strftime('%Y-%m-%d'), freq='10min', periods=nints)
    rs = np.random.default_rng(42)
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
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[dates], dims=['datetime'])})
    if dim == 2:
        if add_nans:
            data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_name is None:
            feature_name = 'ssam'
        if freq_name is None:
            freq_name = 'frequency'
        xrd = xr.Dataset({feature_name: xr.DataArray(data, coords=[freqs, dates], dims=[freq_name, 'datetime'])})
    xrd.attrs['starttime'] = dates[0].isoformat()
    xrd.attrs['endtime'] = dates[-1].isoformat()
    xrd.attrs['station'] = 'MDR'
    xrd.attrs['interval'] = '10min'
    return xrd