from typing import List
from datetime import datetime, timezone, timedelta

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


def merge_arrays(xds_old: xr.DataArray, xds_new: xr.DataArray,
                 resolution: float = None) -> xr.DataArray:
    """
    Merge two xarray datasets with the same datetime index.

    Parameters
    ----------
    xds_old : xr.DataArray
        Old array.
    xds_new : xr.DataArray
        New array.
    resolution : float
        Time resolution in hours.

    Returns
    -------
    xr.DataArray
        Merged array.
    """
    xda_old = xds_old.drop_duplicates(
        'datetime', keep='last')
    xda_new = xds_new.drop_duplicates(
        'datetime', keep='last')
    xda_new = xda_new.combine_first(xda_old)
    if resolution is not None:
        new_dates = pd.date_range(
            xda_new.datetime.values[0],
            xda_new.datetime.values[-1],
            freq=f'{resolution}h')
        xda_new = xda_new.reindex(datetime=new_dates)
    return xda_new


def extract_consecutive_integers(nums: List[int]) -> List[List[int]]:
    """
    Extract consecutive integers from a list of integers.
    """
    if not len(nums) > 0:
        return []

    nums.sort()  # Sort the array
    result = []
    temp = [nums[0]]  # Initialize the first group with the first number

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:  # Check if consecutive
            temp.append(nums[i])
        else:
            result.append(temp)  # Add the current group to the result
            temp = [nums[i]]  # Start a new group

    result.append(temp)  # Add the last group
    return result


def get_labels(xda: xr.DataArray, threshold: float) -> dict:
    """
    Generate labels for time windows where the values are exceeding the threshold.
    """
    # add labels to time windows where the values are exceeding the 85th percentile
    labels = {xda.name: []}
    idx = np.where(xda > threshold)[0]
    window_list = extract_consecutive_integers(idx)
    ids = 0
    for _w in window_list:
        timeEnd = None
        # Convert to unix time milliseconds
        timeStart = xda.datetime.isel(dict(datetime=_w[0])).values.astype(
            'datetime64[ms]').astype(int)
        timeStart = int(timeStart)
        if len(_w) > 1:
            timeEnd = xda.datetime.isel(
                dict(datetime=_w[-1])).values.astype('datetime64[ms]').astype(int)
            timeEnd = int(timeEnd)
        label = dict(time=timeStart,
                     timeEnd=timeEnd,
                     title='Greater 85th percentile',
                     description='Values exceed 85th percentile',
                     tags=['anomaly'],
                     id=ids)
        ids += 1
        labels[xda.name].append(label)
    return labels


def main():
    from tonik import Storage
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Generating test data")
    rootdir = '/tmp'
    g = Storage('volcanoes', rootdir=rootdir)
    st1 = g.get_substore('Mt Doom', 'MDR', '00', 'BHZ')
    st2 = g.get_substore('Misty Mountain', 'MMS', '10', 'HHZ')

    # Get start time of the current 10 minute window
    tstart = datetime.now(timezone.utc).timestamp()
    tstart -= tstart % 600
    tstart = datetime.fromtimestamp(tstart, timezone.utc)
    tstart -= timedelta(days=30)
    tstart = tstart.replace(tzinfo=None)
    logger.info(f"Start time: {tstart}")

    # Generate test data
    xdf_1D = generate_test_data(dim=1, tstart=tstart, add_nans=False, seed=42)
    xdf_1D_1 = generate_test_data(
        dim=1, tstart=tstart, add_nans=False, seed=24)
    xdf_2D = generate_test_data(
        dim=2, tstart=tstart, add_nans=False, seed=1234)
    xdf_2D_1 = generate_test_data(
        dim=2, tstart=tstart, add_nans=False, seed=4321)
    logger.info("Saving test data to " + st1.path)
    st1.save(xdf_1D)
    st1.save(xdf_2D)
    logger.info("Saving test data to " + st2.path)
    st2.save(xdf_1D_1)
    st2.save(xdf_2D_1)

    # add labels to RSAM time windows where the values are exceeding the 85th percentile
    logger.info("Adding labels to DSAR data")
    labels = get_labels(xdf_1D.dsar, float(xdf_1D.dsar.quantile(0.85)))
    logger.info("Saving labels to " + st1.path + '/labels.json')
    st1.save_labels(labels)


if __name__ == '__main__':
    main()
