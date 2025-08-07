import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
try:
    from zarr.errors import PathNotFoundError
except ImportError:
    class PathNotFoundError(Exception):
        pass

from .utils import merge_arrays

logger = logging.getLogger(__name__)


def get_dt(times):
    """
    Infer the sampling of the time dimension.
    """
    pd_times = pd.to_datetime(times)
    dt = pd.infer_freq(pd_times)
    if dt is None:
        dt = pd_times.diff().median()
    try:
        dt = pd.Timedelta(dt)
    except ValueError:
        dt = pd.Timedelta(f"1{dt}")
    return dt


def get_encoding(xda: xr.DataArray, chunks: int = 1,
                 timedim: str = 'datetime') -> dict:
    """
    Determine the chunk size for the datetime dimension. Other dimensions are assumed to be 
    small enough to not require chunking.

    Parameters
    ----------
    coords : xr.core.coordinates.DatasetCoordinates
        Coordinates of the dataset.
    chunks : int, optional
        Number of chunks in days to divide the datetime dimension into, by default 1.
    """
    if timedim not in xda.coords:
        raise ValueError(
            f"Datetime coordinate {timedim} not found in dataset coordinates.")
    dt = get_dt(xda.coords[timedim])
    chunklength = int(pd.Timedelta('%dD' % chunks) / dt)
    encoding = {}
    shapes = []
    for key in xda.coords:
        if key == timedim:
            # encoding['datetime'] = {'chunks': (chunklength,)}
            shapes.append(chunklength)
        else:
            # encoding[key] = {'chunks': (xda.coords[key].size,)}
            shapes.append(xda.coords[key].size)
    encoding[xda.name] = {'chunks': tuple(shapes)}
    return encoding


def fill_time_gaps(xds: xr.Dataset, timedim: str = 'datetime') -> xr.Dataset:
    """
    Fill gaps in time series with NaN values by reindexing to a complete datetime range.

    Parameters
    ----------
    xds : xr.Dataset
        Input dataset with potential time gaps
    freq : str, optional
        Frequency string (e.g., 'H', 'D', '15min'). If None, will try to infer.
    timedim : str
        Name of the time dimension, by default 'datetime'

    Returns
    -------
    xr.Dataset
        Dataset with gaps filled with NaN
    """
    if timedim not in xds.coords:
        raise ValueError(
            f"{timedim} coordinate not found in dataset coordinates.")

    # Infer sample interval
    dt = get_dt(xds.coords[timedim])
    start_time = xds[timedim].values[0]
    end_time = xds[timedim].values[-1]
    complete_time = pd.date_range(start=start_time, end=end_time, freq=dt)

    # Reindex to fill gaps with NaN
    return xds.reindex({timedim: complete_time}, method='ffill')


def fill_time_gaps_between_datasets(xds_existing: xr.DataArray, xds_new: xr.DataArray,
                                    timedim: str = 'datetime') -> xr.DataArray:
    """
    Fill gaps between existing and new datasets.

    Parameters
    ----------
    xds_existing : xr.Dataset
        Existing dataset on disk
    xds_new : xr.Dataset  
        New dataset to append
    timedim : str
        Name of the time dimension

    Returns
    -------
    xr.Dataset
        Combined dataset with gaps filled
    """
    # Get time ranges
    existing_end = xds_existing[timedim].values
    new_start = xds_new[timedim].values[0]

    # Infer frequency from existing data
    dt = get_dt(xds_new.coords[timedim])
    gap_start = existing_end + dt
    gap_end = new_start - dt
    if gap_start <= gap_end:
        gap_times = pd.date_range(start=gap_start, end=gap_end, freq=dt)

        # Create NaN array with same shape as variable but for gap times
        gap_shape = (len(gap_times),) + \
            xds_new.shape[1:]  # Skip time dimension
        gap_values = np.full(gap_shape, np.nan)

        # Create coordinates for gap dataset
        gap_coords = {timedim: gap_times}
        for coord_name, coord in xds_new.coords.items():
            if coord_name != timedim:
                gap_coords[coord_name] = coord

        gap_data = xr.DataArray(
            gap_values,
            coords=gap_coords,
            dims=xds_new.dims,
            name=xds_new.name
        )

        # Combine: existing + gap + new
        combined = xr.concat([gap_data, xds_new], dim=timedim)
        return combined
    else:
        return xds_new


def xarray2zarr(xds: xr.Dataset, path: str, mode: str = 'a', group='original',
                chunks: int = 1, timedim: str = 'datetime', fill_gaps: bool = False) -> None:
    """
    Write xarray dataset to zarr files.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to write.
    path : str
        Path to write the dataset.
    mode : str, optional
        Write mode, by default 'a'.
    group : str, optional
        Group name, by default 'original'
    chunks : int, optional
        Chunk size as the number of days.
    timedim : str
        Name of the time dimension, by default 'datetime'
    fill_gaps : bool, optional
        Whether to fill time gaps with NaN before writing, by default False

    Returns
    -------
    None
    """
    # Fill gaps if requested
    if fill_gaps:
        xds = fill_time_gaps(xds, timedim=timedim)

    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        encoding = get_encoding(xds[feature], chunks)
        if not os.path.exists(fout) or mode == 'w':
            xds[feature].to_zarr(
                fout, group=group, mode='w',
                encoding=encoding)
        else:
            try:
                xds_existing = xr.open_zarr(fout, group=group)

                # Fill gaps in existing data too if requested
                if fill_gaps:
                    xds_existing = fill_time_gaps(
                        xds_existing, timedim=timedim)

            except (PathNotFoundError, FileNotFoundError, KeyError):
                xds[feature].to_zarr(
                    fout, group=group, encoding=encoding)
                continue
            if xds_existing.datetime[0] > xds.datetime[0] or xds_existing.datetime[-1] > xds.datetime[-1]:
                xda_new = merge_arrays(xds_existing[feature], xds[feature])
                if fill_gaps:
                    xda_new = fill_time_gaps(xda_new, timedim=timedim)[feature]
                xda_new.to_zarr(fout, group=group, mode='w', encoding=encoding)
            else:
                try:
                    overlap = xds_existing.datetime.where(
                        xds_existing[timedim] == xds[timedim])
                    if overlap.size > 0:
                        xds[feature].loc[dict(datetime=overlap)].to_zarr(
                            fout, group=group, mode='r+', region='auto')
                        xds[feature].drop_sel(datetime=overlap).to_zarr(
                            fout, group=group, append_dim=timedim)
                    else:
                        xda_new = fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: -1}),
                                                                  xds[feature])
                        xda_new.to_zarr(fout, group=group,
                                        mode='a', append_dim=timedim)
                except Exception as e:
                    msg = f"Appending {feature} to {fout} failed: {e}\n"
                    msg += "Attempting to merge the two datasets."
                    logger.error(msg)
                    # remove duplicate datetime entries
                    xda_new = merge_arrays(xds_existing[feature], xds[feature])
                    if fill_gaps:
                        xda_new = fill_time_gaps(
                            xda_new, timedim=timedim)[feature]
                    xda_new.to_zarr(fout, group=group,
                                    mode='w', encoding=encoding)
