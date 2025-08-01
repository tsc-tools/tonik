import logging
import os

import pandas as pd
import xarray as xr
try:
    from zarr.errors import PathNotFoundError
except ImportError:
    class PathNotFoundError(Exception):
        pass

from .utils import merge_arrays

logger = logging.getLogger(__name__)


def get_encoding(coords: xr.core.coordinates.DatasetCoordinates, chunks: int = 1) -> dict:
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
    if 'datetime' not in coords:
        raise ValueError(
            "Datetime coordinate not found in dataset coordinates.")

    dt = pd.infer_freq(coords['datetime'])
    if dt is None:
        dates = pd.to_datetime(coords['datetime'].values)
        dt = dates.diff().median()
    try:
        chunklength = int(pd.Timedelta('%dD' % chunks) / pd.Timedelta(dt))
    except ValueError:
        # pd.infer_freq leaves out the unit if it's 1
        chunklength = int(pd.Timedelta('1D') / pd.Timedelta("1%s" % dt))
    encoding = {}
    for key in coords:
        if key == 'datetime':
            encoding['datetime'] = {'chunks': (chunklength,)}
        else:
            encoding[key] = {'chunks': (coords[key].size,)}
    return encoding


def xarray2zarr(xds: xr.Dataset, path: str, mode: str = 'a', group='original',
                chunks: int = 1) -> None:
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

    Returns
    -------
    None
    """
    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        encoding = get_encoding(xds[feature].coords, chunks)
        if not os.path.exists(fout) or mode == 'w':
            xds[feature].to_zarr(
                fout, group=group, mode='w',
                encoding=encoding)
        else:
            try:
                xds_existing = xr.open_zarr(fout, group=group, chunks={})
            except (PathNotFoundError, FileNotFoundError):
                xds[feature].to_zarr(
                    fout, group=group, mode='a')
                continue
            if xds_existing.datetime[0] > xds.datetime[0] or xds_existing.datetime[-1] > xds.datetime[-1]:
                xda_new = merge_arrays(xds_existing[feature], xds[feature])
                xda_new.to_zarr(fout, group=group, mode='w', encoding=encoding)
            else:
                try:
                    overlap = xds_existing.datetime.where(
                        xds_existing.datetime == xds.datetime)
                    if overlap.size > 0:
                        xds[feature].loc[dict(datetime=overlap)].to_zarr(
                            fout, group=group, mode='r+', region='auto')
                        xds[feature].drop_sel(datetime=overlap).to_zarr(
                            fout, group=group, mode='a', append_dim="datetime")
                    else:
                        xds[feature].to_zarr(
                            fout, group=group, append_dim='datetime')
                except Exception as e:
                    msg = f"Appending {feature} to {fout} failed: {e}\n"
                    msg += "Attempting to merge the two datasets."
                    logger.error(msg)
                    # remove duplicate datetime entries
                    xda_new = merge_arrays(xds_existing[feature], xds[feature])
                    xda_new.to_zarr(fout, group=group,
                                    mode='w', encoding=encoding)
