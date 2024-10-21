import logging
import os

import xarray as xr
from zarr.errors import PathNotFoundError

from .utils import merge_arrays

logger = logging.getLogger(__name__)


def xarray2zarr(xds: xr.Dataset, path: str, mode: str = 'a', group='original'):
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

    Returns
    -------
    None
    """
    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        if not os.path.exists(fout) or mode == 'w':
            xds[feature].to_zarr(
                fout, group=group, mode='w')
        else:
            try:
                xds_existing = xr.open_zarr(fout, group=group)
            except PathNotFoundError:
                xds[feature].to_zarr(fout, group=group, mode='a')
                continue
            if xds_existing.datetime[0] > xds.datetime[0] or xds_existing.datetime[-1] > xds.datetime[-1]:
                xda_new = merge_arrays(xds_existing[feature], xds[feature])
                xda_new.to_zarr(fout, group=group, mode='w')
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
                    xda_new.to_zarr(fout, group=group, mode='w')
