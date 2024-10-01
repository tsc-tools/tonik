import logging
import os

import xarray as xr

logger = logging.getLogger(__name__)


def xarray2zarr(xds, path, mode='a'):
    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        if not os.path.exists(fout) or mode == 'w':
            xds[feature].to_zarr(
                fout, group='original', mode='w')
        else:
            xds_existing = xr.open_zarr(fout, group='original')
            try:
                overlap = xds_existing.datetime.where(
                    xds_existing.datetime == xds.datetime)
                if overlap.size > 0:
                    xds[feature].loc[dict(datetime=overlap)].to_zarr(
                        fout, group='original', mode='r+', region='auto')
                    xds[feature].drop_sel(datetime=overlap).to_zarr(
                        fout, group='original', mode='a', append_dim="datetime")
                else:
                    xds[feature].to_zarr(
                        fout, group='original', append_dim='datetime')
            except Exception as e:
                msg = f"Appending {feature} to {fout} failed: {e}\n"
                msg += "Attempting to merge the two datasets."
                logger.error(msg)
                # remove duplicate datetime entries
                xda_existing = xds_existing[feature].drop_duplicates(
                    'datetime', keep='last')
                xda_new = xds[feature].drop_duplicates('datetime', keep='last')
                xda_new.combine_first(xda_existing).to_zarr(
                    fout, group='original', mode='w')
