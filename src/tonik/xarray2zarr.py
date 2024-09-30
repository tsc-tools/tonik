import os

import xarray as xr


def xarray2zarr(xds, path, mode='w'):
    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        if not os.path.exists(fout) or mode == 'w':
            xds[feature].to_zarr(
                fout, group='original', mode='w')
        else:
            xds_existing = xr.open_zarr(fout, group='original')
            overlap = xds_existing.datetime.where(
                xds_existing.datetime == xds.datetime)
            if overlap.size > 0:
                xds.loc[dict(datetime=overlap)].to_zarr(
                    fout, group='original', mode='r+', region='auto')
                xds.drop_sel(datetime=overlap).to_zarr(
                    fout, group='original', mode='a', append_dim="datetime")
            else:
                xds[feature].to_zarr(
                    fout, group='original', append_dim='datetime')
