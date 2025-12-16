from datetime import datetime, timezone
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

from .utils import merge_arrays, fill_time_gaps, get_dt

logger = logging.getLogger(__name__)


def get_chunks(xda: xr.DataArray, chunks: int = 1,
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
    return chunklength


def fill_time_gaps_between_datasets(xds_existing: xr.DataArray, xds_new: xr.DataArray, mode: str,
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
    if mode not in ['a', 'p']:
        raise ValueError(
            'Mode has to be either "a" for append or "p" for prepend')

    # get the sample interval
    dt = get_dt(xds_new.coords[timedim])

    existing_endpoint = xds_existing[timedim].values
    # Get time ranges
    if mode == 'a':
        gap_start = existing_endpoint + dt
        gap_end = xds_new[timedim].values[0] - dt
    elif mode == 'p':
        gap_end = existing_endpoint - dt
        gap_start = xds_new[timedim].values[-1] + dt

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
        if mode == 'a':
            combined = xr.concat([gap_data, xds_new], dim=timedim)
        elif mode == 'p':
            combined = xr.concat([xds_new, gap_data], dim=timedim)
        return combined
    else:
        return xds_new


def _build_append_payload_full_chunks(payload: xr.DataArray, mode: str,
                                      chunklen: int, timedim: str = "datetime") -> xr.DataArray:
    """
    Construct the sequence to append so that the final total length is a multiple of `chunklen`

    Parameters
    ----------
    payload : xr.DataArray
        DataArray to append
    mode : str
        'a' for append, 'p' for prepend
    chunklen : int
        Chunk length in number of time steps
    timedim : str
        Name of the time dimension

    Returns
    -------
    xr.DataArray
        Padded DataArray with length a multiple of chunklen
    """
    if mode not in ['a', 'p']:
        raise ValueError(
            'Mode has to be either "a" for append or "p" for prepend')

    # pad the tail so that payload_len % chunklen == 0
    pay_len = payload.sizes[timedim]
    need = -pay_len % chunklen  # 0..chunklen-1

    if need > 0:
        dt = get_dt(payload.coords[timedim])
        if mode == 'a':
            start = payload[timedim].values[-1] + dt
        elif mode == 'p':
            start = payload[timedim].values[0] - (need+1)*dt
        pad_times = pd.date_range(start=start, periods=need, freq=dt)
        pad_shape = []
        for i, d in enumerate(payload.dims):
            if d == timedim:
                pad_shape.append(need)
            else:
                pad_shape.append(payload.shape[i])
        pad_vals = np.full(pad_shape, np.nan)
        pad_coords = {timedim: pad_times}
        for c in payload.coords:
            if c != timedim:
                pad_coords[c] = payload.coords[c]
        pad_da = xr.DataArray(pad_vals, coords=pad_coords,
                              dims=payload.dims, name=payload.name, attrs=payload.attrs)
        if mode == 'a':
            payload = xr.concat([payload, pad_da], dim=timedim)
        elif mode == 'p':
            payload = xr.concat([pad_da, payload], dim=timedim)
        payload = payload.chunk({timedim: chunklen})
    return payload


def _update_meta_data(fout: str,
                      last_datapoint: np.datetime64,
                      resolution: float | None = None,
                      meta_group: str = "meta") -> None:
    """
    Append current update time (and last_datapoint) to meta group.

    Parameters
    ----------
    fout : str
        Base zarr store path (per-variable .zarr directory).
    last_datapoint : np.datetime64
        Latest data time in the feature.
    resolution : float | None
        Optional time resolution (hours) to store once.
    meta_group : str
        Group name for metadata.
    """

    now = np.datetime64(datetime.now(tz=timezone.utc), 'ns')
    new_update = xr.DataArray([now],
                              coords={'update': [now]},
                              dims=['update'],
                              name='update_log')
    new_last = xr.DataArray([last_datapoint],
                            coords={'endtime': [now]},
                            dims=['endtime'],
                            name='last_datapoint')

    try:
        meta = xr.open_zarr(fout, group=meta_group, chunks=None)
        # Existing vars -> concatenate
        update_old = meta.get('update_log')
        last_old = meta.get('last_datapoint')
        res_da_old = meta.get('resolution').values[()]
        new_update = xr.concat([update_old, new_update], dim='update')
        new_last = xr.concat([last_old, new_last], dim='endtime')
        if abs(resolution - res_da_old) > 1e-5:
            raise ValueError(f"Resolution mismatch for {fout}: "
                             f"{res_da_old} != {resolution}")
        res_da = xr.DataArray(resolution, name='resolution')
    except Exception:
        # First creation
        res_da = xr.DataArray(
            resolution, name='resolution') if resolution is not None else None

    vars = {'update_log': new_update, 'last_datapoint': new_last}
    if res_da is not None:
        vars['resolution'] = res_da
    xr.Dataset(vars).to_zarr(fout, group=meta_group, mode='w')


def xarray2zarr(xds: xr.Dataset, path: str, mode: str = 'a', group='original',
                chunks: int = 10, timedim: str = 'datetime') -> None:
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

    if timedim not in xds.dims:
        raise ValueError(f"{timedim} dimension not found in Dataset.")

    # Fill gaps
    xds = xds.drop_duplicates(timedim, keep='last')
    xds = fill_time_gaps(xds, timedim=timedim)

    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        # nchunks = get_chunks(xds[feature], chunks)
        nchunks = chunks
        last_dp = xds[feature][timedim].values[-1]
        _update_meta_data(fout, last_dp, resolution=float(
            get_dt(xds[timedim]) / pd.Timedelta(1, 'h')))
        try:
            xds_existing = xr.open_zarr(fout, group=group)
            has_store = True
        except (PathNotFoundError, FileNotFoundError, KeyError):
            has_store = False

        if not has_store:
            xda_new = _build_append_payload_full_chunks(
                xds[feature], 'a', nchunks)
            xda_new.to_zarr(fout, group=group, mode='w',
                            write_empty_chunks=True)
            continue

        if xds_existing[timedim][0] > xds[timedim][-1]:
            logger.debug("Prepending data to existing zarr store.")
            xda_new = fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: 0}),
                                                      xds[feature], mode='p')
            xda_new = _build_append_payload_full_chunks(
                xda_new, 'p', nchunks)
            combined = xda_new.combine_first(xds_existing[feature]).compute()
            combined.chunk({timedim: nchunks}).to_zarr(fout, group=group, mode='w',
                                                       write_empty_chunks=True)

        elif xds_existing[timedim][-1] < xds[timedim][0]:
            logger.debug("Appending data to existing zarr store.")
            xda_new = fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: -1}),
                                                      xds[feature], mode='a')
            xda_new = _build_append_payload_full_chunks(
                xda_new, 'a', nchunks)
            xda_new.to_zarr(fout, group=group, mode='a',
                            append_dim=timedim)

        elif xds_existing[timedim][0] > xds[timedim][0] and xds_existing[timedim][-1] < xds[timedim][-1]:
            logger.debug(
                "Data in zarr store contained in new data. Rewriting zarr store.")
            xda_new = _build_append_payload_full_chunks(
                xds[feature], 'a', nchunks)
            xda_new.to_zarr(fout, group=group, mode='w',
                            write_empty_chunks=True)

        else:
            logger.debug("Data in zarr store overlaps with new data.")
            logger.debug(
                f"Endtime of existing data: {xds_existing[timedim][-1].values}")
            logger.debug(f"Starttime of new data: {xds[timedim][0].values}")
            xds_existing = xds_existing.drop_duplicates(timedim, keep='last')
            overlap = xds_existing[timedim].where(
                xds_existing[timedim] == xds[timedim])
            xds[feature].loc[{timedim: overlap}].to_zarr(
                fout, group=group, mode='r+', region='auto')
            remainder = xds[feature].drop_sel({timedim: overlap})
            if remainder.sizes[timedim] > 0:
                mode = 'a'
                if remainder[timedim][-1] < xds_existing[timedim][0]:
                    mode = 'p'
                xda_new = fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: 0}),
                                                          xds[feature], mode=mode)
                xda_new = _build_append_payload_full_chunks(
                    xda_new, mode, nchunks)
                xda_new.to_zarr(fout, group=group, mode='a',
                                append_dim=timedim)
