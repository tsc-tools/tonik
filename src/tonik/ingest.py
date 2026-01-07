# src/tonik/ingest.py
import json
import logging
import os
import pickle
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import xarray as xr

from .xarray2netcdf import xarray2netcdf
from .xarray2zarr import xarray2zarr

logger = logging.getLogger(__name__)

__all__ = ["enqueue_dataset", "IngestWorker"]


def _norm_timeseries(xds: xr.Dataset, timedim: str) -> xr.Dataset:
    xds = xds.sortby(timedim)
    xds = xds.drop_duplicates(timedim, keep='last')
    xds[timedim] = xds[timedim].astype('datetime64[ns]')
    return xds


def enqueue_dataset(data: xr.Dataset, target_path: str, *, backend: str,
                    ingest_config: dict, save_kwargs: Optional[dict] = None) -> dict:
    """
    Enqueue a dataset for ingestion.
    Parameters
    ----------
    data : xr.Dataset
        The dataset to enqueue.
    target_path : str
        The target path where the dataset should be saved.
    backend : str
        The backend to use for saving the dataset ('zarr' or 'netcdf').
    ingest_config : dict
        Configuration for the ingest queue, must include 'queue_path'.
    save_kwargs : Optional[dict], optional
        Additional keyword arguments to pass to the save function, by default None.
    Returns
    -------
    dict
        A message dictionary representing the enqueued dataset.
    """

    queue_path = ingest_config.get("queue_path")
    if not queue_path:
        raise ValueError("ingest_config must provide a 'queue_path'.")
    queue_path = os.path.abspath(queue_path)
    payload_dir = os.path.join(queue_path, "payloads")
    message_dir = os.path.join(queue_path, "messages")
    os.makedirs(payload_dir, exist_ok=True)
    os.makedirs(message_dir, exist_ok=True)
    timedim = save_kwargs.get(
        "timedim", "datetime") if save_kwargs else "datetime"

    if isinstance(data, xr.DataArray):
        name = data.name or "data"
        data = data.to_dataset(name=name)

    dataset = _norm_timeseries(data, timedim=timedim)
    entry_id = uuid.uuid4().hex
    payload_path = os.path.join(payload_dir, f"{entry_id}.nc")
    kwargs_path = os.path.join(payload_dir, f"{entry_id}.pkl")

    dataset.to_netcdf(payload_path, engine="h5netcdf")
    with open(kwargs_path, "wb") as handle:
        pickle.dump(save_kwargs or {}, handle)

    message = {
        "id": entry_id,
        "target_path": os.path.abspath(target_path),
        "backend": backend,
        "payload_path": payload_path,
        "kwargs_path": kwargs_path,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    tmp_path = os.path.join(message_dir, f"{entry_id}.json.tmp")
    final_path = os.path.join(message_dir, f"{entry_id}.json")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(message, handle)
    os.replace(tmp_path, final_path)
    logger.debug("Queued dataset %s for %s backend at %s",
                 entry_id, backend, target_path)
    return message


class IngestWorker:
    def __init__(self, queue_path: str, poll_interval: float = 10.0,
                 target_prefix: Optional[str] = None):
        self.queue_path = os.path.abspath(queue_path)
        self.messages_dir = os.path.join(self.queue_path, "messages")
        self.payloads_dir = os.path.join(self.queue_path, "payloads")
        os.makedirs(self.messages_dir, exist_ok=True)
        os.makedirs(self.payloads_dir, exist_ok=True)
        self.poll_interval = poll_interval
        self.target_prefix = os.path.abspath(
            target_prefix) if target_prefix else None

    def _iter_messages(self):
        for name in sorted(os.listdir(self.messages_dir)):
            if not name.endswith(".json"):
                continue
            msg_path = os.path.join(self.messages_dir, name)
            with open(msg_path, "r", encoding="utf-8") as handle:
                message = json.load(handle)
            target = os.path.abspath(message.get("target_path", ""))
            if self.target_prefix and not target.startswith(self.target_prefix):
                continue
            yield msg_path, message

    def run_once(self) -> int:
        processed = 0
        for msg_path, message in self._iter_messages():
            payload_path = message.get("payload_path")
            kwargs_path = message.get("kwargs_path")
            if not payload_path or not os.path.exists(payload_path):
                logger.warning(
                    "Missing payload for %s, dropping message", msg_path)
                os.remove(msg_path)
                if kwargs_path and os.path.exists(kwargs_path):
                    os.remove(kwargs_path)
                continue

            dataset = None
            try:
                with xr.open_dataset(payload_path, engine='h5netcdf') as ds_on_disk:
                    dataset = ds_on_disk.load()

                kwargs = {}
                if kwargs_path and os.path.exists(kwargs_path):
                    with open(kwargs_path, "rb") as handle:
                        kwargs = pickle.load(handle)

                backend = message.get("backend", "zarr")
                if backend == "zarr":
                    xarray2zarr(dataset, message["target_path"], **kwargs)
                elif backend == "netcdf":
                    xarray2netcdf(dataset, message["target_path"], **kwargs)
                else:
                    raise ValueError(f"Unsupported backend '{backend}'")
            except Exception as exc:
                logger.error("Failed to ingest %s: %s",
                             msg_path, exc, exc_info=True)
                continue
            finally:
                if dataset is not None:
                    dataset.close()

            os.remove(payload_path)
            if kwargs_path and os.path.exists(kwargs_path):
                os.remove(kwargs_path)
            os.remove(msg_path)
            processed += 1
        return processed

    def run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            processed = self.run_once()
            if processed == 0:
                stop_event.wait(self.poll_interval)
