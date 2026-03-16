from datetime import datetime
import json
import logging
import os
import threading
from typing import Optional

import numpy as np
import torch
import xarray as xr

from .ingest import IngestWorker, enqueue_dataset
from .xarray2netcdf import xarray2netcdf
from .xarray2zarr import xarray2zarr


logger = logging.getLogger(__name__)


class Path(object):
    def __init__(self, name, parentdir, create=True, backend='zarr',
                 archive_starttime=datetime(2000, 1, 1), ingest_config=None):
        self.name = name
        self.create = create
        self.backend = backend
        self.archive_starttime = archive_starttime
        self.engine = 'h5netcdf' if self.backend == 'netcdf' else self.backend
        self.path = os.path.join(parentdir, name)
        if create:
            try:
                os.makedirs(self.path, exist_ok=True)
            except FileExistsError:
                pass
        else:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Path {self.path} not found")
        self.children = {}
        self.ingest_config = ingest_config.copy() if ingest_config else None

    def __str__(self):
        return self.path

    def __getitem__(self, key):
        if key is None:
            raise ValueError("Key cannot be None")
        try:
            return self.children[key]
        except KeyError:
            self.children[key] = Path(
                key, self.path, self.create, self.backend, self.archive_starttime,
                ingest_config=self.ingest_config)
            return self.children[key]

    def feature_path(self, feature):

        if self.backend == 'netcdf':
            file_ending = '.nc'
        elif self.backend == 'zarr':
            file_ending = '.zarr'
        _feature_path = os.path.join(self.path, feature + file_ending)
        if not os.path.exists(_feature_path):
            raise FileNotFoundError(f"File {_feature_path} not found")
            self.children[feature] = Path(feature + file_ending, self.path)
        return _feature_path

    def __call__(self, feature, group='original', metadata=False):
        """
        Request a particular feature

        :param feature: Feature name
        :type feature: str

        """
        filename = self.feature_path(feature)

        if metadata:
            with xr.open_dataset(filename, group='meta', engine=self.engine) as ds:
                return ds

        if self.endtime < self.starttime:
            raise ValueError('Startime has to be smaller than endtime.')

        logger.debug(
            f"Reading feature {feature} between {self.starttime} and {self.endtime}")

        xd_index = dict(datetime=slice(self.starttime, self.endtime))
        with xr.open_dataset(filename, group=group, engine=self.engine) as ds:
            rq = ds[feature].loc[xd_index].load()
            rq.attrs = ds.attrs

        return rq

    def load(self, *args, **kwargs):
        """
        Load a feature from disk
        """
        self.__call__(*args, **kwargs)

    def save(self, data, **kwargs):
        """
        Save a feature to disk
        """
        if self.ingest_config and self.ingest_config.get('queue_path'):
            enqueue_dataset(
                data,
                target_path=self.path,
                backend=self.backend,
                ingest_config=self.ingest_config,
                save_kwargs=kwargs,
            )
            logger.debug("Queued data for %s backend at %s",
                         self.backend, self.path)
            return

        if self.backend == 'netcdf':
            xarray2netcdf(data, self.path,
                          archive_starttime=self.archive_starttime, **kwargs)
        elif self.backend == 'zarr':
            xarray2zarr(data, self.path,
                        archive_starttime=self.archive_starttime, **kwargs)

    def shape(self, feature):
        """
        Get shape of a feature on disk
        """
        filename = self.feature_path(feature)
        with xr.open_dataset(filename, group='original', engine=self.engine) as ds:
            return ds[feature].sizes

    def save_labels(self, labels):
        """
        Save all labels. Labels are stored in a list of dictionaries with the following keys:
        {
            'time': 'time, or the beginning of the time window',
            'timeEnd': 'end of the time window [optional]',
            'title': 'title of the label',
            'text': 'A more detailed description of the label',
            'tags': 'tags to sort labels',
            'id': 'unique id of the label'
        }
        """
        filename = os.path.join(self.path, 'labels.json')
        with open(filename, 'w') as f:
            json.dump(labels, f)

    def get_labels(self):
        """
        Load all labels.
        """
        filename = os.path.join(self.path, 'labels.json')
        with open(filename) as f:
            return json.load(f)


class Storage(Path):
    """
    Query computed features

    :param rootdir: Path to parent directory.
    :type rootdir: str
    :param starttime: Begin of request
    :type starttime: :class:`datetime.datetime`
    :param endtime: Begin of request
    :type endtime: :class:`datetime.datetime`

    >>> import datetime
    >>> g = Storage('Whakaari', /tmp)
    >>> start = datetime.datetime(2012,1,1,0,0,0)
    >>> end = datetime.datetime(2012,1,2,23,59,59)
    >>> g.starttime = start
    >>> g.endtime = end
    >>> c = g.channel(site='WIZ', sensor='00', channel='HHZ')
    >>> rsam = c("rsam")
    """

    def __init__(self, name, rootdir, starttime=None, endtime=None, create=True, backend='netcdf',
                 ingest_config=None, archive_starttime=datetime(2000, 1, 1)):
        self.stores = set()
        self.starttime = starttime
        self.endtime = endtime
        self.archive_starttime = archive_starttime
        self._ingest_worker: Optional[IngestWorker] = None
        self._ingest_thread: Optional[threading.Thread] = None
        self._ingest_stop_event: Optional[threading.Event] = None
        super().__init__(name, rootdir, create, backend, archive_starttime,
                         ingest_config=ingest_config)

    def print_tree(self, site, indent=0, output=''):
        output += ' ' * indent + site.path + '\n'
        for site in site.children.values():
            output += self.print_tree(site, indent + 2)
        return output

    def __repr__(self):
        rstr = f"Group: {self.name}\n"
        rstr = self.print_tree(self, 0, rstr)
        return rstr

    def get_substore(self, *args):
        # return the store for a given site, sensor, or channel
        # if one of them is None return the store for the level above
        # if all are None return the root store
        try:
            st = self
            for arg in args:
                st = st[arg]
        except KeyError:
            return self

        st.starttime = self.starttime
        st.endtime = self.endtime
        self.stores.add(st)
        return st

    def from_directory(self):
        """
        Construct the storage group from the root directory
        """
        for root, dirs, files in os.walk(self.path):
            if files:
                try:
                    subdirs = root.split(self.path)[1].split(os.sep)[1:]
                except IndexError:
                    st = self.get_substore()
                else:
                    try:
                        st = self.get_substore(*subdirs)
                    except TypeError as e:
                        raise e
                for _f in files:
                    if _f.endswith('.nc'):
                        st.feature_path(_f.replace(
                            '.nc', '').replace('.zarr', ''))

    @staticmethod
    def directory_tree_to_dict(path):
        name = os.path.basename(path)
        if name.endswith('.zarr'):
            return name.replace('.zarr', '')
        elif os.path.isdir(path):
            dir_contents = os.listdir(path)
            if 'labels.json' in dir_contents:
                dir_contents.remove('labels.json')
            return {name: [Storage.directory_tree_to_dict(os.path.join(path, child)) for child in sorted(dir_contents)]}
        else:
            if name.endswith('.nc'):
                return name.replace('.nc', '')
            else:
                return

    def to_dict(self):
        """
        Convert the storage group to json
        """
        return Storage.directory_tree_to_dict(self.path)

    def get_starttime(self):
        return self.__starttime

    def set_starttime(self, time):
        if time is None:
            self.__starttime = None
            self.__sdate = None
            return
        self.__starttime = time
        self.__sdate = '{}{:02d}{:02d}'.format(time.year,
                                               time.month,
                                               time.day)
        for s in self.stores:
            if s is not self:
                s.starttime = time

    def get_endtime(self):
        return self.__endtime

    def set_endtime(self, time):
        if time is None:
            self.__endtime = None
            self.__edate = None
            return
        self.__endtime = time
        self.__edate = '{}{:02d}{:02d}'.format(time.year,
                                               time.month,
                                               time.day)
        for s in self.stores:
            if s is not self:
                s.endtime = time

    starttime = property(get_starttime, set_starttime)
    endtime = property(get_endtime, set_endtime)

    def _ensure_ingest_worker(self, poll_interval=None) -> IngestWorker:
        if not (self.ingest_config and self.ingest_config.get('queue_path')):
            raise RuntimeError(
                "Ingestion queue is not configured for this Storage instance.")

        if self._ingest_worker is None:
            queue_path = self.ingest_config['queue_path']
            poll = poll_interval or self.ingest_config.get(
                'poll_interval', 10.0)
            self._ingest_worker = IngestWorker(
                queue_path=queue_path,
                poll_interval=poll
            )
        elif poll_interval:
            self._ingest_worker.poll_interval = poll_interval
        return self._ingest_worker

    def run_ingest_once(self, poll_interval=None) -> int:
        worker = self._ensure_ingest_worker(poll_interval)
        return worker.run_once()

    def start_ingest_worker(self, *, background=True, poll_interval=None):
        worker = self._ensure_ingest_worker(poll_interval)
        if not background:
            return worker.run_once()
        if self._ingest_thread and self._ingest_thread.is_alive():
            return self._ingest_thread
        stop_event = threading.Event()
        thread = threading.Thread(
            target=worker.run_forever,
            kwargs={'stop_event': stop_event},
            daemon=True,
            name=f"tonik-ingest-{self.name}",
        )
        thread.start()
        self._ingest_thread = thread
        self._ingest_stop_event = stop_event
        return thread

    def stop_ingest_worker(self, timeout=None):
        if self._ingest_thread and self._ingest_thread.is_alive():
            if self._ingest_stop_event:
                self._ingest_stop_event.set()
            self._ingest_thread.join(timeout=timeout)
        self._ingest_thread = None
        self._ingest_stop_event = None

    def to_pytorch(self, features, window_size=1, stride=1):
        """
        Create a PyTorch Dataset from the Storage that can be used with DataLoader.

        :param features: List of feature names to include in the dataset
        :type features: list of str
        :param window_size: Number of consecutive time steps to include in each sample (default: 1)
        :type window_size: int
        :param stride: Step size between consecutive windows (default: 1)
        :type stride: int
        :return: PyTorch Dataset instance
        :rtype: TonikDataset

        >>> import datetime
        >>> from torch.utils.data import DataLoader
        >>> g = Storage('Whakaari', '/tmp')
        >>> g.starttime = datetime.datetime(2012,1,1,0,0,0)
        >>> g.endtime = datetime.datetime(2012,1,2,23,59,59)
        >>> dataset = g.to_pytorch(['rsam', 'dsar'])
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        ...     # batch is a 2D tensor with shape [batch_size, num_features]
        ...     pass
        """
        return TonikDataset(self, features, window_size, stride)


class TonikDataset:
    """
    PyTorch Dataset wrapper for Tonik Storage.

    This dataset loads time series data from Storage and returns it as PyTorch tensors.
    Each sample is a 2D tensor with shape [window_size, num_features] where features
    are ordered as provided to __init__.
    """

    def __init__(self, storage, features, window_size=1, stride=1):
        """
        Initialize the dataset.

        :param storage: Storage instance with starttime and endtime set
        :param features: List of feature names to load
        :param window_size: Number of consecutive time steps per sample
        :param stride: Step size between consecutive windows (default: 1)
        """
        if storage.starttime is None or storage.endtime is None:
            raise ValueError("Storage must have starttime and endtime set")

        self.storage = storage
        self.features = features if isinstance(features, list) else [features]
        self.window_size = window_size
        self.stride = stride

        # Load all features once to determine the data structure
        self.data = {}
        for feature in self.features:
            try:
                self.data[feature] = self.storage(feature)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Feature '{feature}' not found in storage")

        # Get the length from the first feature
        first_feature = self.features[0]
        total_timesteps = len(self.data[first_feature].datetime)

        # Calculate number of windows with given stride
        # Number of windows = (total_timesteps - window_size) // stride + 1
        if total_timesteps < window_size:
            self.length = 0
        else:
            self.length = (total_timesteps - window_size) // stride + 1

        if self.length <= 0:
            raise ValueError(
                f"Not enough data points for window_size={window_size} with stride={stride}. "
                f"Available data points: {total_timesteps}"
            )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        :param idx: Index of the sample
        :return: 2D PyTorch tensor with shape [window_size, num_features]
                 where features are in the order provided to __init__
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}")

        # Calculate the actual starting index based on stride
        start_idx = idx * self.stride

        # Collect values for all features in order
        feature_values = []
        for feature in self.features:
            data = self.data[feature]

            # Extract window of data
            if self.window_size == 1:
                values = data.isel(datetime=start_idx).values
            else:
                values = data.isel(datetime=slice(
                    start_idx, start_idx + self.window_size)).values

            # Handle NaN values by converting to numpy array first
            values = np.array(values, dtype=np.float32)

            # Ensure values is 1D
            if values.ndim == 0:
                values = values.reshape(1)

            feature_values.append(values)

        # Stack features along axis 1 to create [window_size, num_features] tensor
        # If window_size=1, each feature_values[i] has shape (1,)
        # Stack them to get shape [1, num_features], then we have [window_size, num_features]
        # Shape: [window_size, num_features]
        stacked = np.stack(feature_values, axis=-1)

        return torch.from_numpy(stacked)
