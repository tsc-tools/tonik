import json
import logging
import logging.config
import os

import xarray as xr

from .xarray2netcdf import xarray2netcdf
from .xarray2zarr import xarray2zarr

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {  # The formatter name, it can be anything that I wish
            # What to add in the message
            "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d " "%(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
        "json": {  # The formatter name
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",  # The class to instantiate!
            # Json is more complex, but easier to read, display all attributes!
            "format": """
                    asctime: %(asctime)s
                    created: %(created)f
                    filename: %(filename)s
                    funcName: %(funcName)s
                    levelname: %(levelname)s
                    levelno: %(levelno)s
                    lineno: %(lineno)d
                    message: %(message)s
                    module: %(module)s
                    msec: %(msecs)d
                    name: %(name)s
                    pathname: %(pathname)s
                    process: %(process)d
                    processName: %(processName)s
                    relativeCreated: %(relativeCreated)d
                    thread: %(thread)d
                    threadName: %(threadName)s
                    exc_info: %(exc_info)s
                """,
            "datefmt": "%Y-%m-%d %H:%M:%S",  # How to display dates
        },
    },
    "handlers": {
        "simple": {  # The handler name
            "formatter": "default",  # Refer to the formatter defined above
            "class": "logging.StreamHandler",  # OUTPUT: Same as above, stream to console
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "storage": {  # The name of the logger, this SHOULD match your module!
            "level": "DEBUG",  # FILTER: only INFO logs onwards from "tryceratops" logger
            "handlers": [
                "simple",  # Refer the handler defined above
            ],
        },
    },
    "root": {
        "level": "INFO",  # FILTER: only INFO logs onwards
        "handlers": [
            "simple",  # Refer the handler defined above
        ]
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("__name__")


class Path(object):
    def __init__(self, name, parentdir, create=True, backend='zarr'):
        self.name = name
        self.create = create
        self.backend = backend
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

    def __str__(self):
        return self.path

    def __getitem__(self, key):
        if key is None:
            raise ValueError("Key cannot be None")
        try:
            return self.children[key]
        except KeyError:
            self.children[key] = Path(
                key, self.path, self.create, self.backend)
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

    def __call__(self, feature, group='original'):
        """
        Request a particular feature

        :param feature: Feature name
        :type feature: str

        """
        if self.endtime < self.starttime:
            raise ValueError('Startime has to be smaller than endtime.')

        filename = self.feature_path(feature)

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
        if self.backend == 'netcdf':
            xarray2netcdf(data, self.path, **kwargs)
        elif self.backend == 'zarr':
            xarray2zarr(data, self.path, **kwargs)

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

    def __init__(self, name, rootdir, starttime=None, endtime=None, create=True, backend='netcdf'):
        self.stores = set()
        self.starttime = starttime
        self.endtime = endtime
        super().__init__(name, rootdir, create, backend)

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
