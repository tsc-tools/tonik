from datetime import datetime, timedelta
import glob
import logging
import logging.config
import os
import re
import tempfile

import pandas as pd
import xarray as xr

from .xarray2hdf5 import xarray2hdf5


ERROR_LOG_FILENAME = "tonik.log"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {      
        "default": {  # The formatter name, it can be anything that I wish
            "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d " "%(levelname)s %(message)s",  #  What to add in the message
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
        "logfile": {  # The handler name
            "formatter": "json",  # Refer to the formatter defined above
            "level": "ERROR",  # FILTER: Only ERROR and CRITICAL logs
            "class": "logging.handlers.RotatingFileHandler",  # OUTPUT: Which class to use
            "filename": ERROR_LOG_FILENAME,  # Param for class above. Defines filename to use, load it from constant
            "backupCount": 2,  # Param for class above. Defines how many log files to keep as it grows
        }, 
        "simple": {  # The handler name
            "formatter": "default",  # Refer to the formatter defined above
            "class": "logging.StreamHandler",  # OUTPUT: Same as above, stream to console
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": { 
        "zizou": {  # The name of the logger, this SHOULD match your module!
            "level": "DEBUG",  # FILTER: only INFO logs onwards from "tryceratops" logger
            "handlers": [
                "simple",  # Refer the handler defined above
            ],
        },
    },
    "root": {
        "level": "ERROR",  # FILTER: only INFO logs onwards
        "handlers": [
            "logfile",  # Refer the handler defined above
        ]
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("__name__")


class Path(object):
    def __init__(self, name, parentdir):
        self.name = name
        self.path = os.path.join(parentdir, name)
        os.makedirs(self.path, exist_ok=True)
        self.children = {}
    
    def __str__(self):
        return self.path
    
    def __getitem__(self, key):
        try:
            return self.children[key]
        except KeyError:
            self.children[key] = Path(key, self.path)
            return self.children[key]

    def feature_path(self, feature):
        _feature_path = os.path.join(self.path, feature + ".nc")
        if not os.path.exists(_feature_path):
            raise FileNotFoundError(f"File {_feature_path} not found")
        self.children[feature] = _feature_path
        return _feature_path

    def __call__(self, feature, stack_length=None, interval='10min'):
        """
        Request a particular feature

        :param feature: Feature name
        :type feature: str
        :param stack_length: length of moving average in time
        :type stack_length: str

        """
        if self.endtime <= self.starttime:
            raise ValueError('Startime has to be smaller than endtime.')

        feature = feature.lower()
        filename = self.feature_path(feature)

        logger.debug(f"Reading feature {feature} between {self.starttime} and {self.endtime}")
        num_periods = None
        if stack_length is not None:
            valid_stack_units = ['W', 'D', 'H', 'T', 'min', 'S']
            if not re.match(r'\d*\s*(\w*)', stack_length).group(1)\
                   in valid_stack_units:
                raise ValueError(
                    'Stack length should be one of: {}'.
                        format(', '.join(valid_stack_units))
                )

            if pd.to_timedelta(stack_length) < pd.to_timedelta(interval):
                raise ValueError('Stack length {} is less than interval {}'.
                                 format(stack_length, interval))

            # Rewind starttime to account for stack length
            self.starttime -= pd.to_timedelta(stack_length)

            num_periods = (pd.to_timedelta(stack_length)/
                           pd.to_timedelta(interval))
            if not num_periods.is_integer():
                raise ValueError(
                    'Stack length {} / interval {} = {}, but it needs'
                    ' to be a whole number'.
                        format(stack_length, interval, num_periods))

        xd_index = dict(datetime=slice(self.starttime, self.endtime))
        with xr.open_dataset(filename, group='original', engine='h5netcdf') as ds:
            ds.sortby("datetime")
            rq = ds.loc[xd_index].load()

        # Stack features
        if stack_length is not None:
            logger.debug("Stacking feature...")
            try:
                xdf = rq[feature].rolling(datetime=int(num_periods),
                                        center=False,
                                        min_periods=1).mean()
                # Return requested timeframe to that defined in initialisation
                self.starttime += pd.to_timedelta(stack_length)
                xdf_new = xdf.loc[
                        self.starttime:
                        self.endtime-pd.to_timedelta(interval)]
                xdf_new = xdf_new.rename(feature)
            except ValueError as e:
                logger.error(e)
                logger.error('Stack length {} is not valid for feature {}'.
                             format(stack_length, feature))
            else:
                return xdf_new

        return rq[feature]

    def load(self, *args, **kwargs):
        """
        Load a feature from disk
        """
        self.__call__(*args, **kwargs)

    def save(self, data):
        """
        Save a feature to disk
        """
        xarray2hdf5(data, self.path)


class StorageGroup(Path):
    """
    Query computed features

    :param rootdir: Path to parent directory.
    :type rootdir: str
    :param starttime: Begin of request
    :type starttime: :class:`datetime.datetime`
    :param endtime: Begin of request
    :type endtime: :class:`datetime.datetime`

    >>> import datetime
    >>> g = Group('Whakaari')
    >>> start = datetime.datetime(2012,1,1,0,0,0)
    >>> end = datetime.datetime(2012,1,2,23,59,59)
    >>> g.starttime = start
    >>> g.endtime = end
    >>> c = g.channel(site='WIZ', sensor='00', channel='HHZ')
    >>> rsam = c("rsam")
    """
    def __init__(self, name, rootdir=None, starttime=None, endtime=None):
        self.channels = set() 
        self.starttime = starttime
        self.endtime = endtime
        super().__init__(name, rootdir)

    def __repr__(self):
        rstr = f"Group: {self.name}\n"
        last_site = False
        for j, site in enumerate(self.children.values()):
            if j == len(self.children) - 1:
                last_site = True
            rstr += f"|__ {site.name}\n"
            last_sensor = False
            for i, sensor in enumerate(site.children.values()):
                if i == len(site.children) - 1:
                    last_sensor = True
                rstr += (" " if last_site else "|") + f"  |__ {sensor.name}\n"
                for k, channel in enumerate(sensor.children.values()):
                    rstr += ("   " if last_site else "|  ") 
                    rstr += ("   " if last_sensor else "|  ") 
                    rstr += f"|__ {channel.name}\n"
        return rstr

    def site(self, site):
        return self[site]
    
    def sensor(self, site, sensor):
        return self[site][sensor]
    
    def channel(self, site, sensor, channel):
        c = self[site][sensor][channel]
        c.starttime = self.starttime
        c.endtime = self.endtime
        self.channels.add(c)
        return c

    def from_directory(self):
        feature_files = glob.glob(os.path.join(self.path, '**', '*.nc'),
                                  recursive=True)
        for _f in feature_files:
            subdir = _f.split(self.path)[1].strip(os.sep)
            # split the path into parts
            # get the subdirectories
            site, sensor, channel, ffile = subdir.split(os.sep) 
            fname = ffile.strip('.nc')
            c = self.channel(site, sensor, channel) 

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
        for c in self.channels:
            c.starttime = time

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
        for c in self.channels:
            c.endtime = time


    starttime = property(get_starttime, set_starttime)
    endtime = property(get_endtime, set_endtime)

