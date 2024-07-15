from datetime import datetime, timedelta
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


class LockerRoom:
    """
    Query computed features

    :param rootdir: Path to parent directory.
    :type rootdir: str
    :param starttime: Begin of request
    :type starttime: :class:`datetime.datetime`
    :param endtime: Begin of request
    :type endtime: :class:`datetime.datetime`

    >>> import datetime
    >>> fq = FeatureRequest()
    >>> start = datetime.datetime(2012,1,1,0,0,0)
    >>> end = datetime.datetime(2012,1,2,23,59,59)
    >>> group = 'Whakaari'
    >>> site = 'WIZ'
    >>> chan = 'HHZ'
    >>> fq.group = group
    >>> fq.starttime = start
    >>> fq.endtime = end
    >>> fq.site = site
    >>> fq.channel = chan
    >>> rsam = fq("rsam")
    """
    def __init__(self, group, rootdir=tempfile.gettempdir(),
                 starttime=None, endtime=None):
        self.groupdir = os.path.join(rootdir, group)
        self.lockers = {} 

    def get_locker(self, site, location, channel):
        key = (site, location, channel)
        if key not in self.lockers:
            self.lockers[key] = Locker(site, location, channel, rootdir=self.groupdir)
        return self.lockers[key]

    def __repr__(self):
        rstr = f"LockerRoom: {self.groupdir}\n"
        for site, location, channel in self.lockers.keys():
            rstr += f"Site: {site}, Location: {location}, Channel: {channel}\n" 
        return rstr

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
        for key, locker in self.lockers.items():
            locker.starttime = time

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
        for key, locker in self.lockers.items():
            locker.endtime = time

    starttime = property(get_starttime, set_starttime)
    endtime = property(get_endtime, set_endtime)


class Locker:
    def __init__(self, site=None, location=None, channel=None,
                 rootdir=None, starttime=None, endtime=None,
                 interval='10min'):

        self.site = site
        self.location = location
        self.channel = channel
        self.starttime = starttime
        self.endtime = endtime
        self.rootdir = rootdir
        self.interval = interval

    def __call__(self, feature, stack_length=None):
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
        filename = os.path.join(self.sitedir, '%s.nc' % feature)
        if not os.path.isfile(filename):
            raise ValueError('Feature {} does not exist.'.format(feature))

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

            if pd.to_timedelta(stack_length) < pd.to_timedelta(self.interval):
                raise ValueError('Stack length {} is less than interval {}'.
                                 format(stack_length, self.interval))

            # Rewind starttime to account for stack length
            self.starttime -= pd.to_timedelta(stack_length)

            num_periods = (pd.to_timedelta(stack_length)/
                           pd.to_timedelta(self.interval))
            if not num_periods.is_integer():
                raise ValueError(
                    'Stack length {} / interval {} = {}, but it needs'
                    ' to be a whole number'.
                        format(stack_length, self.interval, num_periods))

        xd_index = dict(datetime=slice(self.starttime,
                                       (self.endtime-
                                        pd.to_timedelta(self.interval))))
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
                        self.endtime-pd.to_timedelta(self.interval)]
                xdf_new = xdf_new.rename(feature)
            except ValueError as e:
                logger.error(e)
                logger.error('Stack length {} is not valid for feature {}'.
                             format(stack_length, feature))
            else:
                return xdf_new

        return rq[feature]

    def get_site(self):
        return self.__site

    def set_site(self, value):
        self.__site = value

    def get_location(self):
        return self.__location

    def set_location(self, value):
        self.__location = value

    def get_channel(self):
        return self.__channel

    def set_channel(self, value):
        self.__channel = value

    @property
    def sitedir(self):
        try:
            __sdir =  os.path.join(self.rootdir,
                                   self.site,
                                   self.location,
                                   self.channel)
            os.makedirs(__sdir, exist_ok=True)
            return __sdir
        except TypeError:
            return None

    site = property(get_site, set_site)
    location = property(get_location, set_location)
    channel = property(get_channel, set_channel)

    def load(self, *args, **kwargs):
        """
        Load a feature from disk
        """
        self.__call__(*args, **kwargs)

    def save(self, data):
        """
        Save a feature to disk
        """
        xarray2hdf5(data, self.sitedir)