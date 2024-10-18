import logging
import tempfile
import timeit
from datetime import datetime

import pytest

from tonik import Storage, generate_test_data

logger = logging.getLogger(__name__)

tstart = datetime(2014, 1, 1)
tend = datetime(2024, 12, 31)
spec = generate_test_data(dim=2, ndays=3650, nfreqs=250, tstart=tstart)


def write_read(backend):
    test_dir = tempfile.mkdtemp()
    sg = Storage('speed_test', test_dir, starttime=tstart, endtime=tend,
                 backend=backend)
    kwargs = {}
    if backend == 'netcdf':
        kwargs['archive_starttime'] = tstart
    sg.save(spec, **kwargs)
    spec_test = sg('ssam')


@pytest.mark.slow
def test_backend_speed():
    logger.info('Testing backend speed with {} data points.'.format(
        spec['ssam'].shape[0]*spec['ssam'].shape[1]))
    execution_time_zarr = timeit.timeit(lambda: write_read('zarr'), number=5)
    logger.info('Write and read with zarr took {} seconds.'.format(
        execution_time_zarr/5))
    execution_time_h5 = timeit.timeit(lambda: write_read('netcdf'), number=5)
    logger.info('Write and read with h5 took {} seconds.'.format(
        execution_time_h5/5))


if __name__ == '__main__':
    test_backend_speed()
