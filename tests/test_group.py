import os
import tempfile

import pytest

from tonik import StorageGroup


def test_group():
    rootdir = tempfile.mkdtemp()
    g = StorageGroup('test_experiment', rootdir)
    g.site('site1')
    g.sensor('site1', 'sensor1')
    c = g.channel('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(rootdir, 'test_experiment/site1/sensor1/channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.channel('MDR1', '00', 'HHZ')
