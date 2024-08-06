import os
import tempfile

import pytest

from tonik import StorageGroup


def test_group():
    rootdir = tempfile.mkdtemp()
    g = StorageGroup('test_experiment', rootdir)
    c = g.get_store('site1', 'sensor1', 'channel1')
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
    c = g.get_store('MDR1', '00', 'HHZ')

# class FeatureRequestTestCase(unittest.TestCase):

#     def setUp(self):
#         # create test directory
#         self.tempdir = tempfile.mkdtemp()
#         self.rootdir = os.path.join(self.tempdir, 'Mt_Doom', 'MDR', 'HHZ')
#         try:
#             os.makedirs(self.rootdir)
#         except FileExistsError:
#             pass

#     def tearDown(self):
#         if os.path.isdir(self.tempdir):
#             shutil.rmtree(self.tempdir)
            
#     def test_false_feature(self):
#         fq = FeatureRequest(rootdir=self.tempdir)
#         with self.assertRaises(ValueError):
#             fakefeature = fq('fakefeature')

#     def test_call_multiple_days(self):
#         startdate = UTCDateTime(2016, 1, 1)._get_datetime()
#         enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
#         xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
#         xarray2hdf5(xdf, self.rootdir)
#         fq = FeatureRequest(rootdir=self.tempdir)
#         fq.starttime = startdate
#         fq.endtime = enddate
#         fq.volcano = 'Mt_Doom'
#         fq.site = 'MDR'
#         fq.channel = 'HHZ'
#         rsam = fq('rsam')
#         # Check data
#         xd_index = dict(datetime=slice(startdate,
#                                        (enddate-pd.to_timedelta(fq.interval))))

#         np.testing.assert_array_almost_equal(
#             rsam.loc[startdate:enddate], xdf.rsam.loc[xd_index], 5)
#         # Check datetime range is correct
#         first_time = pd.to_datetime(rsam.datetime.values[0])
#         last_time = pd.to_datetime(rsam.datetime.values[-1])
#         self.assertEqual(pd.to_datetime(startdate), first_time)
#         self.assertEqual(pd.to_datetime(enddate) - pd.to_timedelta(fq.interval),
#                          last_time)

#     def test_call_single_day(self):
#         fq = FeatureRequest(rootdir=self.rootdir)
#         startdate = datetime.datetime(2016, 1, 2, 1)
#         enddate = datetime.datetime(2016, 1, 2, 12)
#         xdf = generate_test_data(dim=1, tstart=startdate)
#         xarray2hdf5(xdf, self.rootdir)
#         fq = FeatureRequest(rootdir=self.tempdir)
#         fq.starttime = startdate
#         fq.endtime = enddate
#         fq.volcano = 'Mt_Doom'
#         fq.site = 'MDR'
#         fq.channel = 'HHZ'
#         rsam = fq('rsam')
#         # Check datetime range is correct
#         first_time = pd.to_datetime(rsam.datetime.values[0])
#         last_time = pd.to_datetime(rsam.datetime.values[-1])
#         self.assertEqual(pd.to_datetime(startdate), first_time)
#         self.assertEqual(pd.to_datetime(enddate) - pd.to_timedelta(fq.interval),
#                          last_time)

#     def test_rolling_window(self):
#         startdate = UTCDateTime(2016, 1, 1)._get_datetime()
#         enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
#         xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
#         xarray2hdf5(xdf, self.rootdir)
#         fq = FeatureRequest(rootdir=self.tempdir)

#         stack_len_seconds = 3600
#         stack_len_string = '1H'

#         num_windows = int(stack_len_seconds / pd.Timedelta(fq.interval).seconds)
#         fq.starttime = startdate
#         fq.endtime = enddate
#         fq.volcano = 'Mt_Doom'
#         fq.site = 'MDR'
#         fq.channel = 'HHZ'
#         rsam = fq('rsam')
#         rsam_rolling = fq('rsam', stack_length=stack_len_string)

#         # Check correct datetime array
#         np.testing.assert_array_equal(rsam.datetime.values,
#                                       rsam_rolling.datetime.values)
#         # Check correct values
#         rolling_mean = [
#             np.nanmean(rsam.data[(ind-num_windows+1):ind+1])
#             for ind in np.arange(num_windows, len(rsam_rolling.data))
#         ]
#         np.testing.assert_array_almost_equal(
#             np.array(rolling_mean), rsam_rolling.values[num_windows:], 6
#         )

#     def test_cache(self):
#         """
#         Test that data loaded from files and from cache are identical.
#         """
#         startdate = UTCDateTime(2016, 1, 1)._get_datetime()
#         enddate = UTCDateTime(2016, 1, 2, 12)._get_datetime()
#         xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
#         xarray2hdf5(xdf, self.rootdir)
#         fq = FeatureRequest(rootdir=self.tempdir)
#         fq.starttime = startdate 
#         fq.endtime = enddate 
#         fq.volcano = 'Mt_Doom'
#         fq.site = 'MDR'
#         fq.channel = 'HHZ'
#         data_from_files = fq('RSAM')
#         data_from_cache = fq('RSAM')
#         xr.testing.assert_equal(data_from_files, data_from_cache)
