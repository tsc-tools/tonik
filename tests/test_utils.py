from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from tonik.utils import (extract_consecutive_integers,
                         generate_test_data,
                         round_datetime,
                         floor_datetime)


def test_extract_consecutive_integers():
    nums = [1, 2, 3, 5, 6, 7, 8, 10]
    assert extract_consecutive_integers(
        nums) == [[1, 2, 3], [5, 6, 7, 8], [10]]
    assert extract_consecutive_integers([1]) == [[1]]
    assert extract_consecutive_integers(np.array([1, 2, 4])) == [[1, 2], [4]]


def test_generate_test_data():
    """
    Test data generation function.
    """
    tstart = datetime.now(timezone.utc) - timedelta(days=30)
    tstart = floor_datetime(tstart, timedelta(days=1))
    tstart = tstart.replace(tzinfo=None)
    data = generate_test_data(tstart='2023-01-01', freq='1min', seed=42,
                              ndays=3)
    assert 'datetime' in data.coords
    assert data.rsam.shape[0] == 3*24*60  # 24 hours + start point
    assert 'rsam' in data.data_vars
    assert 'dsar' in data.data_vars
    # Check for NaNs
    n_nans = np.isnan(data.dsar.values).sum()
    assert n_nans == 408


def test_floor_datetime_basic_10min():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    out = floor_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10, 0)


def test_floor_datetime_on_boundary():
    dt = datetime.fromisoformat("2025-11-27T10:20:00")
    out = floor_datetime(dt, 600)
    assert out == dt


def test_floor_datetime_timedelta_interval():
    dt = datetime.fromisoformat("2025-11-27T10:29:59")
    out = floor_datetime(dt, timedelta(minutes=10))
    assert out == datetime(2025, 11, 27, 10, 20, 0)


def test_floor_datetime_invalid_interval():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    with pytest.raises(ValueError):
        floor_datetime(dt, 0)
    with pytest.raises(ValueError):
        floor_datetime(dt, -15)


def test_floor_datetime_preserves_timezone_utc():
    dt = datetime(2025, 11, 27, 10, 12, 43, tzinfo=timezone.utc)
    out = floor_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10, 0, tzinfo=timezone.utc)


def test_floor_datetime_with_obspy_UTCDateTime():
    try:
        from obspy import UTCDateTime
    except Exception:
        pytest.skip("obspy not available")

    t = UTCDateTime(2025, 11, 27, 10, 12, 43)
    out = floor_datetime(t, 600)
    assert isinstance(out, UTCDateTime)
    assert out == UTCDateTime(2025, 11, 27, 10, 10, 0)


def test_round_datetime_basic_10min():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10)

    dt = datetime.fromisoformat("2025-11-27T10:10:00")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10)

    dt = datetime.fromisoformat("2025-11-27T10:17:00")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 20)
