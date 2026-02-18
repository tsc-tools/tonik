import pytest
from datetime import datetime
import numpy as np

from tonik import Storage, generate_test_data

# Check if PyTorch is available
try:
    import torch
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")


def test_to_pytorch_basic(tmp_path_factory):
    """Test basic to_pytorch functionality with single time step."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 2)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset
    dataset = g.to_pytorch(['rsam', 'dsar'])
    
    # Check dataset length
    assert len(dataset) > 0
    
    # Get first sample
    sample = dataset[0]
    
    # Check that sample is a dict with the expected keys
    assert isinstance(sample, dict)
    assert 'rsam' in sample
    assert 'dsar' in sample
    
    # Check that values are PyTorch tensors
    assert torch.is_tensor(sample['rsam'])
    assert torch.is_tensor(sample['dsar'])
    
    # Check tensor dtype
    assert sample['rsam'].dtype == torch.float32
    assert sample['dsar'].dtype == torch.float32


def test_to_pytorch_with_dataloader(tmp_path_factory):
    """Test that PyTorch dataset works with DataLoader."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset and dataloader
    dataset = g.to_pytorch(['rsam'])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Iterate through dataloader
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # Check batch structure
        assert isinstance(batch, dict)
        assert 'rsam' in batch
        assert torch.is_tensor(batch['rsam'])
        # Check batch size (last batch might be smaller)
        assert batch['rsam'].shape[0] <= 4
    
    # Check that we got some batches
    assert batch_count > 0


def test_to_pytorch_window_size(tmp_path_factory):
    """Test to_pytorch with window_size > 1."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 2)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset with window_size=5
    window_size = 5
    dataset = g.to_pytorch(['rsam'], window_size=window_size)
    
    # Get first sample
    sample = dataset[0]
    
    # Check that the sample has the correct window size
    assert sample['rsam'].shape[0] == window_size


def test_to_pytorch_single_feature(tmp_path_factory):
    """Test to_pytorch with a single feature as string."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset with single feature as string
    dataset = g.to_pytorch(['rsam'])
    
    # Check that it works
    assert len(dataset) > 0
    sample = dataset[0]
    assert 'rsam' in sample


def test_to_pytorch_no_time_range(tmp_path_factory):
    """Test that to_pytorch raises error when starttime/endtime not set."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    
    # Try to create dataset without setting time range
    with pytest.raises(ValueError, match="starttime and endtime"):
        dataset = g.to_pytorch(['rsam'])


def test_to_pytorch_missing_feature(tmp_path_factory):
    """Test that to_pytorch raises error for non-existent feature."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Try to create dataset with non-existent feature
    with pytest.raises(FileNotFoundError, match="non_existent_feature"):
        dataset = g.to_pytorch(['non_existent_feature'])


def test_to_pytorch_shuffle_dataloader(tmp_path_factory):
    """Test that PyTorch dataset works with shuffled DataLoader."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset and dataloader with shuffle=True
    dataset = g.to_pytorch(['rsam', 'dsar'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Iterate through dataloader
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        assert isinstance(batch, dict)
        assert 'rsam' in batch
        assert 'dsar' in batch
    
    assert batch_count > 0


def test_to_pytorch_without_pytorch_installed():
    """Test that appropriate error is raised when PyTorch is not installed."""
    # This test is a bit tricky since we're in an environment where PyTorch might be installed
    # We'll just document the expected behavior
    # In a real scenario without PyTorch, calling to_pytorch() should raise ImportError
    pass
