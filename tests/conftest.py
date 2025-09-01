"""Shared pytest fixtures for OPUS testing."""

import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
import torch
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration dictionary for testing."""
    return {
        "model": {
            "type": "OPUS",
            "backbone": {
                "type": "ResNet",
                "depth": 50,
                "num_stages": 4,
            },
            "neck": {
                "type": "FPN",
                "in_channels": [256, 512, 1024, 2048],
                "out_channels": 256,
            },
        },
        "dataset": {
            "type": "NuScenesOccDataset",
            "data_root": "data/nuscenes",
            "ann_file": "nuscenes_infos_train.pkl",
        },
        "train_cfg": {
            "max_epochs": 100,
            "batch_size": 1,
        },
        "test_cfg": {
            "batch_size": 1,
        },
    }


@pytest.fixture
def mock_device() -> str:
    """Mock device configuration."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample tensor for testing."""
    return torch.randn(1, 3, 256, 704)


@pytest.fixture
def sample_batch_dict() -> Dict[str, Any]:
    """Create a sample batch dictionary for testing."""
    return {
        "img": torch.randn(1, 6, 3, 256, 704),
        "img_metas": [{
            "filename": "sample.jpg",
            "ori_shape": (256, 704),
            "img_shape": (256, 704),
            "pad_shape": (256, 704),
            "scale_factor": 1.0,
            "flip": False,
        }],
        "gt_occ": torch.randint(0, 18, (1, 200, 200, 16)),
        "gt_semantics": torch.randint(0, 17, (1, 1000)),
        "gt_coords": torch.randn(1, 1000, 3),
    }


@pytest.fixture
def sample_points() -> np.ndarray:
    """Create sample 3D points for testing."""
    return np.random.randn(1000, 3).astype(np.float32)


@pytest.fixture
def sample_occupancy_grid() -> np.ndarray:
    """Create sample occupancy grid for testing."""
    return np.random.randint(0, 18, size=(200, 200, 16), dtype=np.int32)


@pytest.fixture
def mock_nuscenes_dataset():
    """Mock NuScenes dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value={
        "img": torch.randn(6, 3, 256, 704),
        "img_metas": {
            "filename": "sample.jpg",
            "ori_shape": (256, 704),
            "img_shape": (256, 704),
        },
        "gt_occ": torch.randint(0, 18, (200, 200, 16)),
    })
    return dataset


@pytest.fixture
def mock_model():
    """Mock OPUS model for testing."""
    model = Mock()
    model.forward = Mock(return_value={
        "pred_coords": torch.randn(1, 1000, 3),
        "pred_semantics": torch.randn(1, 1000, 17),
        "loss_chamfer": torch.tensor(0.5),
        "loss_semantic": torch.tensor(0.3),
    })
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 10, requires_grad=True)])
    return model


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.param_groups = [{"lr": 0.001}]
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Mock learning rate scheduler for testing."""
    scheduler = Mock()
    scheduler.step = Mock()
    scheduler.get_last_lr = Mock(return_value=[0.001])
    return scheduler


@pytest.fixture
def sample_camera_params() -> Dict[str, Any]:
    """Sample camera parameters for testing."""
    return {
        "intrinsics": np.array([
            [1266.417203, 0.0, 816.2670197],
            [0.0, 1266.417203, 491.50706579],
            [0.0, 0.0, 1.0]
        ]),
        "extrinsics": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "distortion": np.array([-0.13761, 0.14072, 0.00025, 0.00025, 0.0]),
    }


@pytest.fixture
def sample_lidar_data() -> Dict[str, np.ndarray]:
    """Sample LiDAR data for testing."""
    return {
        "points": np.random.randn(10000, 4).astype(np.float32),  # x, y, z, intensity
        "labels": np.random.randint(0, 17, size=(10000,), dtype=np.int32),
    }


@pytest.fixture
def mock_checkpoint():
    """Mock model checkpoint for testing."""
    return {
        "model": {
            "backbone.conv1.weight": torch.randn(64, 3, 7, 7),
            "backbone.bn1.weight": torch.randn(64),
            "backbone.bn1.bias": torch.randn(64),
        },
        "optimizer": {
            "state": {},
            "param_groups": [{"lr": 0.001}],
        },
        "epoch": 10,
        "best_score": 0.85,
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def no_cuda(monkeypatch):
    """Disable CUDA for CPU-only testing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def mock_mmcv_config():
    """Mock mmcv Config object."""
    config = Mock()
    config.model = Mock()
    config.model.type = "OPUS"
    config.data = Mock()
    config.data.train = Mock()
    config.optimizer = Mock()
    config.optimizer.type = "AdamW"
    config.optimizer.lr = 0.001
    return config


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def cleanup_files():
    """Cleanup test files after test execution."""
    files_to_cleanup = []
    
    def register_cleanup(filepath: Path):
        files_to_cleanup.append(filepath)
    
    yield register_cleanup
    
    # Cleanup
    for filepath in files_to_cleanup:
        if filepath.exists():
            if filepath.is_dir():
                shutil.rmtree(filepath)
            else:
                filepath.unlink()