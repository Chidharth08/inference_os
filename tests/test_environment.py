"""Unit tests for environment metadata capture."""

import json
from unittest.mock import patch

from inference_os.telemetry import (
    EnvironmentMetadata,
    GitMetadata,
    GPUMetadata,
    capture_environment,
    capture_git_metadata,
    capture_gpu_metadata,
    capture_package_versions,
    parse_nvidia_smi_output,
)


def test_capture_environment_local() -> None:
    """Verify local environment capture populates core metadata fields."""
    env = capture_environment()

    assert isinstance(env, EnvironmentMetadata)
    assert len(env.timestamp_utc) > 0
    assert len(env.hostname) > 0
    assert len(env.os_name) > 0
    assert len(env.os_release) > 0
    assert len(env.python_version) > 0

    # Git metadata should resolve in the current repository
    assert isinstance(env.git, GitMetadata)
    if env.git.commit_hash is not None:
        assert len(env.git.commit_hash) == 40

    # Key packages must be discovered
    assert env.packages is not None
    assert "inference-os" in env.packages
    assert "httpx" in env.packages
    assert "transformers" in env.packages

    # JSON serializability
    env_dict = env.to_dict()
    serialized = json.dumps(env_dict)
    assert len(serialized) > 0


def test_parse_nvidia_smi_output_single_gpu() -> None:
    """Verify parsing single GPU nvidia-smi CSV line."""
    raw = "NVIDIA GeForce RTX 3090, 580.173.02, 24576\n"
    gpu = parse_nvidia_smi_output(raw)

    assert gpu is not None
    assert isinstance(gpu, GPUMetadata)
    assert gpu.name == "NVIDIA GeForce RTX 3090"
    assert gpu.driver_version == "580.173.02"
    assert gpu.total_memory_mb == 24576
    assert gpu.count == 1


def test_parse_nvidia_smi_output_multi_gpu() -> None:
    """Verify parsing multi-GPU nvidia-smi CSV output."""
    raw = (
        "NVIDIA A100-SXM4-80GB, 535.104.05, 81920\n"
        "NVIDIA A100-SXM4-80GB, 535.104.05, 81920\n"
    )
    gpu = parse_nvidia_smi_output(raw)

    assert gpu is not None
    assert gpu.name == "NVIDIA A100-SXM4-80GB"
    assert gpu.driver_version == "535.104.05"
    assert gpu.total_memory_mb == 81920
    assert gpu.count == 2


def test_parse_nvidia_smi_output_empty() -> None:
    """Verify empty or whitespace nvidia-smi output returns None."""
    assert parse_nvidia_smi_output("") is None
    assert parse_nvidia_smi_output("   \n\n  ") is None


def test_capture_gpu_metadata_when_unavailable() -> None:
    """Verify GPU metadata returns None when nvidia-smi is not found."""
    with patch("shutil.which", return_value=None):
        assert capture_gpu_metadata() is None


def test_capture_git_metadata_fallback() -> None:
    """Verify fallback to empty GitMetadata on error without throwing."""
    with patch("shutil.which", return_value=None):
        git_meta = capture_git_metadata()
        assert git_meta.commit_hash is None
        assert git_meta.branch is None
        assert git_meta.is_dirty is None


def test_capture_package_versions() -> None:
    """Verify package version resolution and missing package handling."""
    versions = capture_package_versions(["httpx", "non_existent_pkg_xyz_12345"])
    assert "httpx" in versions
    assert "non_existent_pkg_xyz_12345" not in versions
