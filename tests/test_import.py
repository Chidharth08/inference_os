"""Verify that the inference_os package imports correctly."""

import inference_os


def test_package_import() -> None:
    """Test package version string presence."""
    assert hasattr(inference_os, "__version__")
    assert isinstance(inference_os.__version__, str)
