"""Verify CLI behavior."""

import pytest

from inference_os.cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that running CLI with --help exits with code 0 and prints usage."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "inference-os" in captured.out
    assert "usage:" in captured.out
    assert "sweep" in captured.out
    assert "run" in captured.out


def test_cli_sweep_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that running CLI sweep with --help prints sweep options."""
    with pytest.raises(SystemExit) as exc_info:
        main(["sweep", "--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "--config" in captured.out

