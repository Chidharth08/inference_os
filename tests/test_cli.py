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
