"""CLI entry point for inference_os."""

import argparse
import sys
from typing import Sequence


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="inference-os",
        description="A reproducible LLM inference experimentation framework.",
    )
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parser.parse_args(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
