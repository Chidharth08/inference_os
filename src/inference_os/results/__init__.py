"""Results storage and persistence modules."""

from inference_os.results.persistence import (
    load_benchmark_run,
    save_benchmark_run,
)

__all__ = [
    "save_benchmark_run",
    "load_benchmark_run",
]
