"""Environment metadata capture for reproducible benchmarking."""

import importlib.metadata
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_PACKAGES_TO_CAPTURE: Sequence[str] = (
    "inference-os",
    "httpx",
    "transformers",
    "tokenizers",
    "torch",
    "vllm",
    "pytest",
    "ruff",
)


@dataclass(frozen=True, slots=True)
class GitMetadata:
    """Git version control state."""

    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    is_dirty: Optional[bool] = None


@dataclass(frozen=True, slots=True)
class GPUMetadata:
    """GPU hardware and driver state."""

    name: str
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    total_memory_mb: Optional[int] = None
    count: int = 1


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    """Complete hardware and software environment snapshot."""

    timestamp_utc: str
    hostname: str
    os_name: str
    os_release: str
    python_version: str
    git: GitMetadata
    gpu: Optional[GPUMetadata] = None
    packages: Optional[dict[str, str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert environment metadata to a clean, serializable dictionary."""
        return asdict(self)


def capture_git_metadata(repo_dir: Optional[Path] = None) -> GitMetadata:
    """Safely capture git commit, branch, and working tree dirty status."""
    cwd = str(repo_dir) if repo_dir is not None else None

    if not shutil.which("git"):
        return GitMetadata()

    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.returncode != 0:
            return GitMetadata()
        commit_hash = commit_res.stdout.strip()

        branch_res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        branch = branch_res.stdout.strip() or None

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(status_res.stdout.strip())

        return GitMetadata(
            commit_hash=commit_hash,
            branch=branch,
            is_dirty=is_dirty,
        )
    except Exception:
        return GitMetadata()


def parse_nvidia_smi_output(output: str) -> Optional[GPUMetadata]:
    """Parse comma-separated nvidia-smi query output."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return None

    # Example line: "NVIDIA GeForce RTX 3090, 580.173.02, 24576"
    parts = [p.strip() for p in lines[0].split(",")]
    if not parts or not parts[0]:
        return None

    name = parts[0]
    driver_version = parts[1] if len(parts) > 1 and parts[1] else None
    total_memory_mb = None
    if len(parts) > 2 and parts[2]:
        try:
            total_memory_mb = int(float(parts[2]))
        except ValueError:
            pass

    return GPUMetadata(
        name=name,
        driver_version=driver_version,
        total_memory_mb=total_memory_mb,
        count=len(lines),
    )


def capture_gpu_metadata() -> Optional[GPUMetadata]:
    """Safely capture GPU device metadata via nvidia-smi if available."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return None
        return parse_nvidia_smi_output(res.stdout)
    except Exception:
        return None


def capture_package_versions(
    packages: Sequence[str] = DEFAULT_PACKAGES_TO_CAPTURE,
) -> dict[str, str]:
    """Look up installed versions of specified packages."""
    results: dict[str, str] = {}
    for pkg in packages:
        try:
            results[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return results


def capture_environment(
    repo_dir: Optional[Path] = None,
    packages_to_capture: Sequence[str] = DEFAULT_PACKAGES_TO_CAPTURE,
) -> EnvironmentMetadata:
    """Capture a complete hardware and software environment snapshot."""
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    hostname = socket.gethostname()
    os_name = platform.system()
    os_release = platform.release()
    python_version = platform.python_version()

    git_meta = capture_git_metadata(repo_dir=repo_dir)
    gpu_meta = capture_gpu_metadata()
    package_versions = capture_package_versions(packages=packages_to_capture)

    return EnvironmentMetadata(
        timestamp_utc=timestamp_utc,
        hostname=hostname,
        os_name=os_name,
        os_release=os_release,
        python_version=python_version,
        git=git_meta,
        gpu=gpu_meta,
        packages=package_versions,
    )
