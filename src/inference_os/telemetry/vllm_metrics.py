"""vLLM Prometheus metric capture and prefix-cache counter deltas."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

_SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
)
_LABEL_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"])*")')

_COUNTER_ALIASES: dict[str, tuple[str, ...]] = {
    "prefix_cache_queries": (
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
    ),
    "prefix_cache_hits": (
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
    ),
    "prompt_tokens": ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
    "prompt_tokens_cached": (
        "vllm:prompt_tokens_cached",
        "vllm:prompt_tokens_cached_total",
    ),
    "num_preemptions": (
        "vllm:num_preemptions",
        "vllm:num_preemptions_total",
    ),
}


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    """One parsed Prometheus exposition sample."""

    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True, slots=True)
class VLLMCacheConfig:
    """Cache settings reported by vLLM's cache-config info metric."""

    enable_prefix_caching: bool
    block_size: int
    cache_dtype: str | None = None
    gpu_memory_utilization: float | None = None
    labels: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class VLLMMetricsSnapshot:
    """Raw and parsed vLLM metrics captured at one instant."""

    captured_at: str
    raw_text: str
    samples: tuple[PrometheusSample, ...]

    def counter(self, logical_name: str) -> float | None:
        """Return an aggregated counter by stable logical name."""
        aliases = _COUNTER_ALIASES[logical_name]
        for alias in aliases:
            values = [sample.value for sample in self.samples if sample.name == alias]
            if values:
                return sum(values)
        return None

    def gauge(self, *names: str) -> float | None:
        """Return the maximum reported gauge value for any supplied name."""
        values = [sample.value for sample in self.samples if sample.name in names]
        return max(values) if values else None

    def cache_config(self) -> VLLMCacheConfig:
        """Extract one internally consistent active cache configuration."""
        candidates = [
            sample
            for sample in self.samples
            if sample.name in {"vllm:cache_config_info", "vllm:cache_config"}
            and sample.value == 1.0
        ]
        if not candidates:
            raise ValueError("vLLM metrics do not expose an active cache_config_info")

        parsed = [_cache_config_from_labels(sample.labels) for sample in candidates]
        first = parsed[0]
        if any(
            item.enable_prefix_caching != first.enable_prefix_caching
            or item.block_size != first.block_size
            or item.cache_dtype != first.cache_dtype
            or item.gpu_memory_utilization != first.gpu_memory_utilization
            for item in parsed[1:]
        ):
            raise ValueError("vLLM metrics expose conflicting active cache configs")
        return first

    def summary(self) -> dict[str, Any]:
        """Return stable cache-relevant values for JSON persistence."""
        config = self.cache_config()
        return {
            "captured_at": self.captured_at,
            "cache_config": asdict(config),
            "counters": {key: self.counter(key) for key in _COUNTER_ALIASES},
            "kv_cache_usage": self.gauge(
                "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
            ),
        }


@dataclass(frozen=True, slots=True)
class PrefixCacheMetricsDelta:
    """Cache observations restricted to one measured benchmark window."""

    prefix_cache_query_tokens: float | None
    prefix_cache_hit_tokens: float | None
    observed_prefix_cache_hit_fraction: float | None
    prompt_tokens: float | None
    cached_prompt_tokens: float | None
    preemptions: float | None
    kv_cache_usage_before: float | None
    kv_cache_usage_after: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_prometheus_text(text: str) -> tuple[PrometheusSample, ...]:
    """Parse the subset of Prometheus text exposition needed by E007."""
    samples: list[PrometheusSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_PATTERN.match(line)
        if match is None:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        labels_text = match.group("labels") or ""
        labels = {
            label_match.group(1): json.loads(label_match.group(2))
            for label_match in _LABEL_PATTERN.finditer(labels_text)
        }
        samples.append(
            PrometheusSample(
                name=match.group("name"),
                labels=labels,
                value=value,
            )
        )
    return tuple(samples)


async def fetch_vllm_metrics(
    base_url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_seconds: float = 30.0,
) -> VLLMMetricsSnapshot:
    """Fetch and parse one snapshot from the vLLM ``/metrics`` endpoint."""
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    try:
        response = await client.get(
            f"{base_url.rstrip('/')}/metrics", timeout=timeout_seconds
        )
        response.raise_for_status()
        raw_text = response.text
    finally:
        if close_client:
            await client.aclose()
    return VLLMMetricsSnapshot(
        captured_at=datetime.now(timezone.utc).isoformat(),
        raw_text=raw_text,
        samples=parse_prometheus_text(raw_text),
    )


def compute_prefix_cache_delta(
    before: VLLMMetricsSnapshot,
    after: VLLMMetricsSnapshot,
) -> PrefixCacheMetricsDelta:
    """Compute non-negative counter deltas for one measured request window."""
    deltas = {
        name: _counter_delta(before.counter(name), after.counter(name), name)
        for name in _COUNTER_ALIASES
    }
    queries = deltas["prefix_cache_queries"]
    hits = deltas["prefix_cache_hits"]
    hit_fraction = None
    if queries is not None and queries > 0 and hits is not None:
        hit_fraction = hits / queries
    return PrefixCacheMetricsDelta(
        prefix_cache_query_tokens=queries,
        prefix_cache_hit_tokens=hits,
        observed_prefix_cache_hit_fraction=hit_fraction,
        prompt_tokens=deltas["prompt_tokens"],
        cached_prompt_tokens=deltas["prompt_tokens_cached"],
        preemptions=deltas["num_preemptions"],
        kv_cache_usage_before=before.gauge(
            "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
        ),
        kv_cache_usage_after=after.gauge(
            "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
        ),
    )


def validate_cache_server_config(
    snapshot: VLLMMetricsSnapshot,
    *,
    expected_enabled: bool,
    expected_block_size: int,
) -> VLLMCacheConfig:
    """Fail fast when the live server disagrees with the E007 condition."""
    config = snapshot.cache_config()
    if config.enable_prefix_caching != expected_enabled:
        raise ValueError(
            "live vLLM prefix-caching state does not match the benchmark config: "
            f"expected {expected_enabled}, observed {config.enable_prefix_caching}"
        )
    if config.block_size != expected_block_size:
        raise ValueError(
            "live vLLM cache block size does not match prompt_reuse metadata: "
            f"expected {expected_block_size}, observed {config.block_size}"
        )
    return config


def validate_pristine_cache_state(snapshot: VLLMMetricsSnapshot) -> None:
    """Require zero prefix-cache activity before an isolated cache-ON run."""
    queries = snapshot.counter("prefix_cache_queries")
    hits = snapshot.counter("prefix_cache_hits")
    if queries not in {None, 0.0} or hits not in {None, 0.0}:
        raise ValueError(
            "E007 cache-ON condition requires a fresh vLLM server with zero "
            f"prefix-cache counters; observed queries={queries}, hits={hits}"
        )


def persist_cache_metric_artifacts(
    run_dir: Path | str,
    *,
    preflight: VLLMMetricsSnapshot,
    before: VLLMMetricsSnapshot,
    after: VLLMMetricsSnapshot,
    delta: PrefixCacheMetricsDelta,
) -> None:
    """Persist auditable raw snapshots and add cache data to ``summary.json``."""
    path = Path(run_dir)
    (path / "cache_metrics_preflight.prom").write_text(
        preflight.raw_text, encoding="utf-8"
    )
    (path / "cache_metrics_before.prom").write_text(before.raw_text, encoding="utf-8")
    (path / "cache_metrics_after.prom").write_text(after.raw_text, encoding="utf-8")
    server_config = asdict(preflight.cache_config())
    (path / "server_cache_config.json").write_text(
        json.dumps(server_config, indent=2), encoding="utf-8"
    )
    delta_data = delta.to_dict()
    (path / "cache_metrics_delta.json").write_text(
        json.dumps(delta_data, indent=2), encoding="utf-8"
    )
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["server_cache_config"] = server_config
    summary["cache"] = delta_data
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _counter_delta(
    before: float | None,
    after: float | None,
    name: str,
) -> float | None:
    if before is None and after is None:
        return None
    if before is None or after is None:
        raise ValueError(f"metric {name} was present in only one snapshot")
    delta = after - before
    if delta < 0:
        raise ValueError(f"metric {name} reset during the measured window")
    return delta


def _cache_config_from_labels(labels: dict[str, str]) -> VLLMCacheConfig:
    try:
        enabled_text = labels["enable_prefix_caching"].strip().lower()
        if enabled_text not in {"true", "false"}:
            raise ValueError("enable_prefix_caching is not boolean")
        enabled = enabled_text == "true"
        block_size = int(labels["block_size"])
    except (KeyError, ValueError) as exc:
        raise ValueError("cache_config_info is missing valid required labels") from exc
    cache_dtype = labels.get("cache_dtype")
    raw_utilization = labels.get("gpu_memory_utilization")
    utilization = float(raw_utilization) if raw_utilization is not None else None
    return VLLMCacheConfig(
        enable_prefix_caching=enabled,
        block_size=block_size,
        cache_dtype=cache_dtype,
        gpu_memory_utilization=utilization,
        labels=dict(labels),
    )
