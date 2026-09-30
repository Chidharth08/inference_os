# Benchmark Methodology

This document will define the measurement standards and experimental policies for `inference_os`.

## Target Scope

Future versions of this document will detail:

* **Benchmark Methodology**: Standardized procedures for executing inference benchmarking runs.
* **Warm-up Policy**: Rules for warming up model caches and GPU execution states prior to measurement collection.
* **Repetition Policy**: Guidelines for run iterations, statistical sample sizes, and variance reduction.
* **Controlled Variables**: Protocols for holding environmental and execution factors constant during sweeps.
* **Hardware Isolation & Validity**: Rules requiring hardware profiles (e.g., RTX 3090 vs L4) to be kept fixed per benchmark series to ensure valid comparison.
* **Timing Methodology**: High-resolution clock measurement standards for request-level milestones.
* **Experimental Validity**: Requirements for identifying confounders, measurement overhead, and valid scope of inferences.

*Specific methodology decisions will be formalized during experiment E000 (Measurement Validation).*

## E007 Prefix-Cache Isolation Policy

E007 treats cache state as an experimental condition.

- Cache-OFF and cache-ON conditions run in separate vLLM server lifetimes.
- The live server's cache configuration and block size are verified through its
  metrics endpoint before requests are sent.
- Cache-ON preflight query and hit counters must be zero, establishing a fresh
  server cache state.
- Model/GPU warm-up prompts differ from every measured prompt within the first
  cacheable block.
- Cache counters are sampled after warm-up and after measurement, so reported
  deltas exclude warm-up traffic.
- The first measured shared-prefix request is a compulsory miss; subsequent
  requests are expected hits and are summarized separately.
- A server restart is required between independent conditions.
- Canonical comparisons use at least three independent repetitions when GPU
  budget permits, with condition order alternated across repetitions.

A cache-performance conclusion requires both latency evidence and observed
cache-hit evidence. The configured cache flag or intended shared prefix alone is
insufficient.
