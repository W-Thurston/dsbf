# ADR 0003: Polars and Arrow internally, NumPy at the statistics boundary

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §5.2

## Context

The legacy engine is pandas-based. Tasks copy and re-scan the same frame, and the measured practice
run shows the cost: 707 s and 2.1 GB peak on a 52k × 60 frame (F-30). The target tiers go beyond
memory (lazy and streaming scans, then warehouses, REFACTOR_PLAN §5.1), and pandas can't
express those without a second implementation. Polars is already a dependency, but the two backends
diverge: each pandas-dependent task converts the Polars frame itself (up to about 60 copies per run,
F-30), and the Polars random-sampling path raises a `TypeError` (F-09).

## Decision

- The internal representation is **Polars / Arrow**.
- Checks that need raw values get **NumPy arrays from a per-column cache** (`dataset.values(col)`),
  which is what scipy, statsmodels and scikit-learn take anyway. There are no per-check pandas copies.
- **pandas is used only at the edges**: accepting user DataFrames (through Arrow, zero-copy where
  possible) and optional export. It moves to the `pandas` extra.
- Sampling happens once, at the `Dataset`, and its provenance flows into every `Evidence.sampled`.

## Alternatives considered

- **Standardise on pandas.** The right call for incremental fixes to the current code, and it avoids
  porting. It is not the right call for a rebuild whose later tiers need lazy and out-of-core scans.

## Consequences

- Statistical code is ported, not wrapped. Each migration has a parity test against the legacy
  output on the golden datasets, and differences are listed in the PR (REFACTOR_PLAN §11).
- A single profiling pass can serve most rule checks, and its performance targets are set by the
  baselines in `tests/perf/`.
- Polars API churn is contained by pinning minor versions and keeping Polars behind the dataset and profile interfaces.
- The legacy pandas path stays until Phase 4 removes the legacy adapter.
