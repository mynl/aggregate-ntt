# Changelog

Keep a Changelog / SemVer, newest first.

## 1.0.0a1 - 2026-10-09

**[Rust-Scaffold] the repository exists and builds.** Cargo and maturin
project on Rust 1.98, PyO3 0.29.3 with `abi3-py313` so one wheel covers 3.13
and every later 3.x, mixed Rust and Python layout with the public surface in
`python/aggregate_ntt/` over a compiled `_engine` module. Pinned alongside:
`rust-numpy` 0.29 (which must track PyO3's minor version), `rayon` 1.12,
`num-bigint` 0.5.1. The seam is declared and versioned at
`PROTOCOL_VERSION = 1`: `protocol_version()`, `engine_version()`, the
`Reconstructor` accumulator for the Chinese remainder phase, and
`residues_mod_prime` for the transform phase. Both kernels raise
`NotImplementedError` and are specified in `dev/plan-rust-engine.md`;
`[Rust-Reconstruction]` is the next milestone.

Verified rather than asserted: `maturin develop --release` builds and installs,
the module imports and reports its version and protocol, and `pytest` gives 5
passing seam tests against 9 `xfail` kernel tests. The conformance suite needs
an `aggregate` checkout carrying `aggregate.ntt` (1.1.0 or later) installed
editable; without it those tests skip rather than fail.
