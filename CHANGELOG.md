# Changelog

Keep a Changelog / SemVer, newest first.

## 1.0.0a2 - 2026-10-09

**[Rust-Reconstruction] the Chinese remainder kernel is real.** The
`Reconstructor` merge stack is implemented: the same balanced product tree
the Python side builds online, with every merge parallelized by `rayon` over
coefficients in coarse chunks behind a released GIL.

- Headline 1000 x 1000 shape (322 primes, 10^6 coefficients): reconstruction
  6.7 s against the 25 s acceptance and 187.0 s for Python at four workers,
  about 28x. Bulk limb-to-int conversion 0.9 s against its 3 s budget, so
  `[Rust-Query-Object]` is not promoted. Peak working set 2.69 GiB, about
  2.3x the answer, matching the Python path's profile.
- Measured divergence from the plan: the narrow merges are parallel too.
  Serial below 128 bits ran 30.2 s; all parallel with 4096-coefficient
  minimum task chunks ran 6.7 s, so the planned width gate collapsed into a
  chunk floor.
- `push` now rejects a residue at or above its prime and construction rejects
  values below 2, loud refusals instead of silently wrong reconstructions.
  `finish` requires every push and refuses a second call. No seam signature
  changed; `PROTOCOL_VERSION` stays 1.
- Conformance: random-residue reconstruction (unique below the modulus, so
  residue checks are bit-for-bit), drain shapes at 1, 2, 3 and 13 primes on
  hardcoded primes that need no `aggregate`, limb-width pinning, and
  end-to-end recovery of `exact_compound_poly` coefficients through the
  engine. 20 passed, 5 xfailed (transform placeholder and the four tests
  awaiting `ntt_engine` in `aggregate`).
- `dev/bench-reconstruction.py` reproduces the acceptance measurement,
  release builds only.

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
passing seam tests against 9 `xfail` kernel tests. The conformance suite needs an
`aggregate` checkout carrying `aggregate.ntt` installed editable by hand,
deliberately not declared as an extra because that surface is newer than
anything on PyPI; without it those tests skip rather than fail.
