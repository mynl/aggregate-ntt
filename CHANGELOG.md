# Changelog

Keep a Changelog / SemVer, newest first.

## 1.0.0a3 - 2026-10-09

**[Transform-32] the per-prime transform kernel is real, and the seam moved
to protocol 2.** `residues_mod_prime` mirrors
`aggregate.ntt._residues_mod_prime` step for step: forward radix-2 NTT of the
severity polynomial, spectral evaluation over the frequency runs (geometric
series on long runs, singular walk when the prime divides the severity
denominator), inverse NTT. One Montgomery field with 64-bit words serves odd
primes below `2**63`, so the same kernel will carry `[Transform-64]`; every
overflow bound is stated beside its line, including the `t + m * p` sum that
killed the numba attempt.

- **`PROTOCOL_VERSION` is now 2.** Landing the kernel changed its signature:
  `ds` is a reduced `u64` scalar, not an array, and all weights arrive
  already reduced modulo the prime (the caller-side reduction is the same
  one line the Python reference runs internally). No protocol 1 caller could
  ever have computed with the old signature (it always raised), but any
  signature change bumps. `aggregate` must set `_ENGINE_PROTOCOL = 2`; until
  it does, `'auto'` falls back to pure Python with one warning and `'rust'`
  refuses, both by its own design.
- Headline 1000 x 1000 case at 321 32-bit primes, 16 workers: transform
  phase alone 4.3 s against the 10 s acceptance and 104.8 s Python serial;
  end to end through the engine (transform, reconstruction, conversion,
  denominator check) 10.3 s against Python's 212.3 s.
- Unreduced weights, an unreduced `ds`, a non-primitive root, an even or
  too-wide prime, inconsistent runs, and an understated support are all loud
  refusals; the aliasing check beyond the support raises exactly as the
  Python reference does.
- Conformance: bit-for-bit per prime against `_residues_mod_prime` on three
  shapes covering every spectral branch plus the singular-prime case, and a
  full-pipeline test (engine transform into engine reconstruction) against
  `exact_compound_poly`. 30 passed, 4 xfailed (the `ntt_engine='rust'` tests
  that flip to XPASS when `aggregate` speaks protocol 2).
- One algorithmic substitution with identical values: `(b - 1)**-1` uses
  Montgomery batch inversion (three multiplies per element and one scalar
  Fermat) instead of the elementwise Fermat exponentiation, which matters
  double at 64-bit primes.
- `dev/bench-transform.py` reproduces the measurement; `--width 64` is ready
  for `[Transform-64]`.

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
