# TODO

Live list for `aggregate-ntt`. What has landed is in `CHANGELOG.md` and the git
log. The design and the measurements are in `dev/plan-rust-engine.md`; the
consuming side is `dev/plan-ntt-rust-integration.md` in the `aggregate`
repository, which is this plan's reflected half.

## In flight

- **[Transform-64]** (`dev/plan-rust-engine.md`, the second half of
  `[Rust-Transform]`) — the kernel already accepts odd primes below `2**63`,
  so what remains is widening `Reconstructor` from `u32` to `u64` primes (a
  compatible widening, no protocol change), 64-bit conformance tests with
  locally generated primes, and the headline measurement at about 160 wide
  primes. The author ruled 2026-10-09 that `aggregate`'s `ntt_primes` grows
  a width parameter and reports the primes over; the engine's ceiling it
  must respect is `p < 2**63` (Montgomery headroom).

## Next

- **[Rust-Query-Object]** — stop materializing 1.25 GB of Python integers
  nobody reads. A handle holding the bigints on the Rust side, answering for
  the exact sum, exact moments, a float density vector and exact coefficients
  over a slice. Promote ahead of `[Rust-Transform]` if the limb conversion
  misses its budget. Needs coordinating with the integration plan, since it
  reshapes how `aggregate` builds `p_exact`.

## Recently landed

- **[Transform-32]** DONE at **1.0.0a3** (2026-10-09): the per-prime
  transform kernel, mirroring `_residues_mod_prime` step for step in one
  Montgomery field (64-bit words, odd primes below `2**63`) that will carry
  `[Transform-64]` unchanged. Headline transform phase 4.3 s against the
  10 s acceptance (Python: 104.8 s serial); end to end through the engine
  10.3 s against 212.3 s. **The seam moved to protocol 2** (`ds` became a
  reduced scalar as the kernel landed); `aggregate` owes
  `_ENGINE_PROTOCOL = 2`. Execution log in `dev/plan-rust-engine.md`,
  benchmark in `dev/bench-transform.py`.

- **[Rust-Reconstruction]** DONE at **1.0.0a2** (2026-10-09): the Chinese
  remainder kernel, a balanced product tree built online as a merge stack
  with every merge `rayon`-parallel over coefficients behind a released GIL.
  Headline shape reconstruction 6.7 s against the 25 s acceptance (Python:
  187 s at four workers); limb conversion 0.9 s against 3 s, so
  `[Rust-Query-Object]` stays behind `[Rust-Transform]`; peak memory about
  2.3x the answer. One measured divergence: the narrow merges parallelize
  too, with coarse task chunks instead of the planned width gate, 30.2 s
  against 6.7 s. Details in the execution log of `dev/plan-rust-engine.md`
  and the benchmark in `dev/bench-reconstruction.py`.

- **[Rust-Scaffold]** DONE at **1.0.0a1** (2026-10-09): repository, Cargo and
  maturin project, PyO3 0.29.3 with `abi3-py313`, mixed Rust and Python
  layout, the versioned seam (`protocol_version`, `engine_version`,
  `Reconstructor`, `residues_mod_prime`) with both kernels raising
  `NotImplementedError`, and a conformance suite whose seam tests run now and
  whose kernel tests are `xfail(strict=True)` so they become failures the
  moment a kernel works and is wrong. Built and ran: 5 passed, 9 xfailed.
  `[Rust-Dependency-Pins]` was folded in here, since the first pins attempted
  (PyO3 0.23) had no `abi3-py313` feature at all and had to be resolved to get
  a build.
