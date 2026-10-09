# TODO

Live list for `aggregate-ntt`. What has landed is in `CHANGELOG.md` and the git
log. The design and the measurements are in `dev/plan-rust-engine.md`; the
consuming side is `dev/plan-ntt-rust-integration.md` in the `aggregate`
repository, which is this plan's reflected half.

## In flight

Nothing in flight. `[Rust-Transform]` is next, and the `aggregate` side's
`[NTT-Rust-Integration]` plan is now unblocked by `[Rust-Reconstruction]`
landing.

## Next

- **[Rust-Transform]** — the per-prime transform. Compiled butterflies for the
  register win, `rayon` across primes for the real parallel speedup Python
  cannot have, and 64-bit primes via `u128`, which halves the prime count from
  321 to about 160. Two consequences to handle deliberately: prime selection
  must learn a width, and the two engines will then use different prime sets,
  so conformance tests must compare *answers* and never audit records. Carry
  the overflow finding: `t + m * p` in a Montgomery reduction overflows at
  these prime sizes, and the high-word formulation is the fix.

- **[Rust-Query-Object]** — stop materializing 1.25 GB of Python integers
  nobody reads. A handle holding the bigints on the Rust side, answering for
  the exact sum, exact moments, a float density vector and exact coefficients
  over a slice. Promote ahead of `[Rust-Transform]` if the limb conversion
  misses its budget. Needs coordinating with the integration plan, since it
  reshapes how `aggregate` builds `p_exact`.

## Recently landed

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
