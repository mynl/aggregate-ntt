# [Rust-Engine] a compiled engine for exact compound distributions

Status: `[Rust-Scaffold]` **DONE** at 1.0.0a1 on 2026-10-09. Everything below
it is specified and not implemented. `[Rust-Reconstruction]` is next.

This plan is written for a reader with no prior context, including no memory of
the `aggregate` session that produced it. Its reflected half is
`dev/plan-ntt-rust-integration.md` in the `aggregate` repository, which states
the consuming side and must be read alongside it.

## What this is, and why it exists

`aggregate` is a Python actuarial library. Its `nttagg` kind computes a
bounded discrete compound (aggregate) distribution **exactly**, as rational
numbers, by running the FFT convolution recipe in prime finite fields (the
number-theoretic transform) and reconstructing the integer coefficients with
the Chinese remainder theorem. The answer is not an approximation: the
coefficients are exact integers over an exact denominator.

That engine is pure Python and numpy, and it is fast enough. The author's
position, stated on 2026-10-09: *for any rational problem it is fine.* This
repository exists because the numbers can be an order of magnitude better,
not because they need to be. It is a side project and should be treated as
one: no deadline, no obligation on `aggregate`, and a hard requirement that
`aggregate` works perfectly without it.

### The measured baseline, on DOOB at `aggregate` 1.2.0a5

Two reference cases, used throughout this plan. "Uniform dice of dice n x n"
means claim count uniform on 1..n and severity uniform on 1..n, which gives
support `n**2` and transform length `2**ceil(log2(n**2 + 1))`.

| case | primes | length | coefficients | bound bits |
|---|---|---|---|---|
| 500 x 500 | 141 | 2**18 | 250,001 | 4,492 |
| 1000 x 1000 | 321 | 2**20 | 1,000,001 | 9,976 |

| case | workers | transform | reconstruction | wall |
|---|---|---|---|---|
| 500 x 500 | 1 | 10.4 s | 14.0 s | 24.5 s |
| 500 x 500 | 8 | 119.3 thread-s | 14.5 s | 16.6 s |
| 1000 x 1000 | 1 | 104.8 s | 207.3 s | 312.9 s |
| 1000 x 1000 | 4 | 709.3 thread-s | 187.0 s | **216.4 s** |
| 1000 x 1000 | 8 | 1355.7 thread-s | 197.7 s | 212.3 s |

Three readings, which between them dictate every milestone below.

1. **The reconstruction is the larger half and has no vectorized form.** It is
   86 percent of the threaded wall clock at the current default of four
   workers. On the Python side it is a per-coefficient loop over Python
   integers, so unlike the transform it has never been vectorized at all.
2. **The transform's thread-seconds inflate 7x to 13x under threading.** 104.8
   serial seconds become 709.3 thread-seconds at four workers. That is the
   GIL: the uint64 ufuncs release it, but a worker must reacquire it between
   ufunc calls, and the main thread holds it in long stretches doing the
   reconstruction's bigint arithmetic. Threads here buy *overlap*, not
   parallel speedup. A compiled extension that releases the GIL once does not
   have this problem, which is the single largest structural win available.
3. **The prime count is capped by a word size, not by mathematics.** Python is
   pinned to primes below `2**32` because a product of two residues must fit
   in `uint64` for the vectorized multiply. Rust has `u128`, so 64-bit primes
   cost nothing structurally, and that halves the prime count for the same
   coefficient bound: 321 becomes about 160.

### Target

The headline 1000 x 1000 case in **low tens of seconds** rather than 212.
Beyond some point the floor is memory bandwidth on 1.25 GB of output rather
than arithmetic, which is what `[Rust-Query-Object]` addresses.

For scale, the alternative considered and shelved: using `gmpy2` for the
reconstruction's bigints inside Python was measured at 2.0x, taking the
headline case to about 115 s. It is specified as stage 3 of
`dev/plan-ntt-crt.md` in `aggregate` and was **shelved in favor of this
repository**. It remains the fallback if this stalls.

## Division of labor, which is also the safety argument

**`aggregate` keeps everything except the two hot kernels.** Specifically it
keeps input validation, clearing denominators to integers, prime selection,
the size guardrails, the audit record, the accuracy and statistics frames,
and all three end-to-end correctness checks.

Those checks are the reason this is safe. They run on *this engine's* answer
exactly as they run on the pure-Python one:

- the reconstructed coefficients sum to exactly the denominator `D`,
- no coefficient is negative,
- no coefficient beyond the support is nonzero,

and the CRT modulus exceeds `D`. Together they are decisive rather than
advisory: a transform that aliased, a prime that was mishandled, a merge that
lost a carry, all break them. On top of that, `aggregate` carries
`exact_compound_poly`, a direct polynomial computation that uses neither a
transform nor the CRT, so it is a genuinely independent oracle for small
cases.

**This engine is therefore verified, not trusted.** That is what makes it
acceptable to let an experimental compiled accelerator compute real answers.

**The runtime dependency goes one way.** `aggregate` may import this package;
this package must never import `aggregate` at runtime. Only the `dev` extra
depends on it, for the conformance suite.

## The seam

Declared in `src/lib.rs`, surfaced in `python/aggregate_ntt/__init__.py`, and
versioned by `PROTOCOL_VERSION`, currently 1.

```
protocol_version() -> int           seam contract version
engine_version()   -> str           this crate's version, for the audit
Reconstructor(primes, n_coefficients)
    .push(residues: uint64[n_coefficients])     one prime, in order
    .finish() -> (limbs: uint64[n_coefficients, n_limbs], modulus: bytes)
residues_mod_prime(prime, root, length, support, sev_index, sev_weights,
                   freq_runs, ds, kmax) -> uint64[support + 1]
```

Three deliberate choices in that signature.

**The reconstruction is a stateful accumulator, not a one-shot call.** The
one-shot form would take every prime's residue vector at once, which for the
headline case is 1.3 GB crossing the boundary on top of the 1.25 GB the
answer occupies, and the Python caller currently consumes residue vectors one
at a time as its thread pool produces them. The accumulator preserves that
online property and keeps the merge stack on the Rust side.

**`finish` returns fixed-width limbs, not Python integers.** Building a
million Python integers one at a time across PyO3 would eat the winnings.
Every coefficient is below the final modulus, so the width is uniform, and
one contiguous `(n_coefficients, n_limbs)` array is a memory copy. The caller
converts in bulk, or defers (see `[Rust-Query-Object]`).

**`residues_mod_prime` mirrors the Python private function of the same name**
argument for argument, so the two can be compared directly during
development. This is the one place reaching for a private name on the Python
side is legitimate, because it is the comparison target.

### Changing the seam

Bump `PROTOCOL_VERSION` on **any** incompatible change, record it in
`CHANGELOG.md`, and update the reflected plan in `aggregate`. `aggregate`
refuses a protocol version it does not recognize rather than computing against
a surface that moved. A loud refusal is acceptable; a silently wrong answer is
not.

## [Rust-Scaffold], DONE at 1.0.0a1

Cargo and maturin project. PyO3 with `abi3-py313` so one wheel covers 3.13 and
every later 3.x. Mixed layout: Rust in `src/`, the public Python surface in
`python/aggregate_ntt/`. Seam declared and versioned, both kernels raising
`NotImplementedError`, conformance suite scaffolded with the seam tests live
and the kernel tests marked `xfail(strict=True)` so they convert to failures
the moment a kernel starts working and is wrong.

**Verified, not merely written.** `cargo check` is clean, `maturin develop
--release` produces an `abi3` wheel for 3.13 and later, the module imports,
and the suite runs: 5 seam tests pass and 9 kernel tests `xfail`. The first
pins attempted (PyO3 0.23) were stale and had no `abi3-py313` feature at all;
the resolved set is **PyO3 0.29.3, `rust-numpy` 0.29, `rayon` 1.12,
`num-bigint` 0.5.1** on Rust 1.98. PyO3 and `rust-numpy` must stay on matching
minor versions, which is why they move together.

## [Rust-Reconstruction], the first kernel

### The algorithm

The same balanced product tree the Python side uses, so the two can be
compared level by level. One merge combines two partial reconstructions over
coprime moduli, elementwise across coefficients. With `a` exact mod `Ma` and
`b` exact mod `Mb`, the value exact mod `Ma * Mb` is

```
a + Ma * ((b - a) * Ma**(-1) mod Mb)
```

which is in range because the bracket is reduced mod `Mb`. The scalar inverse
depends only on the moduli, so it is computed once per merge and shared across
every coefficient.

The tree is built online as a **merge stack**: each pushed prime becomes a
one-prime partial, and any two partials of equal weight on top of the stack
are merged at once, weight being the number of primes behind a partial. The
stack holds at most `log2(n)` partials, so peak memory is roughly twice the
answer rather than the answer plus every prime's residue vector. The
structure is a binary counter: the merges on push `i` are the trailing ones of
`i - 1`, so the expensive full-width merges fall at pushes 64, 128, 256, and
the final drain combines what the binary decomposition of the prime count
leaves behind (321 = 256 + 64 + 1, so the drain does the two largest merges of
the entire run).

### Parallelism: inside the merge, over coefficients

A merge is an elementwise map over coefficients with no cross-coefficient
dependency, so `rayon` parallelizes it directly, and the GIL is released for
the duration with `Python::allow_threads`. This is the right axis for three
reasons: it preserves the online property that keeps memory bounded, it needs
no restructuring, and the work concentrates exactly where parallelism pays,
since the few full-width merges dominate the phase.

Do **not** parallelize by prime subtree, which needs partials combined across
threads for no gain, and do not parallelize the narrow merges: at one or two
primes the arithmetic is smaller than `rayon`'s per-task overhead. Gate on
operand width with a threshold, measured, in the spirit of the Python side's
`_CRT_MPZ_CUTOVER`.

**Measurement to carry across from the `aggregate` session**, because it
predicts where the work is. A level profile of the headline case, 1,000
coefficients, CPython integers against `gmpy2`:

| level | merges | CPython int | GMP | ratio |
|---|---|---|---|---|
| final drain, 256 primes | 1 | 0.035 s | 0.005 s | 7.0x |
| 128 | 1 | 0.032 s | 0.004 s | 8.0x |
| 64 | 2 | 0.020 s | 0.003 s | 6.7x |
| 32 | 5 | 0.014 s | 0.003 s | 4.7x |
| 16 | 10 | 0.009 s | 0.003 s | 3.0x |
| 8 | 20 | 0.008 s | 0.005 s | 1.6x |
| 4 | 40 | 0.009 s | 0.008 s | 1.1x |
| 2 | 80 | 0.014 s | 0.016 s | 0.9x |
| 1 | 160 | 0.023 s | 0.025 s | 0.9x |
| conversion from residues | 321 | 0.012 s | 0.027 s | 0.4x |

Two lessons. The cost is **spread**, not concentrated: the top two merges are
38 percent and the bottom four levels plus conversion are 37 percent. And a
better bigint library helps only the top, which is why the all-`gmpy2`
experiment returned 1.37x where its own big merges were 7x faster. A compiled
implementation attacks both halves, since the bottom levels are
interpreter-and-allocation overhead that simply ceases to exist in Rust.

### [Rust-Bigint-Choice]

Start with `num-bigint`: MIT/Apache, Karatsuba and Toom-3 multiplication,
**schoolbook division**. Division is the step the merge spends its time in, so
if a profile says it dominates, move to `rug`, which binds GMP and has
subquadratic division. The licensing is containable precisely because this is
a separate distribution: GMP is LGPL, and the obligation would live here and
never reach `aggregate`, which is BSD. Never vendor GMP into `aggregate` and
never make it a dependency there.

A third option worth a measurement before settling: the moduli here are known
in advance and reused across every coefficient, so a **Barrett or Montgomery
reciprocal computed once per merge** turns each coefficient's division into
two multiplications. This was tried in Python and was slightly *slower*,
because CPython's division of near-equal-size operands already has a
single-digit quotient and is effectively linear. That argument does not
transfer automatically to a different bigint library, so re-measure rather
than assuming either way.

### Acceptance

- Bit-for-bit equality with the Python reconstruction on the conformance
  corpus, and with `exact_compound_poly` end to end on small cases.
- `aggregate`'s three end-to-end checks pass on the engine's answer.
- Reconstruction phase of the headline case **at or below 25 s**, against
  187.0 s at four workers and 207.3 s serial. That is 8x and should be
  conservative given no GIL and no per-coefficient interpreter overhead.
- The boundary conversion measured and reported separately, so it cannot hide
  inside the phase total. Budget: 10^6 coefficients of 161 limbs converting in
  **under 3 s**. If it exceeds that, `[Rust-Query-Object]` is promoted ahead
  of `[Rust-Transform]`.
- Peak resident memory no worse than the Python path's, roughly twice the
  answer.

## [Rust-Transform], the second kernel

Forward transform of the severity polynomial, spectral evaluation of the
compound polynomial over the frequency outcome runs, inverse transform. The
Python implementation in `aggregate.ntt` is the specification; read
`_residues_mod_prime`, `_spectral_walk`, `_spectral_walk_singular` and
`_outcome_runs` before starting, including the closed-form geometric sum for
equal-weight runs, which is worth an order of magnitude on a uniform claim
count and must be carried over.

Three things this milestone gets that Python cannot have.

**No GIL**, so `rayon` across primes gives real parallel speedup instead of
the 7x to 13x thread-second inflation measured above.

**Registers instead of memory passes.** Each numpy ufunc call streams the
whole array through memory; a compiled butterfly keeps its operands in
registers. Expect 5x to 20x on the same arithmetic. At `2**20` the array no
longer fits in cache, so use a four-step or six-step decomposition to keep the
inner transforms cache resident.

**64-bit primes**, via `u128` intermediates, halving the prime count from 321
to about 160. This halves the transform's work outright and also halves the
reconstruction's bottom levels, which are pure per-coefficient overhead.

Two consequences of 64-bit primes that must be handled, not discovered:

1. **Prime selection must learn a width.** `aggregate.ntt.ntt_primes` scans
   below `2**32`. The integration plan covers extending it, or having the
   engine supply its own primes and report them. Either way the pure-Python
   path keeps 32-bit primes, because numpy has no 128-bit integer.
2. **The two engines will then use different prime sets, and that is fine.**
   The reconstructed integers are determined by the mathematics, not by which
   primes were used, so conformance still holds bit for bit. What differs is
   the *audit*: `primes_used`, `prime_bits_*`. Conformance tests must compare
   answers, never audit records.

### The overflow trap, recorded so it is not rediscovered

An earlier attempt to compile this phase with numba was abandoned, and its
review left a finding that applies verbatim here: a Montgomery reduction
written as `t + m * p` **overflows** at these prime sizes, because every prime
sits just under its word boundary and the sum reaches almost `2**65`. The
high-word formulation is the fix. More generally, and this is a standing rule
in `CLAUDE.md`: every modular arithmetic kernel in this crate states its
overflow bound in a comment.

### Acceptance

- Bit-for-bit equality with `_residues_mod_prime` per prime, at 32-bit primes,
  which is the directly comparable configuration.
- End to end equality with the pure-Python path and with
  `exact_compound_poly`, at both prime widths.
- Transform phase of the headline case **at or below 10 s** wall clock using
  all cores, against 104.8 s serial.
- Headline case end to end in **low tens of seconds**.

## [Rust-Query-Object], do not materialize what nobody reads

The headline answer is a million integers of about 10,000 bits, 1.25 GB. Most
consumers never want them: they want exact moments, a float density vector,
and exact probabilities at a handful of tail points. Materializing 1.25 GB of
Python integers to answer those is the dominant remaining cost once the
kernels are compiled.

So `finish` gains an alternative that returns a handle holding the bigints on
the Rust side, with methods for what callers actually ask: the exact sum (the
denominator check), exact moments, a correctly rounded float probability
vector, and exact coefficients over a requested slice. `aggregate` then builds
`p_exact` lazily instead of eagerly.

This reshaping is good for the pure-Python path too, so coordinate it with the
integration plan rather than treating it as an engine-only concern.

## [Rust-Build-Local], packaging

Two build profiles, deliberately different.

**Local, which is the point of the exercise.** `maturin develop --release`
with `RUSTFLAGS=-C target-cpu=native`, which uses the author's actual
instruction set. A published wheel can never do this, and it is most of why
compiling is worth it here.

**Published, if it is ever published.** Baseline instruction set, no
`target-cpu`, `abi3-py313` so one wheel per platform covers every later 3.x.
Nothing in `aggregate` may depend on a wheel existing.

**Never quote a measurement from a debug build.** `maturin develop` without
`--release` is 10x to 50x slower and has produced more than one false
conclusion in projects of this shape.

## Explicitly not in this plan

**Replacing `aggregate`'s pure-Python engine.** It stays, it stays the
default when this package is absent, and it stays the conformance reference.
This repository is never a prerequisite for a correct answer.

**A float CRT fast path.** `H/M` can be recovered as a sum of fractional
parts in O(n) vectorized passes, far faster than any exact reconstruction, but
it carries roughly 1e-16 *absolute* accuracy and so destroys the deep tail,
which is the entire reason the exact engine exists. If it is ever built it is
a separate mode with a separate name, never a silent substitution.

**Process-based parallelism in Python**, which was stage 2 of
`aggregate`'s `dev/plan-ntt-crt.md`. This repository supersedes it: threads
without a GIL are what that stage was trying to approximate with processes and
pickling.

**Distributions other than the bounded discrete compound.** The whole method
rests on the answer being a polynomial with bounded integer coefficients.
