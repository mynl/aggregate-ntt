//! Chinese remainder reconstruction, the engine's first milestone.
//!
//! `[Rust-Reconstruction]` in `dev/plan-rust-engine.md` specifies it; this is
//! the implementation. All arithmetic in this module is arbitrary precision
//! (`num_bigint::BigUint`) or explicitly bounded `u64`, so the overflow
//! discipline of `CLAUDE.md` is discharged by construction, with the bounds
//! stated where fixed-width values appear.
//!
//! # The shape, and why it is a stateful accumulator
//!
//! The obvious signature takes every prime's residue vector at once and
//! returns the answer. It is also wrong here: 321 vectors of 10^6 `u64` is
//! 2.6 GB handed across the boundary on top of the 1.25 GB the answer itself
//! occupies, and the Python caller currently consumes residue vectors one at
//! a time as its thread pool produces them. So the accumulator preserves that
//! online property: push each prime's residues as it arrives, and the merge
//! stack lives on the Rust side.
//!
//! # The merge stack
//!
//! The same structure `aggregate.ntt` uses in pure Python, so the two
//! implementations can be compared level by level. Each pushed prime becomes
//! a one-prime partial, and any two partials of equal weight on top of the
//! stack merge at once, weight counting the primes behind a partial. The
//! stack is a binary counter: it holds at most `log2(n)` partials, so peak
//! memory is roughly twice the answer rather than the answer plus every
//! prime's residue vector, and the expensive full-width merges fall at pushes
//! 64, 128, 256 with the final drain combining what the binary decomposition
//! of the prime count leaves behind.
//!
//! One merge is a Garner step on composite moduli: with `a` exact mod `Ma`
//! and `b` exact mod `Mb` coprime, the value exact mod `Ma * Mb` is
//!
//! ```text
//! a + Ma * ((b - a) * Ma^(-1) mod Mb)
//! ```
//!
//! in range because the bracket is reduced mod `Mb`. Only the scalar inverse
//! depends on the moduli, so it is computed once per merge and shared across
//! every coefficient.
//!
//! # Parallelism: inside the merge, over coefficients
//!
//! A merge is an elementwise map with no cross-coefficient dependency, so
//! `rayon` parallelizes it directly, behind a released GIL
//! (`Python::detach`). Parallelizing by prime subtree instead would need
//! partials combined across threads for no gain. Every merge is parallel,
//! including the narrow ones, with coarse task chunks so the bookkeeping
//! amortizes; see `PARALLEL_MIN_CHUNK` for the measurement that decided it.
//!
//! # The boundary, which is where a naive port loses its winnings
//!
//! `finish` must not build a million Python integers one at a time. Every
//! coefficient is smaller than the final modulus, so the width is uniform:
//! the answer comes back as one contiguous `(n_coefficients, n_limbs)` array
//! of little-endian 64-bit limbs, which is a memory copy rather than a
//! million object allocations, and the caller converts in bulk. Milestone 3
//! (`[Rust-Query-Object]`) adds a query object that answers for moments and
//! tail probabilities without materializing the coefficients at all.

use num_bigint::BigUint;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// Fewest coefficients per `rayon` task inside a merge.
///
/// The plan expected the narrow merges to stay serial behind a modulus-width
/// gate, because at one or two primes the per-coefficient Garner step is
/// tens of nanoseconds. Measured on DOOB on the headline shape (322 primes
/// of about 31 bits, 10^6 coefficients), the opposite holds once elements
/// are handed out in chunks this coarse: all-parallel reconstructs in 12.1 s
/// against 30.2 s with merges below 128 bits serial, because the narrow
/// levels are most of the merge count and the chunking keeps `rayon`'s
/// per-task bookkeeping far below even their arithmetic. So every merge is
/// parallel and the width gate collapsed into this chunk floor; the
/// measurement is in the execution log of `dev/plan-rust-engine.md`.
const PARALLEL_MIN_CHUNK: usize = 4096;

/// One partial reconstruction: coefficients exact modulo `modulus`, with
/// `weight` primes behind them.
///
/// `weight` is what makes the merge stack a *balanced* tree: a partial is
/// only ever merged with one of equal weight, so operand sizes double level
/// by level and no merge is lopsided.
struct Partial {
    coefficients: Vec<BigUint>,
    modulus: BigUint,
    weight: usize,
}

/// Merge two partial reconstructions over coprime moduli into one.
///
/// `left` plays `a` and `right` plays `b` in the Garner step, matching the
/// Python merge stack's argument order so the two can be compared level by
/// level (the reconstructed integers are unique below the product either
/// way).
///
/// The `BigUint` subtraction inside would panic on underflow, so the step is
/// arranged to stay nonnegative: `a mod Mb` and `b` both lie below `Mb`, and
/// the branch adds `Mb` before subtracting when `b` is the smaller.
fn merge(left: Partial, right: Partial) -> Partial {
    let inverse = left
        .modulus
        .modinv(&right.modulus)
        .expect("moduli are coprime: disjoint products of distinct primes");
    let ma = left.modulus;
    let mb = right.modulus;
    let garner = |(a, b): (BigUint, BigUint)| -> BigUint {
        let a_reduced = &a % &mb;
        let difference = if b >= a_reduced {
            b - a_reduced
        } else {
            &mb - a_reduced + b
        };
        let bracket = difference * &inverse % &mb;
        a + &ma * bracket
    };
    let coefficients: Vec<BigUint> = left
        .coefficients
        .into_par_iter()
        .zip(right.coefficients.into_par_iter())
        .with_min_len(PARALLEL_MIN_CHUNK)
        .map(garner)
        .collect();
    Partial {
        coefficients,
        modulus: ma * mb,
        weight: left.weight + right.weight,
    }
}

/// Online Chinese remainder reconstruction over NTT primes.
///
/// Construct with the primes in the order residues will be pushed, push one
/// residue vector per prime, then call `finish` once.
#[pyclass]
pub struct Reconstructor {
    primes: Vec<u64>,
    n_coefficients: usize,
    pushed: usize,
    finished: bool,
    stack: Vec<Partial>,
}

#[pymethods]
impl Reconstructor {
    /// Create an accumulator for `primes` over `n_coefficients` coefficients.
    ///
    /// The primes must be distinct `u64` values; they are the moduli the
    /// caller's transform phase worked in, in the order its residues arrive.
    /// The reconstruction itself is arbitrary-precision, so it carries no
    /// width ceiling of its own; the system ceiling of `2**63` belongs to
    /// the transform's Montgomery arithmetic (`residues_mod_prime`), which
    /// is where primes are selected against it. Widened from `u32` at
    /// 1.0.0a4 for `[Transform-64]`: a compatible widening, every earlier
    /// call shape still valid, so `PROTOCOL_VERSION` stays 2.
    #[new]
    fn new(primes: Vec<u64>, n_coefficients: usize) -> PyResult<Self> {
        if primes.is_empty() {
            return Err(PyValueError::new_err("primes must not be empty"));
        }
        if primes.iter().any(|&p| p < 2) {
            return Err(PyValueError::new_err("primes must be at least 2"));
        }
        let mut sorted = primes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != primes.len() {
            return Err(PyValueError::new_err("primes must be distinct"));
        }
        Ok(Self {
            primes,
            n_coefficients,
            pushed: 0,
            finished: false,
            stack: Vec::new(),
        })
    }

    /// Number of primes this accumulator expects.
    #[getter]
    fn n_primes(&self) -> usize {
        self.primes.len()
    }

    /// Number of residue vectors pushed so far.
    #[getter]
    fn pushed(&self) -> usize {
        self.pushed
    }

    /// Push one prime's residue vector, in the order the primes were given.
    ///
    /// The array must hold `n_coefficients` values, each already reduced
    /// modulo the corresponding prime; an unreduced value is rejected rather
    /// than folded in, because a wrong answer must lose to a loud refusal.
    /// The GIL is released for the merge work, so the caller's transform
    /// threads keep running while the stack absorbs the push.
    fn push(&mut self, py: Python<'_>, residues: PyReadonlyArray1<u64>) -> PyResult<()> {
        if self.finished {
            return Err(PyValueError::new_err("finish has already been called"));
        }
        if self.pushed >= self.primes.len() {
            return Err(PyValueError::new_err(
                "all residue vectors have already been pushed",
            ));
        }
        let view = residues.as_array();
        if view.len() != self.n_coefficients {
            return Err(PyValueError::new_err(format!(
                "expected {} coefficients, got {}",
                self.n_coefficients,
                view.len()
            )));
        }
        // Copy out while the GIL is held: the borrowed array cannot cross
        // detach, and the copy also normalizes any striding.
        let values: Vec<u64> = view.iter().copied().collect();
        let prime = self.primes[self.pushed];
        if let Some(bad) = values.iter().find(|&&r| r >= prime) {
            return Err(PyValueError::new_err(format!(
                "residue {bad} is not reduced modulo prime {prime}"
            )));
        }
        self.pushed += 1;
        let stack = &mut self.stack;
        py.detach(|| {
            // One heap allocation per coefficient; parallel because 10^6
            // small allocations per push add up to seconds across a run.
            let coefficients = values
                .into_par_iter()
                .with_min_len(PARALLEL_MIN_CHUNK)
                .map(BigUint::from)
                .collect();
            let mut partial = Partial {
                coefficients,
                modulus: BigUint::from(prime),
                weight: 1,
            };
            // Binary-counter merges: combine equal-weight neighbors so the
            // product tree stays balanced.
            while stack.last().is_some_and(|top| top.weight == partial.weight) {
                let left = stack.pop().expect("just checked nonempty");
                partial = merge(left, partial);
            }
            stack.push(partial);
        });
        Ok(())
    }

    /// Finish and return `(limbs, modulus_bytes)`.
    ///
    /// `limbs` is a `(n_coefficients, n_limbs)` array of little-endian 64-bit
    /// limbs, one row per coefficient, zero padded to the uniform width of
    /// the final modulus. `modulus_bytes` is the product of the primes,
    /// little-endian, which the caller turns into an integer with
    /// `int.from_bytes(..., 'little')`. Callable once, after every prime's
    /// residues have been pushed; the GIL is released for the drain.
    fn finish(&mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyBytes>)> {
        if self.finished {
            return Err(PyValueError::new_err("finish has already been called"));
        }
        if self.pushed != self.primes.len() {
            return Err(PyValueError::new_err(format!(
                "only {} of {} residue vectors have been pushed",
                self.pushed,
                self.primes.len()
            )));
        }
        self.finished = true;
        let mut stack = std::mem::take(&mut self.stack);
        let n_coefficients = self.n_coefficients;
        let (limbs, modulus_bytes) = py.detach(move || {
            // Drain smallest weight first, mirroring the Python stack: the
            // accumulated result is always the right operand of the merge.
            let mut accumulated = stack.pop().expect("at least one prime was pushed");
            while let Some(left) = stack.pop() {
                accumulated = merge(left, accumulated);
            }
            // Every coefficient lies below the modulus, so the modulus width
            // is the uniform row width. max(1) keeps a (n, 1) shape for the
            // degenerate one-tiny-prime case rather than a zero-width array.
            let n_limbs = usize::try_from(accumulated.modulus.bits().div_ceil(64))
                .expect("limb count fits usize")
                .max(1);
            let mut limbs = Array2::<u64>::zeros((n_coefficients, n_limbs));
            limbs
                .as_slice_mut()
                .expect("freshly allocated array is contiguous")
                .par_chunks_mut(n_limbs)
                .zip(accumulated.coefficients.into_par_iter())
                .for_each(|(row, value)| {
                    // iter_u64_digits is little-endian and never longer than
                    // n_limbs; the tail of the row stays zero.
                    for (slot, digit) in row.iter_mut().zip(value.iter_u64_digits()) {
                        *slot = digit;
                    }
                });
            (limbs, accumulated.modulus.to_bytes_le())
        });
        let limbs_py = PyArray2::from_owned_array(py, limbs).into_any().unbind();
        let modulus_py = PyBytes::new(py, &modulus_bytes).unbind();
        Ok((limbs_py, modulus_py))
    }
}
