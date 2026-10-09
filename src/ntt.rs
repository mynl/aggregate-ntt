//! Per-prime number-theoretic transform, the engine's second milestone.
//!
//! `[Rust-Transform]` in `dev/plan-rust-engine.md` specifies it; this is the
//! implementation. The Python reference is `aggregate.ntt._residues_mod_prime`
//! and the functions under it (`ntt`, `_spectral_walk`,
//! `_spectral_walk_singular`); this mirrors them step for step so the two can
//! be compared directly, with inputs arriving already reduced modulo the
//! working prime (see `residues_mod_prime` for the contract).
//!
//! # Arithmetic: one Montgomery field for both prime widths
//!
//! Everything runs in Montgomery form with `R = 2**64` over an odd prime
//! `p < 2**63`. One code path serves both the 32-bit primes the pure-Python
//! path is pinned to (products of two residues must fit `uint64` for numpy)
//! and the 64-bit primes that are this engine's prize, halving the prime
//! count for the same coefficient bound. The `2**63` ceiling is structural:
//! `add`, `sub` and the pre-subtraction Montgomery product all produce values
//! below `2 * p`, which must fit in `u64`.
//!
//! The sequence that defeated the abandoned numba attempt, recorded in
//! `aggregate`'s `dev/done/plan-ntt-speed.md`, was a Montgomery reduction
//! written as `t + m * p` in a word too narrow for the sum. Here the sum is
//! taken in `u128` where it provably fits; the bound is stated at the line.
//!
//! # Parallelism
//!
//! None inside a call, deliberately. `residues_mod_prime` is a pure function
//! that releases the GIL for the whole computation, and the caller runs one
//! call per prime on its thread pool, so across-prime parallelism uses every
//! core with no `rayon` oversubscription against the reconstruction's own
//! pool.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;

/// Runs shorter than this keep the per-outcome walk; at and above it the
/// closed-form geometric series pays for its two array exponentiations.
/// Mirrors `aggregate.ntt._GEOMETRIC_RUN_MIN` so the two implementations take
/// the same branch on the same input, which keeps them comparable step by
/// step even though both branches produce identical residues.
const GEOMETRIC_RUN_MIN: u64 = 8;

/// Montgomery arithmetic over an odd prime `p < 2**63`, word size `2**64`.
#[derive(Clone, Copy)]
struct Field {
    p: u64,
    /// `-(p**-1) mod 2**64`, the Montgomery reduction multiplier.
    p_inv_neg: u64,
    /// `2**64 mod p`: the Montgomery form of one.
    one: u64,
    /// `2**128 mod p`: converts into Montgomery form via `mul`.
    r2: u64,
}

impl Field {
    /// Build the field constants for an odd prime below `2**63`.
    fn new(p: u64) -> Self {
        debug_assert!(p % 2 == 1 && (3..(1u64 << 63)).contains(&p));
        // Newton-Hensel lifting: x -> x * (2 - p * x) doubles the number of
        // correct low bits of p**-1 mod 2**64. x = p is correct to three
        // bits (p * p ≡ 1 mod 8 for odd p), so five steps reach 96 bits,
        // past the 64 needed. Every operation wraps mod 2**64 by design.
        let mut inv = p;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        let one = ((1u128 << 64) % p as u128) as u64;
        let r2 = ((one as u128 * one as u128) % p as u128) as u64;
        Field {
            p,
            p_inv_neg: inv.wrapping_neg(),
            one,
            r2,
        }
    }

    /// Montgomery product `a * b * 2**-64 mod p` for `a, b < p`.
    ///
    /// Overflow bound: `t = a * b < p**2 < 2**126` and
    /// `m * p < 2**64 * p < 2**127`, so `t + m * p < 2**127 + 2**126`,
    /// which fits `u128` with a bit to spare. This is the sum that, taken
    /// one word too narrow, killed the numba attempt; in `u128` at
    /// `p < 2**63` it provably cannot wrap. The low 64 bits of the sum are
    /// zero by construction of `m`, the shifted value is below `2 * p`
    /// (`t / 2**64 < p` and `m * p / 2**64 < p`), and one conditional
    /// subtraction lands it below `p`.
    #[inline(always)]
    fn mul(&self, a: u64, b: u64) -> u64 {
        let t = a as u128 * b as u128;
        let m = (t as u64).wrapping_mul(self.p_inv_neg);
        let u = ((t + m as u128 * self.p as u128) >> 64) as u64;
        if u >= self.p {
            u - self.p
        } else {
            u
        }
    }

    /// `a + b mod p` for `a, b < p`. The sum is below `2 * p < 2**64`.
    #[inline(always)]
    fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.p {
            s - self.p
        } else {
            s
        }
    }

    /// `a - b mod p` for `a, b < p`. `a + p < 2**64` for `p < 2**63`.
    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a + self.p - b
        }
    }

    /// Montgomery form of `x < p`.
    #[inline]
    fn montgomery(&self, x: u64) -> u64 {
        self.mul(x, self.r2)
    }

    /// Plain residue of a Montgomery-form value.
    #[inline]
    fn plain(&self, x: u64) -> u64 {
        self.mul(x, 1)
    }

    /// Scalar `base**exponent`, both in Montgomery form.
    fn pow(&self, base: u64, mut exponent: u64) -> u64 {
        let mut result = self.one;
        let mut current = base;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.mul(result, current);
            }
            exponent >>= 1;
            if exponent > 0 {
                current = self.mul(current, current);
            }
        }
        result
    }

    /// Scalar `x**-1 mod p` in Montgomery form, by Fermat: `x**(p - 2)`.
    fn inv(&self, x: u64) -> u64 {
        self.pow(x, self.p - 2)
    }
}

/// `dst[i] = dst[i] * src[i]`, elementwise Montgomery products.
#[inline]
fn mul_assign(dst: &mut [u64], src: &[u64], field: &Field) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = field.mul(*d, s);
    }
}

/// In-place iterative radix-2 decimation-in-time transform, Montgomery form.
///
/// `w` is a primitive `length`-th root of unity in Montgomery form, already
/// validated by the caller; pass the inverse root for the inverse transform
/// (the `1/length` scaling is the caller's, folded into its Montgomery exit).
/// The structure mirrors `aggregate.ntt.ntt` level for level: bit-reversal
/// permutation, then butterflies over twiddles `w**(j * stride)` read from a
/// table of the first `length / 2` powers.
fn ntt_in_place(a: &mut [u64], field: &Field, w: u64) {
    let length = a.len();
    debug_assert!(length >= 2 && length.is_power_of_two());
    let bits = length.trailing_zeros();
    for i in 0..length {
        let j = ((i as u64).reverse_bits() >> (64 - bits)) as usize;
        if j > i {
            a.swap(i, j);
        }
    }
    // The twiddle table is rebuilt per call like the Python side, a serial
    // chain of length / 2 Montgomery products, negligible against the
    // length * log2(length) butterflies it feeds.
    let half = length / 2;
    let mut table = Vec::with_capacity(half);
    table.push(field.one);
    for i in 1..half {
        let previous = table[i - 1];
        table.push(field.mul(previous, w));
    }
    let mut m = 1;
    while m < length {
        let stride = length / (2 * m);
        for block in a.chunks_exact_mut(2 * m) {
            let (u, v) = block.split_at_mut(m);
            for j in 0..m {
                let t = field.mul(v[j], table[j * stride]);
                let x = u[j];
                u[j] = field.add(x, t);
                v[j] = field.sub(x, t);
            }
        }
        m *= 2;
    }
}

/// Elementwise `base**exponent`, Montgomery form, by ascending-bit binary
/// exponentiation; mirrors `aggregate.ntt._array_pow`.
fn array_pow(base: &[u64], mut exponent: u64, field: &Field) -> Vec<u64> {
    let mut result = vec![field.one; base.len()];
    if exponent == 0 {
        return result;
    }
    let mut current = base.to_vec();
    loop {
        if exponent & 1 == 1 {
            mul_assign(&mut result, &current, field);
        }
        exponent >>= 1;
        if exponent == 0 {
            return result;
        }
        for x in current.iter_mut() {
            *x = field.mul(*x, *x);
        }
    }
}

/// Elementwise inverse by Montgomery's batch trick, Montgomery form.
///
/// One scalar Fermat inversion plus three multiplies per element, against
/// the roughly `2 * log2(p)` full-array multiplies of the Python side's
/// vectorized Fermat exponentiation (`_array_pow` to `p - 2`). The inverse
/// is unique, so the values are identical; only the route differs, and at
/// 64-bit primes the exponent doubles while this stays three multiplies.
/// Zero cells come back zero, matching `0**(p - 2) mod p = 0` on the Python
/// side; the caller overwrites those cells anyway.
fn batch_inverse(values: &[u64], field: &Field) -> Vec<u64> {
    let n = values.len();
    // prefix[i] is the product of the nonzero entries of values[..=i].
    let mut prefix = Vec::with_capacity(n);
    let mut accumulated = field.one;
    for &x in values {
        if x != 0 {
            accumulated = field.mul(accumulated, x);
        }
        prefix.push(accumulated);
    }
    let mut inverse_accumulated = field.inv(accumulated);
    let mut out = vec![0u64; n];
    for i in (0..n).rev() {
        let x = values[i];
        if x != 0 {
            let before = if i == 0 { field.one } else { prefix[i - 1] };
            out[i] = field.mul(inverse_accumulated, before);
            inverse_accumulated = field.mul(inverse_accumulated, x);
        }
    }
    out
}

/// Bring `power` from `b**held` to `b**(held + gap)` by square-and-multiply
/// over the bits of `gap`, mirroring the gap bridging in `_spectral_walk`.
/// Allocates the running square lazily, so a gap of one costs a single
/// elementwise multiply and no allocation.
fn bridge_gap(power: &mut [u64], b: &[u64], mut gap: u64, field: &Field) {
    if gap == 0 {
        return;
    }
    let mut squared: Option<Vec<u64>> = None;
    let mut first = true;
    loop {
        if !first {
            match squared.as_mut() {
                None => {
                    let mut s = b.to_vec();
                    for x in s.iter_mut() {
                        *x = field.mul(*x, *x);
                    }
                    squared = Some(s);
                }
                Some(s) => {
                    for x in s.iter_mut() {
                        *x = field.mul(*x, *x);
                    }
                }
            }
        }
        first = false;
        if gap & 1 == 1 {
            mul_assign(power, squared.as_deref().unwrap_or(b), field);
        }
        gap >>= 1;
        if gap == 0 {
            return;
        }
    }
}

/// Spectral evaluation of `H` with the severity denominator folded in.
///
/// Mirrors `aggregate.ntt._spectral_walk`: with `b = a_hat * ds**-1`,
/// `H_hat = ds**kmax * sum_i F_i * b**k_i`, a maximal run of consecutive
/// outcomes contributing the geometric series
/// `W * b**u * (b**(v - u + 1) - 1) * (b - 1)**-1`, the `b == 1` cells
/// taking the degenerate value `run_length mod p` (index 0 hits this on
/// every prime, since `a_hat_0 = A(1) = ds` makes `b_0 = 1`). Runs shorter
/// than `GEOMETRIC_RUN_MIN` walk outcome by outcome. All values Montgomery.
fn spectral_walk(
    a_hat: &[u64],
    runs: &[(u64, u64, u64)],
    ds: u64,
    kmax: u64,
    field: &Field,
) -> Vec<u64> {
    let n = a_hat.len();
    let inv_ds = field.inv(field.montgomery(ds));
    let mut b = vec![0u64; n];
    for (bi, &ai) in b.iter_mut().zip(a_hat) {
        *bi = field.mul(ai, inv_ds);
    }
    let one = field.one;
    let mut total = vec![0u64; n];
    let mut power = vec![one; n];
    let mut held: u64 = 0;
    let mut inv_bm1: Option<Vec<u64>> = None;
    for &(start, run_length, weight) in runs {
        bridge_gap(&mut power, &b, start - held, field);
        held = start;
        let w = field.montgomery(weight);
        if run_length < GEOMETRIC_RUN_MIN {
            for j in 0..run_length {
                for i in 0..n {
                    let term = field.mul(power[i], w);
                    total[i] = field.add(total[i], term);
                }
                if j + 1 < run_length {
                    mul_assign(&mut power, &b, field);
                    held += 1;
                }
            }
        } else {
            let inverse = inv_bm1.get_or_insert_with(|| {
                let mut bm1 = vec![0u64; n];
                for (c, &bi) in bm1.iter_mut().zip(&b) {
                    *c = field.sub(bi, one);
                }
                batch_inverse(&bm1, field)
            });
            let series = array_pow(&b, run_length, field);
            let degenerate = field.montgomery(run_length % field.p);
            for i in 0..n {
                let s = if b[i] == one {
                    degenerate
                } else {
                    field.mul(field.sub(series[i], one), inverse[i])
                };
                let s = field.mul(field.mul(s, power[i]), w);
                total[i] = field.add(total[i], s);
            }
        }
    }
    let scale = field.pow(field.montgomery(ds), kmax);
    for t in total.iter_mut() {
        *t = field.mul(*t, scale);
    }
    total
}

/// Spectral evaluation when the prime divides the severity denominator.
///
/// Mirrors `aggregate.ntt._spectral_walk_singular`: the geometric path needs
/// `ds**-1 mod p`, which does not exist when `p` divides `ds` (the caller
/// signals this by passing the reduced `ds == 0`). This is the plain
/// ascending walk on `a_hat` with the explicit `ds**(kmax - k)` coefficient,
/// under which `ds ≡ 0` kills every term below the largest outcome
/// (`0**0 = 1` carries the survivor). Runs expand back to their outcomes.
fn spectral_walk_singular(
    a_hat: &[u64],
    runs: &[(u64, u64, u64)],
    ds: u64,
    kmax: u64,
    field: &Field,
) -> Vec<u64> {
    let n = a_hat.len();
    let ds_mont = field.montgomery(ds);
    let mut total = vec![0u64; n];
    let mut power = vec![field.one; n];
    let mut held: u64 = 0;
    for &(start, run_length, weight) in runs {
        for j in 0..run_length {
            let k = start + j;
            bridge_gap(&mut power, a_hat, k - held, field);
            held = k;
            let coefficient = field.mul(field.montgomery(weight), field.pow(ds_mont, kmax - k));
            for i in 0..n {
                total[i] = field.add(total[i], field.mul(power[i], coefficient));
            }
        }
    }
    total
}

/// One prime's residues of the cleared-denominator compound polynomial.
///
/// Forward transform of the severity polynomial, spectral evaluation of the
/// compound polynomial over the frequency outcome runs, inverse transform.
/// Pure function of its arguments; the GIL is released for the whole
/// computation, so the caller's thread pool gets real across-prime
/// parallelism.
///
/// Parameters mirror `aggregate.ntt._residues_mod_prime` with one deliberate
/// difference: **everything arrives already reduced modulo `prime`**. The
/// cleared integer weights on the Python side are arbitrary-precision (they
/// can exceed any single prime), and the Python reference reduces them per
/// prime at the top of its own body; this seam moves that one-line reduction
/// to the caller so the boundary carries fixed-width `u64` only. So:
/// `sev_weights` is the severity polynomial's cleared weights mod `prime`
/// on the sparse support `sev_index`; `freq_runs` is the frequency outcome
/// decomposition into maximal equal-weight runs as `(start, run_length,
/// weight mod prime)` triples; `ds` is the severity denominator mod `prime`
/// (zero exactly when the prime divides it, which routes the singular walk);
/// `kmax` is the largest claim count. `prime` must be odd and below `2**63`
/// (the Montgomery headroom bound) and `root` a primitive `length`-th root
/// of unity. Unreduced inputs are rejected, never folded in.
///
/// Returns the `uint64` residue vector truncated to `support + 1`, raising
/// `AssertionError` if any residue beyond the support is nonzero (the
/// circular-aliasing check, decisive per prime, mirrored from Python).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn residues_mod_prime(
    py: Python<'_>,
    prime: u64,
    root: u64,
    length: usize,
    support: usize,
    sev_index: PyReadonlyArray1<i64>,
    sev_weights: PyReadonlyArray1<u64>,
    freq_runs: Vec<(u64, u64, u64)>,
    ds: u64,
    kmax: u64,
) -> PyResult<Py<PyAny>> {
    if prime < 3 || prime % 2 == 0 || prime >= (1 << 63) {
        return Err(PyValueError::new_err(format!(
            "prime must be an odd prime below 2**63, got {prime} \
             (the ceiling is Montgomery headroom: 2 * prime must fit u64)"
        )));
    }
    if length < 2 || !length.is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "transform length must be a power of two >= 2, got {length}"
        )));
    }
    if support >= length {
        return Err(PyValueError::new_err(format!(
            "support {support} does not fit transform length {length}"
        )));
    }
    let field = Field::new(prime);
    let root_mont = if root < prime {
        field.montgomery(root)
    } else {
        0
    };
    if root >= prime
        || field.pow(root_mont, length as u64) != field.one
        || field.pow(root_mont, (length / 2) as u64) == field.one
    {
        return Err(PyValueError::new_err(format!(
            "root {root} is not a primitive {length}-th root of unity \
             mod {prime}"
        )));
    }
    if ds >= prime {
        return Err(PyValueError::new_err(format!(
            "ds {ds} is not reduced modulo prime {prime}"
        )));
    }
    let index_view = sev_index.as_array();
    let weight_view = sev_weights.as_array();
    if index_view.len() != weight_view.len() {
        return Err(PyValueError::new_err(format!(
            "sev_index has {} entries but sev_weights has {}",
            index_view.len(),
            weight_view.len()
        )));
    }
    let mut indices = Vec::with_capacity(index_view.len());
    for &i in index_view.iter() {
        if i < 0 || i as usize >= length {
            return Err(PyValueError::new_err(format!(
                "severity index {i} is outside the transform length {length}"
            )));
        }
        indices.push(i as usize);
    }
    let weights: Vec<u64> = weight_view.iter().copied().collect();
    if let Some(bad) = weights.iter().find(|&&w| w >= prime) {
        return Err(PyValueError::new_err(format!(
            "severity weight {bad} is not reduced modulo prime {prime}"
        )));
    }
    if freq_runs.is_empty() {
        return Err(PyValueError::new_err("freq_runs must not be empty"));
    }
    let mut next_start: u64 = 0;
    for &(start, run_length, weight) in &freq_runs {
        if run_length == 0 {
            return Err(PyValueError::new_err(
                "freq_runs run_length must be positive",
            ));
        }
        if start < next_start {
            return Err(PyValueError::new_err(
                "freq_runs must be ascending and non-overlapping",
            ));
        }
        if weight >= prime {
            return Err(PyValueError::new_err(format!(
                "frequency weight {weight} is not reduced modulo \
                 prime {prime}"
            )));
        }
        next_start = start + run_length;
    }
    if next_start - 1 != kmax {
        return Err(PyValueError::new_err(format!(
            "freq_runs end at {} but kmax is {kmax}",
            next_start - 1
        )));
    }
    let head = py.detach(move || -> PyResult<Vec<u64>> {
        // Severity polynomial on the padded lattice, Montgomery form.
        let mut a = vec![0u64; length];
        for (&i, &w) in indices.iter().zip(&weights) {
            a[i] = field.montgomery(w);
        }
        ntt_in_place(&mut a, &field, root_mont);
        let h_hat = if ds == 0 {
            spectral_walk_singular(&a, &freq_runs, ds, kmax, &field)
        } else {
            spectral_walk(&a, &freq_runs, ds, kmax, &field)
        };
        let mut residues = h_hat;
        ntt_in_place(&mut residues, &field, field.inv(root_mont));
        // One pass both scales by 1/length and leaves Montgomery form:
        // mul(x_mont, s_plain) = x * s mod p, with s = length**-1 mod p as
        // a plain residue.
        let length_inv_plain = field.plain(field.inv(field.montgomery(length as u64 % prime)));
        for r in residues.iter_mut() {
            *r = field.mul(*r, length_inv_plain);
        }
        if residues[support + 1..].iter().any(|&r| r != 0) {
            return Err(PyAssertionError::new_err(format!(
                "NTT internal check failed: nonzero residue beyond the \
                 support mod {prime}"
            )));
        }
        residues.truncate(support + 1);
        Ok(residues)
    })?;
    Ok(PyArray1::from_vec(py, head).into_any().unbind())
}
