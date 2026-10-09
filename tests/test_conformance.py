"""
Conformance of the compiled engine against the pure-Python reference.

Every kernel here has an independent, slower, already-trusted reference in
:mod:`aggregate.ntt`, so every test in this file asserts **bit-for-bit
equality of integers**, never a tolerance. Exact arithmetic has no tolerance,
and a test that allows one has stopped testing the property that matters.

Three references, in increasing independence:

- :func:`aggregate.ntt.exact_compound` on its pure-Python path, the same
  algorithm this engine reimplements.
- :func:`aggregate.ntt.exact_compound_poly`, a direct polynomial computation
  that does not use a transform at all, so it shares no machinery with either
  implementation.
- The three end-to-end checks inside ``exact_compound``, which run on this
  engine's answer too: coefficients sum to exactly the denominator, none is
  negative, and none beyond the support is nonzero.

Both kernels have landed, so the per-kernel tests run for real. The four
end-to-end tests driving :func:`aggregate.ntt.exact_compound` with
``ntt_engine='rust'`` stay expected failures until ``aggregate`` speaks seam
protocol 2 (its ``_ENGINE_PROTOCOL``), at which point they flip to XPASS and
their markers come off.
"""

import random
from fractions import Fraction
from math import lcm

import numpy as np
import pytest

import aggregate_ntt

aggregate_ntt_module = aggregate_ntt


# --------------------------------------------------------------- [the seam]

def test_protocol_version_is_an_integer():
    """The seam contract version is readable and sane."""
    assert isinstance(aggregate_ntt.PROTOCOL_VERSION, int)
    assert aggregate_ntt.PROTOCOL_VERSION >= 1


def test_engine_version_matches_cargo():
    """``engine_version`` reports the Cargo version, not the PEP 440 one.

    The two spellings differ deliberately (``1.0.0-alpha.1`` against
    ``1.0.0a1``); what matters is that the compiled module reports its own
    build rather than the Python package metadata, so a stale ``.pyd`` is
    visible in an ``aggregate`` audit record.
    """
    assert aggregate_ntt.__version__
    assert aggregate_ntt.engine_version() == aggregate_ntt.__version__


def test_reconstructor_rejects_bad_primes():
    """Construction validates before any work is queued."""
    with pytest.raises(ValueError, match='must not be empty'):
        aggregate_ntt.Reconstructor([], 4)
    with pytest.raises(ValueError, match='distinct'):
        aggregate_ntt.Reconstructor([17, 17], 4)


def test_reconstructor_reports_its_shape():
    r = aggregate_ntt.Reconstructor([17, 97], 4)
    assert r.n_primes == 2
    assert r.pushed == 0


def test_reconstructor_rejects_a_wrong_length_push():
    """A residue vector of the wrong length is caught, not reinterpreted."""
    r = aggregate_ntt.Reconstructor([17, 97], 4)
    with pytest.raises(ValueError, match='expected 4 coefficients'):
        r.push(np.zeros(5, dtype=np.uint64))


def test_reconstructor_rejects_an_unreduced_residue():
    """A residue at or above its prime is refused, never folded in.

    The seam contract says residues arrive already reduced; a violation
    means the caller's transform is broken, and a loud refusal beats a
    silently wrong reconstruction.
    """
    r = aggregate_ntt.Reconstructor([17, 97], 3)
    with pytest.raises(ValueError, match='not reduced modulo prime 17'):
        r.push(np.array([0, 17, 1], dtype=np.uint64))


def test_reconstructor_lifecycle_is_enforced():
    """Finish needs every push; finish and push both refuse a second life."""
    r = aggregate_ntt.Reconstructor([17, 97], 2)
    r.push(np.array([3, 5], dtype=np.uint64))
    with pytest.raises(ValueError, match='only 1 of 2'):
        r.finish()
    r.push(np.array([10, 20], dtype=np.uint64))
    r.finish()
    with pytest.raises(ValueError, match='finish has already been called'):
        r.finish()
    with pytest.raises(ValueError, match='finish has already been called'):
        r.push(np.array([1, 2], dtype=np.uint64))


# ------------------------------------------------- [reconstruction kernel]

def _reconstruct(primes, residues):
    """Drive the engine over ``residues``, returning ``(numerators, M)``."""
    n = len(residues[0])
    r = aggregate_ntt.Reconstructor([int(p) for p in primes], n)
    for head in residues:
        r.push(np.asarray(head, dtype=np.uint64))
    limbs, modulus_bytes = r.finish()
    modulus = int.from_bytes(modulus_bytes, 'little')
    numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]
    return numerators, modulus


@pytest.mark.parametrize('seed', range(4))
def test_reconstruction_matches_python_crt(seed):
    """The engine reconstructs the integers the Python merge stack does.

    Residues are drawn at random per prime, which is a harder test than a
    real case: a real residue vector is the image of a single integer below
    the modulus, where random residues exercise the full range of the
    reconstruction including coefficients near the modulus. The value in
    ``0 .. M - 1`` matching every residue is unique, so checking residues and
    range *is* bit-for-bit equality with any correct reconstruction.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    ntt_primes = ntt.ntt_primes

    pairs = ntt_primes(2 ** 10, 10 ** 60)
    primes = [p for p, _ in pairs]
    rng = np.random.default_rng(seed)
    residues = [rng.integers(0, p, size=64, dtype=np.uint64) for p in primes]

    numerators, modulus = _reconstruct(primes, residues)

    assert modulus == np.prod([int(p) for p in primes], dtype=object)
    for j, value in enumerate(numerators):
        assert 0 <= value < modulus
        for prime, head in zip(primes, residues):
            assert value % prime == int(head[j])


# Distinct primes congruent to 1 mod 1024, hardcoded so the drain shapes run
# on a machine without an aggregate checkout. 13 primes exercise the binary
# counter through several levels and a drain of 13 = 8 + 4 + 1.
_STANDALONE_PRIMES = [
    4294957057, 4294955009, 4294935553, 4294929409, 4294924289,
    4294921217, 4294914049, 4294895617, 4294890497, 4294884353,
    4294878209, 4294850561, 4294828033,
]

# Distinct 63-bit primes congruent to 1 mod 1024, just below the engine's
# 2**63 transform ceiling, hardcoded and verified like the set above. These
# pin the [Transform-64] configuration: wide primes through both kernels.
_STANDALONE_PRIMES_64 = [
    9223372036854758401, 9223372036854739969, 9223372036854675457,
    9223372036854618113, 9223372036854615041,
]


@pytest.mark.parametrize('n_primes', [1, 2, 3, 13])
def test_reconstruction_drain_shapes(n_primes):
    """Every merge-stack drain shape, including the single-prime pass-through.

    The prime counts cover a bare push (1), one merge (2), a drain across
    unequal weights (3), and a multi-level drain (13 = 8 + 4 + 1).
    """
    primes = _STANDALONE_PRIMES[:n_primes]
    rng = np.random.default_rng(n_primes)
    residues = [rng.integers(0, p, size=17, dtype=np.uint64) for p in primes]

    numerators, modulus = _reconstruct(primes, residues)

    assert modulus == np.prod([int(p) for p in primes], dtype=object)
    for j, value in enumerate(numerators):
        assert 0 <= value < modulus
        for prime, head in zip(primes, residues):
            assert value % prime == int(head[j])


def test_reconstruction_accepts_64bit_primes():
    """The widened seam: `u64` primes through the merge stack.

    Same uniqueness argument as the 32-bit drain shapes; what this adds is
    primes above `2**32`, refused by the seam before 1.0.0a4.
    """
    primes = _STANDALONE_PRIMES_64
    rng = np.random.default_rng(64)
    residues = [rng.integers(0, p, size=17, dtype=np.uint64) for p in primes]

    numerators, modulus = _reconstruct(primes, residues)

    assert modulus == np.prod([int(p) for p in primes], dtype=object)
    for j, value in enumerate(numerators):
        assert 0 <= value < modulus
        for prime, head in zip(primes, residues):
            assert value % prime == int(head[j])


def test_limb_width_is_the_modulus_width():
    """Rows are zero padded to exactly the final modulus's 64-bit width."""
    primes = _STANDALONE_PRIMES[:5]
    residues = [np.zeros(3, dtype=np.uint64) for _ in primes]
    r = aggregate_ntt.Reconstructor(primes, 3)
    for head in residues:
        r.push(head)
    limbs, modulus_bytes = r.finish()
    modulus = int.from_bytes(modulus_bytes, 'little')
    n_limbs = -(-modulus.bit_length() // 64)
    assert limbs.shape == (3, n_limbs)
    assert limbs.dtype == np.uint64


@pytest.mark.parametrize('seed', range(4))
def test_reconstruction_recovers_oracle_coefficients(seed):
    """End to end against the polynomial oracle, residues taken directly.

    ``exact_compound_poly`` multiplies polynomials in Python integers with no
    transform and no CRT. Reducing its exact coefficients modulo each prime
    and reconstructing must give back the same integers bit for bit, because
    the CRT modulus exceeds the denominator, which bounds every coefficient.
    This pins the reconstruction against a reference that shares no machinery
    with it, without needing the transform kernel.
    """
    ntt = pytest.importorskip('aggregate.ntt')

    rng = random.Random(seed)
    m = rng.randint(1, 4)
    outcomes = sorted(set(rng.randint(0, 12) for _ in range(m)))
    weights = [rng.randint(1, 9) for _ in outcomes]
    total = sum(weights)
    probs = [Fraction(w, total) for w in weights]
    sev_outcomes = sorted(set(rng.randint(0, 9)
                              for _ in range(rng.randint(1, 4))))
    sev_weights = [rng.randint(1, 9) for _ in sev_outcomes]
    sev_total = sum(sev_weights)
    sev_probs = [Fraction(w, sev_total) for w in sev_weights]

    expected, denominator, _ = ntt.exact_compound_poly(
        outcomes, probs, sev_outcomes, sev_probs)

    pairs = ntt.ntt_primes(2 ** 10, denominator)
    primes = [p for p, _ in pairs]
    residues = [np.array([v % p for v in expected], dtype=np.uint64)
                for p in primes]
    numerators, modulus = _reconstruct(primes, residues)

    assert modulus > denominator
    assert numerators == expected


@pytest.mark.xfail(reason='aggregate speaks seam protocol 1 and this engine '
                          'speaks 2, so ntt_engine="rust" refuses; flips to '
                          'XPASS when aggregate bumps _ENGINE_PROTOCOL',
                   strict=False)
@pytest.mark.parametrize('seed', range(4))
def test_exact_compound_matches_the_polynomial_oracle(seed):
    """End to end against the oracle that shares no machinery with either.

    ``exact_compound_poly`` multiplies polynomials directly in Python
    integers, with no transform and no CRT, so agreement here is evidence
    about the whole pipeline rather than about one kernel.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    exact_compound = ntt.exact_compound
    exact_compound_poly = ntt.exact_compound_poly

    rng = random.Random(seed)
    m = rng.randint(1, 4)
    outcomes = [rng.randint(0, 8) for _ in range(m)]
    weights = [rng.randint(1, 9) for _ in range(m)]
    total = sum(weights)
    probs = [Fraction(w, total) for w in weights]
    sev_outcomes = [rng.randint(0, 6) for _ in range(rng.randint(1, 4))]
    sev_weights = [rng.randint(1, 9) for _ in sev_outcomes]
    sev_total = sum(sev_weights)
    sev_probs = [Fraction(w, sev_total) for w in sev_weights]

    fast = exact_compound(outcomes, probs, sev_outcomes, sev_probs,
                          ntt_engine='rust')
    slow = exact_compound_poly(outcomes, probs, sev_outcomes, sev_probs)
    assert fast[1] == slow[1]
    assert fast[0] == slow[0]


# ----------------------------------------------------- [transform kernel]

def _cleared(probs):
    """Least common denominator and integer weights of exact probabilities.

    The same construction ``aggregate.ntt._clear_denominators`` performs,
    reimplemented here (it is two lines) so the only private reach in this
    file stays ``_residues_mod_prime``, the sanctioned comparison target.
    """
    denominator = lcm(*[f.denominator for f in probs])
    return denominator, [int(f * denominator) for f in probs]


def _maximal_runs(outcomes, weights):
    """Maximal equal-weight consecutive runs, as ``aggregate`` builds them."""
    runs = []
    start, weight, run_length = outcomes[0], weights[0], 1
    for k, f in zip(outcomes[1:], weights[1:]):
        if k == start + run_length and f == weight:
            run_length += 1
        else:
            runs.append((start, run_length, weight))
            start, weight, run_length = k, f, 1
    runs.append((start, run_length, weight))
    return runs


def _transform_inputs(freq_outcomes, freq_probs, sev_outcomes, sev_probs):
    """Everything ``exact_compound`` derives before the per-prime phase."""
    df, big_f = _cleared(freq_probs)
    ds, big_q = _cleared(sev_probs)
    kmax = max(freq_outcomes)
    support = kmax * max(sev_outcomes)
    length = 1 << support.bit_length()
    denominator = df * ds ** kmax
    sev_index = np.array(sev_outcomes, dtype=np.int64)
    freq_runs = _maximal_runs(freq_outcomes, big_f)
    return sev_index, big_q, freq_runs, ds, kmax, support, length, denominator


def _engine_residues(prime, root, length, support, sev_index, big_q,
                     freq_runs, ds, kmax):
    """Call the engine with the caller-reduced inputs the seam specifies."""
    return aggregate_ntt.residues_mod_prime(
        prime, root, length, support, sev_index,
        np.array([q % prime for q in big_q], dtype=np.uint64),
        [(start, run_length, weight % prime)
         for start, run_length, weight in freq_runs],
        ds % prime, kmax)


# Three shapes that between them cover every spectral branch: a uniform case
# (one long run, the geometric series and its b == 1 degenerate mask), an
# irregular case (singleton runs, gap bridging, a zero frequency outcome),
# and a weighted case with mixed denominators (nontrivial ds and df, one run
# long enough for the series next to short walked runs).
_TRANSFORM_CASES = [
    (list(range(1, 13)), [Fraction(1, 12)] * 12,
     list(range(1, 13)), [Fraction(1, 12)] * 12),
    ([0, 1, 2, 5, 9], [Fraction(1, 4), Fraction(1, 8), Fraction(1, 8),
                       Fraction(1, 4), Fraction(1, 4)],
     [0, 2, 7], [Fraction(2, 5), Fraction(1, 5), Fraction(2, 5)]),
    (list(range(2, 11)), [Fraction(1, 18)] * 8 + [Fraction(5, 9)],
     [1, 2, 3], [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]),
]


@pytest.mark.parametrize('case', range(len(_TRANSFORM_CASES)))
def test_residues_mod_prime_matches_python(case):
    """One prime's residues, against ``aggregate.ntt._residues_mod_prime``.

    Bit-for-bit per prime on every prime the case needs. The reference call
    is private on purpose: this is the one place a private name is
    legitimately the comparison target. The engine receives caller-reduced
    inputs per its seam contract; the reference reduces internally.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    freq_outcomes, freq_probs, sev_outcomes, sev_probs = \
        _TRANSFORM_CASES[case]
    (sev_index, big_q, freq_runs, ds, kmax, support, length,
     denominator) = _transform_inputs(freq_outcomes, freq_probs,
                                      sev_outcomes, sev_probs)
    for prime, root in ntt.ntt_primes(length, denominator):
        expected, _ = ntt._residues_mod_prime(
            prime, root, length, support, sev_index, big_q, freq_runs,
            ds, kmax)
        got = _engine_residues(prime, root, length, support, sev_index,
                               big_q, freq_runs, ds, kmax)
        assert got.dtype == np.uint64
        assert np.array_equal(got, expected), f'prime {prime}'


def test_residues_mod_prime_singular_prime():
    """A prime dividing the severity denominator takes the singular walk.

    The geometric path needs ``ds**-1 mod p``, which does not exist when
    ``p`` divides ``ds``; the reduced ``ds == 0`` routes the engine's
    singular walk exactly as ``ds % prime == 0`` routes Python's. Built by
    putting an NTT prime itself into the severity denominators.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    pairs = ntt.ntt_primes(16, 10 ** 20)
    singular_prime = pairs[0][0]
    freq_outcomes, freq_probs = [1, 2], [Fraction(1, 2), Fraction(1, 2)]
    sev_outcomes = [0, 1]
    sev_probs = [Fraction(1, singular_prime),
                 Fraction(singular_prime - 1, singular_prime)]
    (sev_index, big_q, freq_runs, ds, kmax, support, length,
     denominator) = _transform_inputs(freq_outcomes, freq_probs,
                                      sev_outcomes, sev_probs)
    assert ds == singular_prime and length == 4
    for prime, root16 in pairs:
        # The pairs carry 16th roots; this case's length is 4, and
        # root16**4 has exact order 4.
        root = pow(root16, 4, prime)
        expected, _ = ntt._residues_mod_prime(
            prime, root, length, support, sev_index, big_q, freq_runs,
            ds, kmax)
        got = _engine_residues(prime, root, length, support, sev_index,
                               big_q, freq_runs, ds, kmax)
        assert np.array_equal(got, expected), f'prime {prime}'
    assert ds % pairs[0][0] == 0, 'the singular branch was exercised'


@pytest.mark.parametrize('seed', range(4))
def test_transform_end_to_end_matches_oracle(seed):
    """The whole engine pipeline against the polynomial oracle.

    Residues from the engine's transform feed the engine's reconstruction,
    and the integers must equal ``exact_compound_poly``'s bit for bit. This
    is the full pipeline with no Python kernel in the loop, which neither
    per-kernel test gives on its own.
    """
    ntt = pytest.importorskip('aggregate.ntt')

    rng = random.Random(1000 + seed)
    freq_outcomes = sorted(set(rng.randint(0, 12)
                               for _ in range(rng.randint(1, 5))))
    freq_weights = [rng.randint(1, 9) for _ in freq_outcomes]
    total = sum(freq_weights)
    freq_probs = [Fraction(w, total) for w in freq_weights]
    sev_outcomes = sorted(set(rng.randint(1, 9)
                              for _ in range(rng.randint(1, 4))))
    sev_weights = [rng.randint(1, 9) for _ in sev_outcomes]
    sev_total = sum(sev_weights)
    sev_probs = [Fraction(w, sev_total) for w in sev_weights]

    expected, denominator, _ = ntt.exact_compound_poly(
        freq_outcomes, freq_probs, sev_outcomes, sev_probs)

    (sev_index, big_q, freq_runs, ds, kmax, support, length,
     _) = _transform_inputs(freq_outcomes, freq_probs, sev_outcomes,
                            sev_probs)
    pairs = ntt.ntt_primes(length, denominator)
    primes = [p for p, _ in pairs]
    reconstructor = aggregate_ntt.Reconstructor(primes, support + 1)
    for prime, root in pairs:
        reconstructor.push(_engine_residues(
            prime, root, length, support, sev_index, big_q, freq_runs,
            ds, kmax))
    limbs, modulus_bytes = reconstructor.finish()
    modulus = int.from_bytes(modulus_bytes, 'little')
    numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]

    assert modulus > denominator
    assert numerators == expected
    assert sum(numerators) == denominator


def _wide_pairs(length, bound):
    """63-bit ``(prime, root)`` pairs with product above ``bound``.

    The width-64 configuration ``aggregate.ntt.ntt_primes`` supplies once it
    grows a width parameter: ``p = c * length + 1`` scanned down from
    ``2**63``, primality by the library's deterministic Miller-Rabin. The
    root needs no factoring: for a non-residue witness ``x``,
    ``r = x**((p - 1) / length)`` has order dividing ``length`` (a power of
    two), and order exactly ``length`` iff ``r**(length / 2) != 1``.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    pairs = []
    product = 1
    c = (2 ** 63 - 2) // length
    while product <= bound:
        p = c * length + 1
        c -= 1
        if p >= 2 ** 63 or not ntt.is_prime_u64(p):
            continue
        for x in range(2, 50):
            r = pow(x, (p - 1) // length, p)
            if r != 1 and pow(r, length // 2, p) != 1:
                pairs.append((p, r))
                product *= p
                break
    return pairs


@pytest.mark.parametrize('seed', range(4))
def test_transform_end_to_end_matches_oracle_64bit(seed):
    """The whole pipeline at 64-bit primes, the production configuration.

    Same integers as the oracle bit for bit, from a different prime set than
    the pure-Python path can use: the reconstructed coefficients are
    determined by the mathematics, not by which primes carried them, which
    is exactly why conformance compares answers and never audit records.
    """
    ntt = pytest.importorskip('aggregate.ntt')

    rng = random.Random(2000 + seed)
    freq_outcomes = sorted(set(rng.randint(0, 12)
                               for _ in range(rng.randint(1, 5))))
    freq_weights = [rng.randint(1, 9) for _ in freq_outcomes]
    total = sum(freq_weights)
    freq_probs = [Fraction(w, total) for w in freq_weights]
    sev_outcomes = sorted(set(rng.randint(1, 9)
                              for _ in range(rng.randint(1, 4))))
    sev_weights = [rng.randint(1, 9) for _ in sev_outcomes]
    sev_total = sum(sev_weights)
    sev_probs = [Fraction(w, sev_total) for w in sev_weights]

    expected, denominator, _ = ntt.exact_compound_poly(
        freq_outcomes, freq_probs, sev_outcomes, sev_probs)

    (sev_index, big_q, freq_runs, ds, kmax, support, length,
     _) = _transform_inputs(freq_outcomes, freq_probs, sev_outcomes,
                            sev_probs)
    pairs = _wide_pairs(length, denominator)
    assert all(p > 2 ** 32 for p, _ in pairs), 'the width is the point'
    primes = [p for p, _ in pairs]
    reconstructor = aggregate_ntt.Reconstructor(primes, support + 1)
    for prime, root in pairs:
        reconstructor.push(_engine_residues(
            prime, root, length, support, sev_index, big_q, freq_runs,
            ds, kmax))
    limbs, modulus_bytes = reconstructor.finish()
    modulus = int.from_bytes(modulus_bytes, 'little')
    numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]

    assert modulus > denominator
    assert numerators == expected
    assert sum(numerators) == denominator


def test_residues_mod_prime_refusals():
    """Contract violations are loud refusals, never silent folding."""
    ntt = pytest.importorskip('aggregate.ntt')
    (prime, root), = ntt.ntt_primes(16, 2)
    sev_index = np.array([1, 2], dtype=np.int64)
    sev_weights = np.array([1, 1], dtype=np.uint64)
    good = dict(length=16, support=4, sev_index=sev_index,
                sev_weights=sev_weights, freq_runs=[(1, 2, 1)], ds=2, kmax=2)

    def call(**overrides):
        arguments = {'prime': prime, 'root': root, **good, **overrides}
        return aggregate_ntt.residues_mod_prime(*arguments.values())

    with pytest.raises(ValueError, match='odd prime below 2\\*\\*63'):
        call(prime=2 ** 63 + 1)
    with pytest.raises(ValueError, match='odd prime'):
        call(prime=100)
    with pytest.raises(ValueError, match='not a primitive'):
        call(root=1)
    with pytest.raises(ValueError, match='power of two'):
        call(length=12)
    with pytest.raises(ValueError, match='does not fit transform length'):
        call(support=16)
    with pytest.raises(ValueError, match='ds .* not reduced'):
        call(ds=prime)
    with pytest.raises(ValueError, match='severity weight .* not reduced'):
        call(sev_weights=np.array([1, prime], dtype=np.uint64))
    with pytest.raises(ValueError, match='frequency weight .* not reduced'):
        call(freq_runs=[(1, 2, prime)])
    with pytest.raises(ValueError, match='end at 2 but kmax is 3'):
        call(kmax=3)
    with pytest.raises(ValueError, match='outside the transform length'):
        call(sev_index=np.array([1, 16], dtype=np.int64))


def test_residues_mod_prime_aliasing_check_is_decisive():
    """A support smaller than the truth raises, mirroring the Python check.

    Understating the support leaves true coefficients beyond it, which the
    engine must refuse exactly as ``_residues_mod_prime`` does, because a
    truncated answer is a wrong answer.
    """
    ntt = pytest.importorskip('aggregate.ntt')
    freq_outcomes, freq_probs = [1, 2, 3, 4], [Fraction(1, 4)] * 4
    sev_outcomes, sev_probs = [1, 2, 3, 4], [Fraction(1, 4)] * 4
    (sev_index, big_q, freq_runs, ds, kmax, support, length,
     denominator) = _transform_inputs(freq_outcomes, freq_probs,
                                      sev_outcomes, sev_probs)
    prime, root = ntt.ntt_primes(length, denominator)[0]
    with pytest.raises(AssertionError, match='beyond the support'):
        _engine_residues(prime, root, length, support - 1, sev_index,
                         big_q, freq_runs, ds, kmax)
