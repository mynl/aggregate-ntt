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

``[Rust-Reconstruction]`` has landed, so its tests run for real. The
transform kernel still raises ``NotImplementedError`` and its test stays an
expected failure until ``[Rust-Transform]`` lands.
"""

import random
from fractions import Fraction

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


@pytest.mark.xfail(reason='needs [Rust-Reconstruction] here and the '
                          'ntt_engine keyword from the integration plan in '
                          'aggregate; not strict, since it fails two ways',
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

@pytest.mark.xfail(reason='[Rust-Transform] kernel not implemented yet',
                   raises=NotImplementedError, strict=True)
def test_residues_mod_prime_matches_python():
    """One prime's residues, against ``aggregate.ntt._residues_mod_prime``.

    Filled in with the real argument shapes when ``[Rust-Transform]``
    starts; the reference call is private on purpose, since this is the one
    place a private name is legitimately the comparison target.
    """
    raise NotImplementedError(
        '[Rust-Transform] kernel not implemented yet')
