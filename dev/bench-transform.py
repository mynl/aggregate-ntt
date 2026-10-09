"""
Benchmark [Rust-Transform] on the headline 1000 x 1000 case.

Reproduces the acceptance measurements in ``dev/plan-rust-engine.md``: the
transform phase of the 1000 x 1000 uniform dice of dice case (support 10^6,
length 2^20), which on the Python side at ``aggregate`` 1.2.0a5 took 104.8 s
serial and 709.3 thread-seconds at four workers. Acceptance is at or below
10 s wall clock using all cores, with the end-to-end case (transform,
reconstruction, conversion, the denominator check) in low tens of seconds.

Pass ``--width 64`` to run with 64-bit primes (generated here, just below
``2**63``), which halves the prime count for the same coefficient bound and
is the production configuration once ``aggregate`` selects wide primes. The
default ``--width 32`` uses ``aggregate.ntt.ntt_primes``, the configuration
directly comparable to the pure-Python engine.

Run against a release build only (``maturin develop --release`` with
``RUSTFLAGS=-C target-cpu=native``), per [Rust-Build-Local]:

    uv run python dev/bench-transform.py [--width 32|64] [--workers N]
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import aggregate_ntt
from aggregate.ntt import is_prime_u64, ntt_primes

N = 1000
KMAX = N
DS = N
SUPPORT = N * N
LENGTH = 1 << SUPPORT.bit_length()
DENOMINATOR = N * N ** N
SEV_INDEX = np.array(range(1, N + 1), dtype=np.int64)
FREQ_RUNS = [(1, N, 1)]


def primes_64(length, bound):
    """64-bit NTT primes just below ``2**63`` with product above ``bound``.

    Scans ``p = c * length + 1`` downward from ``2**63`` with the library's
    deterministic Miller-Rabin. The root is found without factoring
    ``p - 1``: for random ``x``, ``r = x**((p - 1) / length)`` has order
    dividing ``length`` (a power of two), and order exactly ``length`` iff
    ``r**(length / 2) != 1``, which half of all ``x`` satisfy. This is the
    shape ``aggregate.ntt.ntt_primes`` grows a width parameter for; until it
    does, the benchmark selects its own.
    """
    pairs = []
    product = 1
    c = (2 ** 63 - 2) // length
    while product <= bound:
        p = c * length + 1
        c -= 1
        if p >= 2 ** 63 or not is_prime_u64(p):
            continue
        for x in range(2, 100):
            r = pow(x, (p - 1) // length, p)
            if r != 1 and pow(r, length // 2, p) != 1:
                pairs.append((p, r))
                product *= p
                break
        else:
            raise AssertionError(f'no {length}-th root found mod {p}')
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, choices=(32, 64), default=32)
    parser.add_argument('--workers', type=int,
                        default=min(16, os.cpu_count() or 1))
    arguments = parser.parse_args()

    print(f'engine {aggregate_ntt.__version__}, '
          f'protocol {aggregate_ntt.PROTOCOL_VERSION}, '
          f'{arguments.width}-bit primes, {arguments.workers} workers')
    t0 = time.perf_counter()
    if arguments.width == 32:
        pairs = ntt_primes(LENGTH, DENOMINATOR)
    else:
        pairs = primes_64(LENGTH, DENOMINATOR)
    print(f'{len(pairs)} primes in {time.perf_counter() - t0:.1f} s, '
          f'support {SUPPORT:,}, length 2**{LENGTH.bit_length() - 1}, '
          f'bound {DENOMINATOR.bit_length():,} bits')

    def run(pair):
        prime, root = pair
        # The uniform case's cleared weights are all one, so the caller-side
        # reduction the seam asks for is the identity here.
        weights = np.ones(N, dtype=np.uint64)
        return aggregate_ntt.residues_mod_prime(
            prime, root, LENGTH, SUPPORT, SEV_INDEX, weights,
            FREQ_RUNS, DS % prime, KMAX)

    # Transform phase alone, nothing else competing for the cores: this is
    # the number the plan's 10 s acceptance names. Results are discarded as
    # they arrive, so peak memory stays a few residue vectors.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for head in pool.map(run, pairs):
            pass
    t_transform_alone = time.perf_counter() - t0

    # The deployed shape: the main thread pushes into the reconstruction
    # (which fans out on rayon) while the pool keeps transforming, so the
    # two phases overlap and compete for cores exactly as they would under
    # aggregate. Phase splits here are attribution, not isolation.
    primes = [p for p, _ in pairs]
    reconstructor = aggregate_ntt.Reconstructor(primes, SUPPORT + 1)
    t0 = time.perf_counter()
    t_crt = 0.0
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for head in pool.map(run, pairs):
            t1 = time.perf_counter()
            reconstructor.push(head)
            t_crt += time.perf_counter() - t1
    t_transform = time.perf_counter() - t0 - t_crt
    t1 = time.perf_counter()
    limbs, modulus_bytes = reconstructor.finish()
    t_crt += time.perf_counter() - t1

    t1 = time.perf_counter()
    numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]
    t_convert = time.perf_counter() - t1

    modulus = int.from_bytes(modulus_bytes, 'little')
    assert modulus > DENOMINATOR
    assert sum(numerators) == DENOMINATOR, 'denominator check failed'
    print('denominator check OK: coefficients sum exactly to D')
    total = t_transform + t_crt + t_convert
    print(f'transform alone              {t_transform_alone:8.2f} s   '
          f'(acceptance: <= 10 s; Python 104.8 s serial)')
    print(f'overlapped: main-thread rest {t_transform:8.2f} s')
    print(f'overlapped: reconstruction   {t_crt:8.2f} s')
    print(f'bulk int conversion          {t_convert:8.2f} s')
    print(f'end to end                   {total:8.2f} s   '
          f'(Python 212.3 s at 8 workers)')


if __name__ == '__main__':
    main()
