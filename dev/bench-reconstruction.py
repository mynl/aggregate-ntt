"""
Benchmark [Rust-Reconstruction] on the headline 1000 x 1000 shape.

Reproduces the acceptance measurement in ``dev/plan-rust-engine.md``: the
reconstruction phase of the 1000 x 1000 uniform dice of dice case, which on
the Python side at ``aggregate`` 1.2.0a5 took 207.3 s serial and 187.0 s at
four workers. Acceptance is at or below 25 s, with the bulk conversion of the
limb array to Python integers measured separately against its own 3 s budget.

Residues are random per prime rather than taken from a real transform, which
exercises the reconstruction identically (its cost depends only on the shape)
and keeps the benchmark independent of the transform phase. A spot check at
the end verifies a sample of reconstructed values against the pushed
residues, so a wrong answer cannot post a fast time.

Run against a release build only (``maturin develop --release`` with
``RUSTFLAGS=-C target-cpu=native``), per [Rust-Build-Local]:

    uv run python dev/bench-reconstruction.py
"""

import ctypes
import ctypes.wintypes
import sys
import time

import numpy as np


def peak_working_set_bytes():
    """Peak working set of this process, Windows only, else ``None``.

    Measures the plan's memory acceptance (peak no worse than the Python
    path's, roughly twice the answer) without a psutil dependency.
    """
    if sys.platform != 'win32':
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [('cb', ctypes.wintypes.DWORD),
                    ('PageFaultCount', ctypes.wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL('kernel32')
    get_info = kernel32.K32GetProcessMemoryInfo
    get_info.argtypes = [ctypes.wintypes.HANDLE,
                         ctypes.POINTER(ProcessMemoryCounters),
                         ctypes.wintypes.DWORD]
    get_info.restype = ctypes.wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.wintypes.HANDLE(-1)  # GetCurrentProcess pseudo-handle
    if not get_info(handle, ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize

import aggregate_ntt
from aggregate.ntt import ntt_primes

N_COEFFICIENTS = 1_000_001
BOUND_BITS = 9_976
SPOT_CHECKS = 5

print(f'engine {aggregate_ntt.__version__}, '
      f'protocol {aggregate_ntt.PROTOCOL_VERSION}')
pairs = ntt_primes(2 ** 20, 1 << BOUND_BITS)
primes = [p for p, _ in pairs]
print(f'{len(primes)} primes, {N_COEFFICIENTS:,} coefficients, '
      f'bound {BOUND_BITS:,} bits')

rng = np.random.default_rng(2026)
spots = rng.integers(0, N_COEFFICIENTS, size=SPOT_CHECKS)
kept = {}

reconstructor = aggregate_ntt.Reconstructor(primes, N_COEFFICIENTS)
t_push = 0.0
for prime in primes:
    residues = rng.integers(0, prime, size=N_COEFFICIENTS, dtype=np.uint64)
    kept[prime] = residues[spots].copy()
    t0 = time.perf_counter()
    reconstructor.push(residues)
    t_push += time.perf_counter() - t0

t0 = time.perf_counter()
limbs, modulus_bytes = reconstructor.finish()
t_finish = time.perf_counter() - t0

t0 = time.perf_counter()
modulus = int.from_bytes(modulus_bytes, 'little')
numerators = [int.from_bytes(row.tobytes(), 'little') for row in limbs]
t_convert = time.perf_counter() - t0

for j, spot in enumerate(spots):
    value = numerators[spot]
    assert 0 <= value < modulus
    for prime in primes:
        assert value % prime == int(kept[prime][j]), (spot, prime)
print(f'spot checks OK at {SPOT_CHECKS} coefficients x {len(primes)} primes')

print(f'limb array {limbs.shape}, {limbs.nbytes / 2**30:.2f} GiB, '
      f'modulus {modulus.bit_length():,} bits')
print(f'push (merge stack)   {t_push:8.2f} s')
print(f'finish (drain+limbs) {t_finish:8.2f} s')
print(f'reconstruction total {t_push + t_finish:8.2f} s   '
      f'(acceptance: <= 25 s; Python 187.0 s at 4 workers)')
print(f'bulk int conversion  {t_convert:8.2f} s   (budget: < 3 s)')
peak = peak_working_set_bytes()
if peak is not None:
    print(f'peak working set     {peak / 2**30:8.2f} GiB  '
          f'(answer is {limbs.nbytes / 2**30:.2f} GiB as limbs; acceptance: '
          f'about twice the answer, like the Python path)')
