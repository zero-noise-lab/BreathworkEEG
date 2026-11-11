"""Compile and expose the exact LZ76 phrase counter used by NeuroKit2."""

from __future__ import annotations

import ctypes
import fcntl
import os
import hashlib
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fast_lz76.c"
CACHE = Path(tempfile.gettempdir()) / f"eeg-methods-lz76-{os.getuid()}"
BUILD_ID = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16] + "-" + platform.machine()
LIBRARY = CACHE / f"_fast_lz76-{BUILD_ID}.so"
LOCK = CACHE / f"lz76-{BUILD_ID}.lock"


def build(force: bool = False) -> Path:
    """Build once, using a file lock to avoid concurrent compiler writes."""
    compiler = shlex.split(os.environ.get("CC", "gcc"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(f"C compiler is unavailable: {compiler!r}")
    CACHE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        stale = not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime
        if force or stale:
            temporary = CACHE / f"._fast_lz76.{os.getpid()}.so"
            command = [
                *compiler,
                "-O3",
                "-std=c99",
                "-shared",
                "-fPIC",
                str(SOURCE),
                "-o",
                str(temporary),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.replace(temporary, LIBRARY)
            finally:
                if temporary.exists():
                    temporary.unlink()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return LIBRARY


_lib = ctypes.CDLL(str(build()))
_lib.lz76_count.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.lz76_count.restype = ctypes.c_uint64


def count(symbols: np.ndarray | list[int]) -> int:
    """Return the LZ76 phrase count for one binary sequence."""
    raw = np.asarray(symbols)
    if raw.ndim != 1 or not np.logical_or(raw == 0, raw == 1).all():
        raise ValueError("LZ76 input must be a 1-D binary sequence")
    sequence = np.ascontiguousarray(raw, dtype=np.uint8)
    if sequence.ndim != 1 or sequence.size < 2:
        raise ValueError("LZ76 input must be a 1-D sequence of length >=2")
    if not np.logical_or(sequence == 0, sequence == 1).all():
        raise ValueError("LZ76 input must contain only 0 and 1")
    pointer = sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    value = int(_lib.lz76_count(pointer, sequence.size))
    if value < 1:
        raise RuntimeError("Compiled LZ76 counter returned an invalid value")
    return value


def normalized(symbols: np.ndarray | list[int]) -> tuple[float, int]:
    """Return c(n) log2(n)/n and the underlying phrase count."""
    sequence = np.asarray(symbols)
    phrase_count = count(sequence)
    value = phrase_count * np.log2(sequence.size) / sequence.size
    return float(value), phrase_count


if __name__ == "__main__":
    # Small deterministic check of the compiled counter.
    expected = {
        (0, 0, 0): 2,
        (0, 1, 0, 1): 3,
        (0, 1, 1, 0, 1, 0, 0, 1): 5,
    }
    for sequence, target in expected.items():
        observed = count(sequence)
        if observed != target:
            raise SystemExit(f"LZ76 self-test failed: {sequence}: {observed} != {target}")
    print(f"Compiled LZ76 self-test passed: {LIBRARY}")
