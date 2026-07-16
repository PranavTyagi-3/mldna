"""Phase 0 spike: streamed SHA-256 throughput on a multi-GB file (ROADMAP.md
Phase 0, item 7).

CLAUDE.md's style rules require streamed SHA-256 for large files — never reading a
multi-GB model into memory. This spike generates a synthetic file, hashes it in
fixed-size chunks, and reports throughput and peak-ish memory discipline (constant
buffer size regardless of file size), then deletes the file.

Fully local — no Azure needed.

Run:  python spikes/07_checksum_throughput.py [--size-gb 2] [--workdir PATH]
"""

import argparse
import hashlib
import os
import tempfile
import time
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB — constant regardless of file size


def generate_file(path: Path, size_bytes: int) -> None:
    written = 0
    block = os.urandom(CHUNK_SIZE)
    with open(path, "wb") as f:
        while written < size_bytes:
            to_write = min(CHUNK_SIZE, size_bytes - written)
            f.write(block[:to_write])
            written += to_write


def streamed_sha256(path: Path) -> tuple[str, float]:
    h = hashlib.sha256()
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    elapsed = time.perf_counter() - t0
    return h.hexdigest(), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gb", type=float, default=2.0)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    size_bytes = int(args.size_gb * 1024 * 1024 * 1024)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="mlops-spike07-"))
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "synthetic_model.bin"

    try:
        print(f"Generating {args.size_gb:.2f} GB synthetic file at {target} ...")
        t0 = time.perf_counter()
        generate_file(target, size_bytes)
        gen_elapsed = time.perf_counter() - t0
        actual_size = target.stat().st_size
        print(f"Generated {actual_size / (1024**3):.2f} GB in {gen_elapsed:.2f}s")

        digest, elapsed = streamed_sha256(target)
        throughput_mb_s = (actual_size / (1024 * 1024)) / elapsed if elapsed > 0 else float("inf")
        print(f"SHA-256: {digest}")
        print(f"Hashed {actual_size / (1024**3):.2f} GB in {elapsed:.2f}s -> {throughput_mb_s:.1f} MB/s")
        print(f"Chunk size used throughout: {CHUNK_SIZE // (1024 * 1024)} MiB (constant, streamed)")

    finally:
        if target.exists():
            target.unlink()
            print(f"Deleted synthetic file: {target}")
        try:
            workdir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
