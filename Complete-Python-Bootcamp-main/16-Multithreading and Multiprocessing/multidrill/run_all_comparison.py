"""
Run All 4 Approaches & Compare Time
- Runs each approach one by one
- Prints a final comparison table
"""

import time
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

approaches = [
    ("01_sequential.py", "Sequential (No Parallelism)"),
    ("02_threadpool.py", "ThreadPool (14 threads)"),
    ("03_processpool.py", "ProcessPool (3 processes)"),
    ("04_hybrid.py", "Hybrid (3 proc x 3 threads)"),
]

if __name__ == "__main__":
    print("MULTITHREADING & MULTIPROCESSING DRILL")
    print("=" * 60)
    print(f"Task: Compute square of 14 numbers, each with 2 sec I/O wait")
    print(f"Sequential expected: 14 x 2 = 28 seconds")
    print("=" * 60)

    timings = []

    for script, label in approaches:
        print(f"\n{'_' * 60}")
        print(f">> Running: {label}")
        print(f"{'_' * 60}")

        start = time.time()
        script_path = os.path.join(SCRIPT_DIR, script)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start

        # Print output from the script
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")

        timings.append((label, elapsed))

    # Final comparison table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Approach':<35} {'Time':>10} {'Speedup':>10}")
    print("-" * 60)

    sequential_time = timings[0][1] if timings else 1

    for label, elapsed in timings:
        speedup = sequential_time / elapsed if elapsed > 0 else 0
        print(f"{label:<35} {elapsed:>8.2f}s {speedup:>8.1f}x")

    print("-" * 60)
    print("[TIP] Higher speedup = faster relative to sequential")
    print("[WIN] For I/O-bound tasks, ThreadPool is the winner!")
