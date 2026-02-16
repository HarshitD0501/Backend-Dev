"""
Approach 4: Hybrid (3 Processes × 3 Threads each = 9 parallel workers)
- 3 processes banate hain, har process ke andar 3 threads
- Har process ka apna GIL → CPU truly parallel
- Har process ke threads → I/O wait mein switch
- Total time ≈ 4 seconds
- ⚠️ Powerful but overkill for simple I/O tasks
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def square_number(number):
    """Simulates an I/O-bound task (e.g., API call)"""
    time.sleep(2)
    return f"Square of {number} = {number * number} (PID: {os.getpid()})"


def process_chunk(chunk):
    """Har process apne andar 3 threads banata hai"""
    with ThreadPoolExecutor(max_workers=3) as thread_executor:
        results = list(thread_executor.map(square_number, chunk))
    return results


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    print("=" * 50)
    print("APPROACH 4: Hybrid (3 Processes × 3 Threads)")
    print("=" * 50)

    start = time.time()

    # Split items into chunks — one chunk per process
    chunk_size = 5  # ceil(14/3) = 5 items per process
    chunks = [numbers[i:i + chunk_size] for i in range(0, len(numbers), chunk_size)]
    # chunks = [[1-5], [6-10], [11-14]]

    with ProcessPoolExecutor(max_workers=3) as process_executor:
        all_results = list(process_executor.map(process_chunk, chunks))

    # Flatten and print results
    for batch in all_results:
        for result in batch:
            print(result)

    elapsed = time.time() - start
    print(f"\n[TIME] Total Time: {elapsed:.2f} seconds")
    print(f"[INFO] Items processed: {len(numbers)}")
    print(f"[TIP]  3 processes x 3 threads = 9 parallel workers")
    print(f"[TIP]  Notice different PIDs -- proves separate processes!")
