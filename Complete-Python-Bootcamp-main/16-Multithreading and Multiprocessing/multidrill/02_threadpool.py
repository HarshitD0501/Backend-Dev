"""
Approach 2: ThreadPoolExecutor (14 threads)
- Sab items ek saath threads mein chalte hain
- GIL release hota hai during time.sleep (I/O wait)
- Total time ≈ 2 seconds (sab parallel)
- ✅✅ BEST for I/O-bound tasks!
"""

import time
from concurrent.futures import ThreadPoolExecutor


def square_number(number):
    """Simulates an I/O-bound task (e.g., API call)"""
    time.sleep(2)
    return f"Square of {number} = {number * number}"


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    print("=" * 50)
    print("APPROACH 2: ThreadPoolExecutor (14 threads)")
    print("=" * 50)

    start = time.time()

    with ThreadPoolExecutor(max_workers=14) as executor:
        results = executor.map(square_number, numbers)

    for result in results:
        print(result)

    elapsed = time.time() - start
    print(f"\n[TIME] Total Time: {elapsed:.2f} seconds")
    print(f"[INFO] Items processed: {len(numbers)}")
    print(f"[TIP]  All 14 threads ran simultaneously, GIL released during sleep")
