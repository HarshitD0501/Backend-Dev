"""
Approach 3: ProcessPoolExecutor (3 processes)
- 3 alag processes chalte hain, har ek apni memory + GIL
- Items batches mein process hote hain: ceil(14/3) = 5 batches
- Total time ≈ 10 seconds
- ✅ Works, but overkill for I/O-bound tasks
"""

import time
from concurrent.futures import ProcessPoolExecutor


def square_number(number):
    """Simulates an I/O-bound task (e.g., API call)"""
    time.sleep(2)
    return f"Square of {number} = {number * number}"


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    print("=" * 50)
    print("APPROACH 3: ProcessPoolExecutor (3 processes)")
    print("=" * 50)

    start = time.time()

    with ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(square_number, numbers)

    for result in results:
        print(result)

    elapsed = time.time() - start
    print(f"\n[TIME] Total Time: {elapsed:.2f} seconds")
    print(f"[INFO] Items processed: {len(numbers)}")
    print(f"[TIP]  3 processes, ~5 batches of 2 sec each")
