"""
Approach 1: Sequential (No Parallelism)
- Ek ek karke sab items process hote hain
- Total time = 14 items × 2 sec = ~28 seconds
"""

import time


def square_number(number):
    """Simulates an I/O-bound task (e.g., API call)"""
    time.sleep(2)
    return f"Square of {number} = {number * number}"


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    print("=" * 50)
    print("APPROACH 1: Sequential (No Parallelism)")
    print("=" * 50)

    start = time.time()

    results = []
    for num in numbers:
        result = square_number(num)
        results.append(result)
        print(result)

    elapsed = time.time() - start
    print(f"\n[TIME] Total Time: {elapsed:.2f} seconds")
    print(f"[INFO] Items processed: {len(numbers)}")
    print(f"[TIP]  Each item waited 2 sec, all ran one-by-one")
