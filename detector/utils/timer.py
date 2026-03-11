import time
import logging


def banchmark_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper
