import os
from bisect import bisect_left

def make_meter_dir(meter_parent_dir, meter_id, filename):
    meter_dir = os.path.join(meter_parent_dir, meter_id)
    os.makedirs(meter_dir, exist_ok=True)
    meter_fp = os.path.join(meter_dir, filename)
    return meter_dir, meter_fp

def find_bounds(x: int, sorted_values: list) -> tuple[int, int]:
    i = bisect_left(sorted_values, x)
    if i < len(sorted_values) and sorted_values[i] == x:
        return sorted_values[i], sorted_values[i]
    lower = sorted_values[i - 1]
    upper = sorted_values[i]
    return lower, upper
