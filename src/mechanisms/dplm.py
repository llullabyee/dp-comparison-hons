'''
Set num_readings to 1 for pointwise privacy. Additionally, mean is included and set to 0 by default.
This seems to be what Ul Hassan et al. do in their tests. Standard deviation confirmed to be calculated
via a two tailed approach — the paper they reference from Barbossa et al. states this directly (seriously
why is there a standard deviation that does not line up with either of the variances in the example guys come on)
'''

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.config import DOMAIN_MAX


def dplm_noise(curr_reading: int, norm_prob: float, max_error: float, peak: int, num_readings: int = 1, mean: float = 0):
    error_Wh = max_error * curr_reading
    z_score = norm.interval(norm_prob)[1]
    std_dev = error_Wh / z_score
    variance = std_dev ** 2
    laplace_scale = math.sqrt(variance / (2 * num_readings))
    noise = np.random.laplace(loc=0, scale=laplace_scale)
    if noise > peak:
        return peak
    return noise


def dplm_mechanism(df: pd.DataFrame, norm_prob: float, epsilon: float, sensitivity: float,
                   peak: int, num_readings: int = 1) -> pd.Series:
    """
    Apply DPLM noise to every row of `df`, with peak clipping and
    excess carry-over into the next reading.

    - Noise scale is computed from the raw reading only.
    - Carried `excess` from the previous row is added AFTER noise.
    - If power + noise + excess >= peak, output is clipped to peak
      and the overflow is carried into the next reading.

    Noise generation and per-reading clipping are fully vectorised; only the
    excess carry-over (an inherently sequential recurrence) is looped, and
    that loop now does plain arithmetic with no per-row scipy/print overhead.
    """
    readings = df["mains"].to_numpy(dtype=float)

    # z_score / scale factor are constant across rows -> compute once.
    z_score = norm.interval(norm_prob)[1]
    print(z_score)
    laplace_scale = sensitivity / epsilon
    max_error_percent = (sensitivity * z_score * math.sqrt(2 * num_readings)) / (epsilon * peak)
    max_error_watts = max_error_percent * peak
    print(f" Max Error in W: {max_error_watts} | Max Error Percentage: {max_error_percent}")
    # scale_factor = (max_error / z_score) / math.sqrt(2 * num_readings)
    print(f"Scale: {laplace_scale}")

    noise = np.random.laplace(loc=0.0, scale=laplace_scale, size=readings.shape)
    base = readings + noise  # reading + (clipped) noise, before excess carry

    # Sequential excess carry-over recurrence (cannot be vectorised due to the
    # reset-on-underflow clamp), but this is now cheap scalar arithmetic.
    noisy_values = np.empty_like(base)
    carry = 0
    for i in range(base.shape[0]):
        if np.isnan(base[i]):
            noisy_values[i] = np.nan
            carry = 0.0                   # don't carry across a gap
            continue
        value = base[i] + carry
        if value >= peak:
            carry = value - peak
            value = peak
        else:
            carry = 0.0
        noisy_values[i] = value
    return pd.Series(noisy_values, name='dplm_noise')
