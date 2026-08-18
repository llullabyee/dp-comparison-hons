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


def dplm_mechanism(df: pd.DataFrame, norm_prob: float, max_error: float,
                   peak: int, excess: int = 0, num_readings: int = 1) -> tuple[pd.Series, float]:
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
    df = df.copy()
    readings = df["power"].to_numpy(dtype=float)

    # z_score / scale factor are constant across rows -> compute once.
    z_score = norm.interval(norm_prob)[1]
    scale_factor = (max_error / z_score) / math.sqrt(2 * num_readings)
    laplace_scale = np.abs(readings) * scale_factor

    noise = np.random.laplace(loc=0.0, scale=laplace_scale)
    np.minimum(noise, peak, out=noise)  # per-reading noise clip (matches dplm_noise)

    base = readings + noise  # reading + (clipped) noise, before excess carry

    # Sequential excess carry-over recurrence (cannot be vectorised due to the
    # reset-on-underflow clamp), but this is now cheap scalar arithmetic.
    noisy_values = np.empty_like(base)
    carry = float(excess)
    for i in range(base.shape[0]):
        value = base[i] + carry
        if value >= peak:
            carry = value - peak
            value = peak
        else:
            carry = 0.0
        noisy_values[i] = value

    df["dplm_noise"] = noisy_values
    return df['dplm_noise'], carry
