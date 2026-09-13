"""
Set num_readings to 1 for pointwise privacy. Additionally, mean is included and set to 0 by default.
This seems to be what Ul Hassan et al. do in their tests. Standard deviation confirmed to be calculated
via a two tailed approach — the paper they reference from Barbossa et al. states this directly (seriously
why is there a standard deviation that does not line up with either of the variances in the example guys come on)
"""

import numpy as np


def dplm_mechanism(
    readings: np.ndarray,
    norm_prob: float,
    epsilon: float,
    sensitivity: float,
    peak: int,
    num_readings: int = 1,
    excess: float = 0,
) -> tuple[np.ndarray, float]:
    """
    Apply DPLM noise to every row of a numpy array of readings, with peak
    clipping and excess carry-over into the next reading.

    - Noise scale is computed from the raw reading only.
    - Carried `excess` from the previous row is added AFTER noise.
    - If power + noise + excess >= peak, output is clipped to peak
      and the overflow is carried into the next reading.

    Noise generation and per-reading clipping are fully vectorised; only the
    excess carry-over (an inherently sequential recurrence) is looped, and
    that loop now does plain arithmetic with no per-row scipy/print overhead.

    Notably, the code here has been refactored using Eibl and Engel's ([17] in
    Ul Hassan's paper) definition of the laplace scale. This is largely due to
    a potential misunderstanding by the original authors, in which they ignore
    the use of epsilon and sensitivity in favour of utilising peak and max max
    error rate, resulting in erroneous sensitivity and/or epsilon values.
    """
    # z_score = norm.interval(norm_prob)[1]
    # max_error_percent = (sensitivity * z_score * np.sqrt(2 * num_readings)) / (
    #     epsilon * peak
    # )
    # max_error_watts = max_error_percent * peak

    laplace_scale = sensitivity / epsilon
    print(f"Scale: {laplace_scale}")

    noise = np.random.laplace(loc=0.0, scale=laplace_scale, size=readings.shape)
    base = readings + noise  # reading + (clipped) noise, before excess carry

    # Sequential excess carry-over recurrence (cannot be vectorised due to the
    # reset-on-underflow clamp), but this is now cheap scalar arithmetic.
    noisy_values = np.empty_like(base)
    carry = excess
    for i in range(base.shape[0]):
        if np.isnan(base[i]):
            noisy_values[i] = np.nan
            carry = 0.0  # don't carry across a gap
            continue
        value = base[i] + carry
        if value >= peak:
            carry = value - peak
            value = peak
        else:
            carry = 0.0
        noisy_values[i] = value
    return noisy_values, carry
