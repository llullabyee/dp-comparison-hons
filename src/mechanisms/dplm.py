'''
Set num_readings to 1 for pointwise privacy. Additionally, mean is included and set to 0 by default.
This seems to be what Ul Hassan et al. do in their tests. Standard deviation confirmed to be calculated
via a two tailed approach — the paper they reference from Barbossa et al. states this directly (seriously
why is there a standard deviation that does not line up with either of the variances in the example guys come on)
'''

import math
import numpy as np
from pandas import DataFrame
from scipy.stats import norm


def dplm_noise(curr_reading: int, norm_prob: float, max_error: float, peak: int, num_readings: int = 1, mean: float = 0):
    error_Wh = max_error * curr_reading
    z_score = norm.interval(norm_prob)[1]
    std_dev = error_Wh / z_score
    variance = std_dev ** 2
    laplace_scale = math.sqrt(variance / (2*num_readings))
    noise = np.random.laplace(loc=0, scale=laplace_scale)
    print(error_Wh,z_score,std_dev,variance,laplace_scale,noise)
    if noise > peak:
        return peak
    return noise

def dplm_mechanism(df: DataFrame, norm_prob: float, max_error: float,
                   peak: int, excess: int = 0) -> int:
    """
    Apply DPLM noise to every row of `df`, with peak clipping and
    excess carry-over into the next reading.

    - Noise scale is computed from the raw reading only.
    - Carried `excess` from the previous row is added AFTER noise.
    - If power + noise + excess >= peak, output is clipped to peak
      and the overflow is carried into the next reading.
    """
    noisy_values = []

    for _, row in df.iterrows():
        reading = row['power']

        noise = dplm_noise(curr_reading=reading,
                           norm_prob=norm_prob,
                           max_error=max_error,
                           peak=peak)

        value = reading + noise + excess
        excess = 0

        if value >= peak:
            excess = value - peak   # carry the overflow forward
            value = peak

        noisy_values.append(value)

    df["power_noisy"] = noisy_values
    return excess
