'''
Set num_readings to 1 for pointwise privacy. Additionally, mean is included and set to 0 by default.
This seems to be what Ul Hassan et al. do in their tests. Standard deviation confirmed to be calculated
via a two tailed approach — the paper they reference from Barbossa et al. states this directly (seriously
why is there a standard deviation that does not line up with either of the variances in the example guys come on)
'''

import math
import numpy as np
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

def dplm_mechanism():
