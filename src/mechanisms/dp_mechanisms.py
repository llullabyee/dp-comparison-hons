
'''
    dp mechanism stuff
'''
import random
import math
from scipy.stats import norm
import numpy as np

from src.utils.config import load_config

config = load_config()
privacy_cfg = config["privacy"]


def gai_discretise(u: int,v: int,x: int) -> int: ### change u to lower and v to upper, use more verbose code?
    intervals = [u,v]

    u_prob = (v-x)/(v-u)
    v_prob = (x-u)/(v-u)
    probs = [u_prob,v_prob]

    result = random.choices(intervals,weights=probs,k=1)
    return result[0]


def gai_perturb(x_i: int, epsilon: float, interval_list: list) -> int:
    len_list = len(interval_list)
    x_position = interval_list.index(x_i)

    p_prob = (math.exp(epsilon))/(len_list-1+math.exp(epsilon))
    q_prob = (1)/(len_list-1+math.exp(epsilon))
    probs = [q_prob] * len_list
    probs[x_position] = p_prob
    print(probs)

    result = random.choices(interval_list,weights=probs,k=1)
    return result[0]


def dplm_noise(curr_reading: int, norm_prob: float, max_error: float, peak: int, num_readings: int = 1, mean: float = 0):
    '''
    Set num_readings to 1 for pointwise privacy. Additionally, mean is included and set to 0 by default.
    This seems to be what Ul Hassan et al. do in their tests. Standard deviation confirmed to be calculated
    via a two tailed approach — the paper they reference from Barbossa et al. states this directly (seriously
    why is there a standard deviation that does not line up with either of the variances in the example guys come on)
    '''
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


interval_list = [5,10,15,20,25,30,35,40]

print(gai_discretise(10,15,12))
print(gai_perturb(x_i=20,epsilon=privacy_cfg["epsilon"],interval_list=interval_list))

dplm_noise(1500,0.98,0.1,3000)
