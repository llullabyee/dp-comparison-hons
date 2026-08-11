import math
import random
import numpy as np
import pandas as pd


def gai_discretise(u: int, v: int, x: int) -> int:  ### change u to lower and v to upper, use more verbose code?
    if u == v:
        return u
    intervals = [u, v]

    u_prob = (v - x) / (v - u)
    v_prob = (x - u) / (v - u)
    probs = [u_prob, v_prob]

    result = random.choices(intervals, weights=probs, k=1)
    return result[0]


def gai_perturb(x_i: int, epsilon: float, interval_list: list) -> int:
    len_list = len(interval_list)
    x_position = interval_list.index(x_i)

    p_prob = (math.exp(epsilon)) / (len_list - 1 + math.exp(epsilon))
    q_prob = (1) / (len_list - 1 + math.exp(epsilon))
    probs = [q_prob] * len_list
    probs[x_position] = p_prob

    result = random.choices(interval_list, weights=probs, k=1)
    return result[0]


def gai_mechanism(df: pd.DataFrame, interval_list: list, epsilon: float) -> pd.Series:
    """
    Vectorised GAI mechanism.

    Produces the ``perturbed`` column with statistics identical to applying
    ``find_bounds`` -> ``gai_discretise`` -> ``gai_perturb`` row-by-row, but
    computed over the whole column with numpy instead of per-row Python loops.
    """
    intervals = np.asarray(interval_list)
    n_int = intervals.size
    power = df["power"].to_numpy()
    n = power.size

    # ---- find_bounds (vectorised bisect_left) --------------------------------
    # idx = bisect_left(intervals, x); exact hits give lower == upper == x.
    idx = np.searchsorted(intervals, power, side="left")
    exact = (idx < n_int) & (intervals[np.clip(idx, 0, n_int - 1)] == power)
    lower = np.where(exact, intervals[np.clip(idx, 0, n_int - 1)],
                     intervals[np.clip(idx - 1, 0, n_int - 1)])
    upper = np.where(exact, lower, intervals[np.clip(idx, 0, n_int - 1)])

    # ---- gai_discretise ------------------------------------------------------
    # choose upper with prob (x - lower) / (upper - lower), else lower.
    span = (upper - lower).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_prob = np.where(span > 0, (power - lower) / span, 0.0)
    discretised = np.where(np.random.random(n) < v_prob, upper, lower)

    # ---- gai_perturb ---------------------------------------------------------
    # keep true value with prob p = e^eps / (n_int - 1 + e^eps),
    # otherwise pick uniformly among the other n_int - 1 intervals.
    exp_eps = math.exp(epsilon)
    p_prob = exp_eps / (n_int - 1 + exp_eps)

    pos = np.searchsorted(intervals, discretised)  # index of each discretised value
    keep = np.random.random(n) < p_prob

    # uniform pick among the (n_int - 1) values that are not `pos`
    other = np.random.randint(0, n_int - 1, size=n)
    other[other >= pos] += 1
    result_idx = np.where(keep, pos, other)

    df["perturbed"] = intervals[result_idx]
    return df["perturbed"]
