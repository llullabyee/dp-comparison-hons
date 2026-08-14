import math
import random
import numpy as np
import pandas as pd


def build_interval_list(epsilon: float, max_value: int) -> list:
    """
    Build the GAI interval grid over ``[0, max_value]``.

    The number of intervals follows the utility heuristic |X| ~= 3 * e^eps + 2,
    so higher privacy budgets (larger epsilon) allow a finer grid. The chosen
    number of points is then spread evenly (integer-valued) across the domain.
    """
    count = max(2, round(3 * math.exp(epsilon) + 2))
    step = max_value / (count - 1)
    return [round(i * step) for i in range(count)]


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

    df["gai_noise"] = intervals[result_idx]
    return df["gai_noise"]


def gai_estimate(perturbed, interval_list: list, epsilon: float) -> float:
    """
    Debiased aggregate estimate for GAI-perturbed data (Gai et al. 2022, Sec 5.4).

    k-RR outputs are categorical "votes", so summing the perturbed values
    directly is biased towards the domain midpoint. This inverts the noise to
    recover each grid value's true frequency and reports the estimated total.

        Phi(X_j) = [C(X_j) * (|X| - 1 + e^eps) - n] / (e^eps - 1)
        R        = sum_j  X_j * Phi(X_j)

    where C(X_j) is the observed count of grid value X_j and n is the number of
    perturbed records. `interval_list` and `epsilon` must match those used to
    perturb the data.

    Returns the estimated total (R). Note Phi values are unbiased frequency
    estimates and may individually be negative or non-integer.
    """
    intervals = np.asarray(interval_list)
    values = np.asarray(perturbed)
    n = values.size
    k = intervals.size
    exp_eps = math.exp(epsilon)

    # C(X_j): observed count of each grid value (values not on the grid -> 0).
    counts = np.zeros(k, dtype=np.int64)
    pos = np.searchsorted(intervals, values)
    on_grid = (pos < k) & (intervals[np.clip(pos, 0, k - 1)] == values)
    np.add.at(counts, pos[on_grid], 1)

    phi = (counts * (k - 1 + exp_eps) - n) / (exp_eps - 1)
    return float(np.sum(intervals * phi))
