import numpy as np


def build_boundary_list(epsilon: float, max_value: int) -> np.ndarray:
    """
    Build the GAI interval grid over ``[0, max_value]``.

    The number of boundaries follows the utility heuristic |X| ~= 3 * e^eps + 2,
    so higher privacy budgets (larger epsilon) allow a finer grid. The chosen
    number of points is then spread evenly (integer-valued) across the domain.
    """
    count = max(2, round(3 * np.exp(epsilon) + 2))
    step = max_value / (count - 1)
    return np.asarray([round(i * step) for i in range(count)])


def return_bounds(column, boundaries, interval_size):
    idx = np.searchsorted(boundaries, column, side="left")
    exact = (idx < interval_size) & (boundaries[np.clip(idx, 0, interval_size - 1)] == column)
    lower = np.where(exact, boundaries[np.clip(idx, 0, interval_size - 1)],
                     boundaries[np.clip(idx - 1, 0, interval_size - 1)])
    upper = np.where(exact, lower, boundaries[np.clip(idx, 0, interval_size - 1)])
    return lower, upper


def gai_mechanism(readings: np.ndarray, boundaries: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Vectorised GAI mechanism.

    Produces the ``perturbed`` column with statistics identical to applying
    ``find_bounds`` -> ``gai_discretise`` -> ``gai_perturb`` row-by-row, but
    computed over the whole column with numpy instead of per-row Python loops.
    """
    n_int = boundaries.size
    n = readings.size

    lower, upper = return_bounds(readings, boundaries, n_int)

    # ---- gai_discretise ------------------------------------------------------
    # choose upper with prob (x - lower) / (upper - lower), else lower.
    span = (upper - lower).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_prob = np.where(span > 0, (readings - lower) / span, 0.0)
    discretised = np.where(np.random.random(n) < v_prob, upper, lower)

    # ---- gai_perturb ---------------------------------------------------------
    # keep true value with prob p = e^eps / (n_int - 1 + e^eps),
    # otherwise pick uniformly among the other n_int - 1 intervals.
    exp_eps = np.exp(epsilon)
    p_prob = exp_eps / (n_int - 1 + exp_eps)

    pos = np.searchsorted(boundaries, discretised)  # index of each discretised value
    keep = np.random.random(n) < p_prob

    # uniform pick among the (n_int - 1) values that are not `pos`
    other = np.random.randint(0, n_int - 1, size=n)
    other[other >= pos] += 1
    result_idx = np.where(keep, pos, other)

    return boundaries[result_idx]


def gai_aggregate(arr: np.ndarray, epsilon: float, num_boundaries: int, boundary_list: np.ndarray):
    q_prob = 1 / (num_boundaries - 1 + np.exp(epsilon))
    p_prob = np.exp(epsilon) * q_prob
    boundary_arr = np.asarray(boundary_list)
    n = arr.shape[0]

    mask = arr != -1
    n_reports = mask.sum(axis=1)

    levels, inv = np.unique(arr[mask], return_inverse=True)
    col_idx = np.searchsorted(boundary_arr, levels)
    row_ids, _ = np.nonzero(mask)

    flat_idx = row_ids * num_boundaries + col_idx[inv]
    counts = np.bincount(flat_idx, minlength=n * num_boundaries).reshape(n, num_boundaries)

    aggregated_perturbed_values = (counts - n_reports[:, None] * q_prob) / (p_prob - q_prob)
    return aggregated_perturbed_values
