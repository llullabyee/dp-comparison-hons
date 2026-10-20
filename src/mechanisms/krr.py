import numpy as np
import pyarrow as pa


def return_bounds(column, boundaries, interval_size):
    """
    finds the boundaries of values, where the boundaries are
    determined by [u,v). If a value falls on an exact boundary,
    the value will be made the lower bound (u) whilst the next
    boundary will be its upper bound. Assumption made: for a value
    falling on or greater than the bounds listed, it will be clipped
    (lower = upper)
    """
    idx = np.searchsorted(boundaries, column, side="left")
    exact = (idx < interval_size) & (
        boundaries[np.clip(idx, 0, interval_size - 1)] == column
    )
    lower = np.where(
        exact,
        boundaries[np.clip(idx, 0, interval_size - 1)],
        boundaries[np.clip(idx - 1, 0, interval_size - 1)],
    )
    upper = np.where(
        exact,
        boundaries[np.clip(idx + 1, 0, interval_size - 1)],
        boundaries[np.clip(idx, 0, interval_size - 1)],
    )
    return lower, upper


def gai_discretise(readings, upper, lower):
    span = (upper - lower).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_prob = np.where(span > 0, (readings - lower) / span, 0.0)
    discretised = np.where(np.random.random(readings.shape) < v_prob, upper, lower)
    return discretised


def gai_perturb(epsilon, n, n_int, boundaries, discretised):
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


def gai_mechanism(
    readings: np.ndarray, boundaries: np.ndarray, epsilon: float
) -> np.ndarray:
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
    discretised = gai_discretise(readings, upper, lower)

    # ---- gai_perturb ---------------------------------------------------------
    perturbed = gai_perturb(epsilon, n, n_int, boundaries, discretised)
    return perturbed


def true_aggregate(
    arr: np.ndarray,
    epsilon: float,
    num_boundaries: int,
    boundary_list: np.ndarray,
):
    q_prob = 1 / (num_boundaries - 1 + np.exp(epsilon))
    p_prob = np.exp(epsilon) * q_prob
    boundary_arr = np.asarray(boundary_list)
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]

    valid = ~np.isnan(arr)  # only NaN is excluded; inf is not silently dropped
    n_reports = valid.sum(axis=1)

    row_ids, _ = np.nonzero(valid)  # same row-major order as arr[valid]
    col_idx = np.searchsorted(boundary_arr, arr[valid])

    # Values are exact boundary points, so every index must be in range.
    assert (col_idx < num_boundaries).all()

    flat_idx = row_ids * num_boundaries + col_idx
    counts = np.bincount(flat_idx, minlength=n * num_boundaries).reshape(
        n, num_boundaries
    )
    return counts, n_reports, q_prob, p_prob


def gai_aggregate(
    arr: np.ndarray,
    epsilon: float,
    num_boundaries: int,
    boundary_list: np.ndarray,
):
    counts, n_reports, q_prob, p_prob = true_aggregate(
        arr, epsilon, num_boundaries, boundary_list
    )

    aggregated_perturbed_values = (counts - n_reports[:, None] * q_prob) / (
        p_prob - q_prob
    )
    return aggregated_perturbed_values


def gai_counts_to_power(tbl: pa.Table) -> np.ndarray:
    """
    Reconstructs total power per timestamp from GAI's debiased per-boundary
    counts: R = sum_j y_j * Phi(y_j), where column headers are Phi(y_j)
    (boundary wattage) and column values are y_j (debiased count).
    """
    level_cols = [c for c in tbl.column_names if c != "timestamp"]
    boundary_values = np.array([float(c) for c in level_cols])

    counts = np.column_stack(
        [tbl.column(c).to_numpy(zero_copy_only=False) for c in level_cols]
    )
    reconstructed = counts @ boundary_values
    return reconstructed
