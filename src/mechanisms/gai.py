import numpy as np
import pyarrow as pa


def build_boundary_list(epsilon: float, max_value: int) -> np.ndarray:
    """
    Build the GAI interval grid over ``[0, max_value]``.

    The number of boundaries follows the utility heuristic |X| ~= 3 * e^eps + 2,
    so higher privacy budgets (larger epsilon) allow a finer grid. The chosen
    number of points is then spread evenly (integer-valued) across the domain.
    """
    target_count = round(3 * np.exp(epsilon) + 2)
    count = max(2, min(max_value + 1, target_count))
    return np.unique(np.round(np.linspace(0, max_value, count)).astype(int))
    # count = max(2, round(3 * np.exp(epsilon) + 2))
    # step = max_value / (count - 1)
    # return np.asarray([round(i * step) for i in range(count)])


def return_bounds(column, boundaries, interval_size):
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
        exact, lower, boundaries[np.clip(idx, 0, interval_size - 1)]
    )
    return lower, upper


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

    pos = np.searchsorted(
        boundaries, discretised
    )  # index of each discretised value
    keep = np.random.random(n) < p_prob

    # uniform pick among the (n_int - 1) values that are not `pos`
    other = np.random.randint(0, n_int - 1, size=n)
    other[other >= pos] += 1
    result_idx = np.where(keep, pos, other)

    return boundaries[result_idx]


def gai_aggregate(
    arr: np.ndarray,
    epsilon: float,
    num_boundaries: int,
    boundary_list: np.ndarray,
):
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
    counts = np.bincount(flat_idx, minlength=n * num_boundaries).reshape(
        n, num_boundaries
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
