from collections import deque

import numpy as np
import pyarrow as pa

from src.utils.appliance_map import ApplianceList


def return_bounds_snap(column, intervals, interval_size):
    """Always assigns to the interval [intervals[idx-1], intervals[idx]],
    even for values landing exactly on a boundary — no zero-width exact case.
    Exception: a value of 0 is always mapped to (0, 0), overriding the
    'first level' assumption noted below.
    """
    column = np.asarray(column, dtype=float)
    idx = np.searchsorted(intervals, column, side="left")
    idx = np.clip(
        idx, 1, interval_size - 1
    )  # ensure idx-1 is valid, and idx never falls on the last point alone

    lower = intervals[idx - 1]
    upper = intervals[idx]

    # value of 0 gets its own (0, 0) bucket rather than being folded into
    # the first interval
    is_zero = column == 0
    lower = np.where(is_zero, 0, lower)
    upper = np.where(is_zero, 0, upper)

    return lower, upper


# def label_intervals(boundaries: list) -> dict:
#     return {
#         i + 1: (boundaries[i], boundaries[i + 1])
#         for i in range(len(boundaries) - 1)
#     }


# def return_levels(lower, boundaries):
#     idx_of_lower = np.searchsorted(boundaries, lower, side="left")
#     return idx_of_lower + 1


def return_levels(lower, upper, boundaries):
    """Maps (lower, upper) bound pairs back to their interval level index.
    Exception: if both lower and upper are 0, return level 0.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    idx_of_lower = np.searchsorted(boundaries, lower, side="left")
    levels = idx_of_lower + 1
    is_zero = (lower == 0) & (upper == 0)
    levels = np.where(is_zero, 0, levels)

    return levels


def ldp_energy_discretise(
    tbl: pa.Table, boundaries: np.ndarray
) -> tuple[pa.Table, int, int]:
    num_levels = boundaries.size - 1
    dtype = np.min_scalar_type(num_levels)  # smallest uint dtype that fits num_levels
    num_rows = tbl.num_rows

    appliance_names = [member.value for member in ApplianceList]
    common_cols = [name for name in tbl.column_names if name in appliance_names]

    discretised = {}
    for column in common_cols:
        col = tbl.column(column).to_numpy()
        lower, upper = return_bounds_snap(col, boundaries, num_levels)
        levels = return_levels(lower, upper, boundaries)
        discretised[column] = levels.astype(dtype, copy=False)

    discretised = {
        name: discretised[name]
        if name in discretised
        else np.full(num_rows, 0, dtype=dtype)
        for name in appliance_names
    }

    return pa.table(discretised), num_levels, num_rows


def ldp_energy_encode_vectorise(
    tbl: pa.Table, num_levels: int, num_rows: int
) -> pa.Table:
    result = {}
    for column in tbl.column_names:
        levels = tbl.column(
            column
        ).to_numpy()  # 0 = missing/not-present, 1..num_levels = valid level
        one_hot = np.zeros((num_rows, num_levels), dtype=np.int16)

        valid = levels > 0
        row_idx = np.arange(num_rows)[valid]
        col_idx = levels[valid] - 1
        one_hot[row_idx, col_idx] = 1

        for level in range(num_levels):
            result[f"{column}_{level + 1}"] = one_hot[:, level]
    return pa.table(result)


# def ldp_energy_randomise(binary_vector: np.ndarray, epsilon_arr: np.ndarray) -> np.ndarray:
#     p_prob = 1/2
#     q_prob = (1 / (1 + np.exp(epsilon_arr)))

#     rand = np.random.random(binary_vector.shape)
#     perturbed = np.where(
#         binary_vector == 1,
#         (rand < p_prob).astype(int),
#         (rand < q_prob).astype(int)
#     )
#     return perturbed


def ldp_probability(epsilon):
    q_prob = 1 / (1 + np.exp(epsilon))
    return q_prob


def ldp_energy_randomise(
    binary_vector: np.ndarray, epsilon_arr: float | np.ndarray
) -> np.ndarray:
    is_2d = binary_vector.ndim == 2
    bv = binary_vector if is_2d else binary_vector[:, None]
    n = bv.shape[0]

    if not isinstance(epsilon_arr, np.ndarray):
        epsilon_col = np.full((n, 1), epsilon_arr, dtype=object)
    else:
        epsilon_col = epsilon_arr.reshape(n, 1)

    sampled_mask = (epsilon_col != None).ravel()  # noqa: E711

    output = np.empty(bv.shape, dtype=np.int16)

    sampled_epsilon = epsilon_col[sampled_mask].astype(
        float
    )  # (k,1) — broadcasts over columns
    q_prob = ldp_probability(sampled_epsilon)  # (k,1)
    rand = np.random.random((sampled_epsilon.shape[0], bv.shape[1]))
    sampled_bits = bv[sampled_mask]  # (k,m)

    output[sampled_mask] = np.where(
        sampled_bits == 1,
        rand < 0.5,
        rand < q_prob,
    ).astype(np.int16)

    last_idx = np.where(sampled_mask, np.arange(n), 0)
    np.maximum.accumulate(last_idx, out=last_idx)
    output = output[last_idx]

    return output if is_2d else output.ravel()


def ldp_budget_uniform(epsilon: float, window_size: int) -> float:
    return epsilon / window_size


def ldp_sampling_method(
    binary_vector: np.ndarray, epsilon: float, window_size: int, excess: int = 0
) -> tuple[np.ndarray, int]:
    n = binary_vector.shape[0]
    epsilon_arr = np.full(n, None, dtype=object)

    start = (window_size - excess) % window_size
    epsilon_arr[start::window_size] = epsilon

    new_excess = (excess + n) % window_size
    return epsilon_arr[:, None], new_excess


def ldp_budget_distribution(
    binary_vector: np.ndarray,
    epsilon: float,
    window_size: int,
    budget_spent: deque[float],
    last_published_release: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    releases = []
    budget_spent_arr = []
    if last_published_release is None:
        last_published_release = np.zeros(binary_vector.shape[1], dtype=np.int16)
    eps_t1 = epsilon / (2 * window_size)

    perturbed_t1_all = ldp_energy_randomise(
        binary_vector, eps_t1
    )  # vectorised for efficiency

    for idx, row in enumerate(binary_vector):
        # %% Mechanism 1
        perturbed_row_t1 = perturbed_t1_all[idx]
        dissimilarity = np.mean((perturbed_row_t1 - last_published_release) ** 2)
        # %% Mechanism 2
        eps_t2 = ((epsilon / 2) - sum(budget_spent)) / 2
        perturbed_row_t2 = ldp_energy_randomise(row, eps_t2)
        error = np.mean((perturbed_row_t2 - row) ** 2)
        # %% Mechanism 3
        if dissimilarity > error:
            last_published_release = perturbed_row_t2
            budget_spent.append(eps_t2)
            budget_spent_arr.append(eps_t2)
            releases.append(last_published_release)
        else:
            budget_spent.append(0)
            budget_spent_arr.append(0)
            releases.append(last_published_release)
    return np.array(releases), np.array(budget_spent_arr), last_published_release


def ldp_budget_absorption(
    binary_vector: np.ndarray,
    epsilon: float,
    window_size: int,
    last_published_idx: int = 0,
    global_idx: int = 0,
    last_published_epsilon: float = 0,
    last_published_release: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, int]:
    if last_published_release is None:
        last_published_release = np.zeros(binary_vector.shape[1], dtype=np.int16)
    releases = []
    budget_spent_arr = []
    eps_t1 = epsilon / (2 * window_size)
    perturbed_t1_all = ldp_energy_randomise(
        binary_vector, eps_t1
    )  # vectorised for efficiency

    for local_idx, row in enumerate(binary_vector):
        idx = global_idx + local_idx
        # %% Mechanism 2
        nullification_count = ((last_published_epsilon) / (eps_t1)) - 1
        if nullification_count >= ((idx + 1) - last_published_idx):
            releases.append(last_published_release)
            budget_spent_arr.append(0)
        else:
            # %% Mechanism 1, reordered for efficiency.
            perturbed_row_t1 = perturbed_t1_all[local_idx]
            dissimilarity = np.mean((perturbed_row_t1 - last_published_release) ** 2)
            # %% Mechanism 2 (again)
            absorption_count = (idx + 1) - (last_published_idx + nullification_count)
            eps_t2 = eps_t1 * min(absorption_count, window_size)
            perturbed_row_t2 = ldp_energy_randomise(row, eps_t2)
            error = np.mean((perturbed_row_t2 - row) ** 2)
            # %% Mechanism 3
            if dissimilarity > error:
                releases.append(perturbed_row_t2)
                budget_spent_arr.append(eps_t2)
                last_published_release = perturbed_row_t2
                last_published_idx = idx + 1
                last_published_epsilon = eps_t2
            else:
                releases.append(last_published_release)
                budget_spent_arr.append(0)
    global_idx += binary_vector.shape[0]
    return (
        np.array(releases),
        np.array(budget_spent_arr),
        last_published_idx,
        last_published_epsilon,
        last_published_release,
        global_idx,
    )


def ldp_energy_mechanism(
    mechanism: str,
    tbl: pa.Table,
    epsilon: float,
    window_size: int,
    last_published_idx: int,
    last_published_epsilon: float,
    budget_spent: deque[float],
    last_published_release: np.ndarray | None = None,
    global_idx: int = 0,
    excess: int = 0,
):
    timestamp = tbl.column("timestamp")
    tbl = tbl.drop_columns(["timestamp"])
    bin_vector_arr = np.column_stack(
        [tbl.column(name).to_numpy() for name in tbl.column_names]
    )

    match mechanism:
        case "LBU":
            eps = ldp_budget_uniform(epsilon, window_size)
            perturbed = ldp_energy_randomise(bin_vector_arr, eps)
        case "LSP":
            epsilon_arr, excess = ldp_sampling_method(
                bin_vector_arr, epsilon, window_size, excess
            )
            perturbed = ldp_energy_randomise(bin_vector_arr, epsilon_arr)
        case "LBD":
            perturbed, budget_spent_arr, last_published_release = (
                ldp_budget_distribution(
                    bin_vector_arr, epsilon, window_size, budget_spent
                )
            )
        case "LBA":
            (
                perturbed,
                budget_spent_arr,
                last_published_idx,
                last_published_epsilon,
                last_published_release,
                global_idx,
            ) = ldp_budget_absorption(
                bin_vector_arr,
                epsilon,
                window_size,
                last_published_idx,
                global_idx,
                last_published_epsilon,
                last_published_release,
            )
        case _:
            raise ValueError(f"Unknown mechanism: {mechanism}")

    perturbed_tbl = pa.table(
        {
            "timestamp": timestamp,
            **{name: perturbed[:, i] for i, name in enumerate(tbl.column_names)},
        }
    )
    return (
        perturbed_tbl,
        last_published_idx,
        last_published_epsilon,
        budget_spent,
        last_published_release,
        global_idx,
        excess,
    )


def ldp_aggregate(tables: np.ndarray, epsilon: float):
    q_prob = ldp_probability(epsilon)
    mask = tables != -1
    n_reports = mask.sum(axis=1)  # (t, m) — users reporting, per appliance
    sums = np.where(mask, tables, 0).sum(axis=1)  # (t, m)
    aggregated_perturbed_values = (sums - n_reports * q_prob) / (0.5 - q_prob)
    return aggregated_perturbed_values  # (t, m)
    return aggregated_perturbed_values  # (t, m)
