from numpy._core import uint8
import pyarrow as pa
import pyarrow.parquet as pq
import math
import numpy as np
import pandas as pd
from src.utils.appliance_map import ApplianceList


def return_bounds_snap(column, intervals, interval_size):
    """Always assigns to the interval [intervals[idx-1], intervals[idx]],
    even for values landing exactly on a boundary — no zero-width exact case.
    Exception: a value of 0 is always mapped to (0, 0), overriding the
    'first level' assumption noted below.
    """
    column = np.asarray(column, dtype=float)

    idx = np.searchsorted(intervals, column, side="left")
    idx = np.clip(idx, 1, interval_size - 1)  # ensure idx-1 is valid, and idx never falls on the last point alone

    lower = intervals[idx - 1]
    upper = intervals[idx]

    # value of 0 gets its own (0, 0) bucket rather than being folded into
    # the first interval
    is_zero = (column == 0)
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


def ldp_energy_discretise(tbl: pa.Table, boundaries: np.ndarray) -> tuple[pa.Table, int, int]:
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
        name: discretised[name] if name in discretised else np.full(num_rows, 0, dtype=dtype)
        for name in appliance_names
    }

    return pa.table(discretised), num_levels, num_rows


def ldp_energy_encode_vectorise(tbl: pa.Table, num_levels: int, num_rows: int) -> pa.Table:
    result = {}
    for column in tbl.column_names:
        levels = tbl.column(column).to_numpy()  # 0 = missing/not-present, 1..num_levels = valid level
        one_hot = np.zeros((num_rows, num_levels), dtype='uint8')

        valid = levels > 0
        row_idx = np.arange(num_rows)[valid]
        col_idx = levels[valid] - 1
        one_hot[row_idx, col_idx] = 1

        for level in range(num_levels):
            result[f'{column}_{level + 1}'] = one_hot[:, level]
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
    q_prob = (1 / (1 + np.exp(epsilon)))
    return q_prob


def ldp_budget_uniform(epsilon: float,
                       window_size: int) -> float:
    return epsilon / window_size


def ldp_energy_randomise(binary_vector, epsilon_arr) -> np.ndarray:
    q_prob = ldp_probability(epsilon_arr)
    print('q_prob calculated?')
    rand = np.random.random(binary_vector.shape[0])
    print('somethings happening')
    return np.where(
        binary_vector == 1,
        rand < 0.5,
        rand < q_prob,
    ).astype('uint8')


def ldp_sampling_method(binary_vector: np.ndarray, epsilon: float, window_size: int) -> np.ndarray:
    n = binary_vector.shape[0]
    epsilon_arr = np.where(np.arange(n) % window_size == 0, epsilon, 0.0)
    return epsilon_arr[:, None]


def concatenate_bits(binary_vector: np.ndarray) -> pd.Series:
    arr = df.to_numpy(dtype=np.uint8) + ord('0')
    n_cols = arr.shape[1]
    raw = arr.tobytes()
    strings = np.frombuffer(raw, dtype=f'S{n_cols}')
    return pd.Series(strings, index=df.index, name='ldp_energy_noise')


def ldp_energy_mechanism(mechanism: str, tbl: pa.Table, epsilon: float, window_size: int):
    timestamp = tbl.column('timestamp')
    tbl = tbl.drop_columns(['timestamp'])
    bin_vector_arr = np.column_stack([tbl.column(name).to_numpy() for name in tbl.column_names])

    match mechanism:
        case 'LBU':
            epsilon = ldp_budget_uniform(epsilon, window_size)
            print('epsilon done')
            perturbed = ldp_energy_randomise(bin_vector_arr, epsilon)
            perturbed_tbl = pa.table({
                name: perturbed[:, i]
                for i, name in enumerate(tbl.column_names)
            })
            print('bitcols done?')
            return perturbed_tbl
        # case 'LSP':
        #     epsilon_arr = ldp_sampling_method(bin_vector_arr, epsilon, window_size)
        #     df[bit_cols] = ldp_energy_randomise(bin_vector_arr, epsilon_arr)
        #     return df
        # case 'LBD':
        #     return ldp_budget_distribution(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
        # case 'LBA':
        #     return ldp_budget_absorption(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
        case _:
            raise ValueError(f"Unknown mechanism: {mechanism}")
