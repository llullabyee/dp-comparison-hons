import pyarrow as pa
import pyarrow.parquet as pq
import math
import numpy as np
import pandas as pd
from src.utils.appliance_map import ApplianceList

EPSILON_COLUMNS = ['LBU', 'LSP', 'eps_t2']

# def return_bounds_snap(column, intervals, interval_size):
#     """Always assigns to the interval [intervals[idx-1], intervals[idx]],
#     even for values landing exactly on a boundary — no zero-width exact case."""
#     idx = np.searchsorted(intervals, column, side="left")
#     idx = np.clip(idx, 1, interval_size - 1)  # ensure idx-1 is valid, and idx never falls on the last point alone
#     # note: Shanmugarasa never mention what level a value of 0 holds. This assumes a value of 0 is given the first level.
#     lower = intervals[idx - 1]
#     upper = intervals[idx]
#     return lower, upper


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


def label_intervals(boundaries: list) -> dict:
    return {
        i + 1: (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    }


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

def ldp_energy_discretise(df: pd.DataFrame, boundary_list: list) -> tuple[pd.DataFrame, int]:
    boundaries = np.asarray(boundary_list)
    num_levels = boundaries.size - 1
    dtype = np.min_scalar_type(num_levels)  # smallest uint dtype that fits num_levels

    appliance_cols = [e.value for e in ApplianceList]
    common_cols = df.columns.intersection(appliance_cols)
    for column in common_cols:
        col = df[column].to_numpy()
        lower, upper = return_bounds_snap(col, boundaries, num_levels)
        levels = return_levels(lower, upper, boundaries)
        df[column] = levels.astype(dtype, copy=False)

    level_df = df.reindex(columns=appliance_cols, fill_value=0)
    level_df[appliance_cols] = level_df[appliance_cols].astype(dtype, copy=False)

    return level_df, num_levels


def ldp_energy_encode_vectorise(df: pd.DataFrame, num_levels: int) -> pd.DataFrame:
    for column in df.columns:
        df[column] = pd.Categorical(df[column], categories=range(1, num_levels + 1))
    return pd.get_dummies(df, columns=list(df.columns)).astype('uint8')


def ldp_energy_randomise(binary_vector: np.ndarray, epsilon_arr: np.ndarray) -> np.ndarray:
    p_prob = 1/2
    q_prob = (1 / (1 + np.exp(epsilon_arr)))

    rand = np.random.random(binary_vector.shape)
    perturbed = np.where(
        binary_vector == 1,
        (rand < p_prob).astype(int),
        (rand < q_prob).astype(int)
    )
    return perturbed

def ldp_budget_uniform(binary_vector: np.ndarray, epsilon: float, window_size: int) -> np.ndarray:
    epsilon_value = epsilon / window_size
    epsilon_arr = np.full_like(binary_vector, epsilon_value)
    return epsilon_arr


def ldp_sampling_method(binary_vector: np.ndarray, epsilon: float, window_size: int, excess: int = 0) -> np.ndarray:
    sampling_timestamps = np.arange(0, 86400, window_size)

    epsilon_arr = np.where(
        np.isin(np.arange(binary_vector.size), sampling_timestamps),
        epsilon,
        0
    )
    return epsilon_arr


# def oue_perturb_vector(c, eps):
#     p_prob = 0.5
#     q_prob = 1 / (1 + np.exp(eps))
#     rand = np.random.random(len(c))
#     return np.where(c == 1, (rand < p_prob).astype(int), (rand < q_prob).astype(int))


def ldp_budget_distribution(df: pd.DataFrame, epsilon: float, window_size: int) -> pd.DataFrame:
    binary_vector_size = len(df["B"].iloc[0])
    init_vector = np.zeros(binary_vector_size, dtype=np.uint8)

    C = np.stack(df["B"].to_numpy()).astype(np.uint8)
    pass

# def ldp_budget_distribution(df: pd.DataFrame, eps_cols: list, epsilon: float, window_size: int) -> pd.DataFrame:
#     bit_cols = [c for c in df.columns if c not in eps_cols]
#     w = window_size
#     d = len(bit_cols)
#     n = len(df)

#     binary_vector = df[bit_cols].values.astype(float)

#     released = np.zeros((n, d))
#     eps2_history = np.zeros(n)
#     r_l = np.zeros(d)

#     for t in range(n):
#         c_t = binary_vector[t]

#         eps_t1 = epsilon / (2 * w)
#         c_bar_t1 = oue_perturb_vector(c_t, eps_t1)
#         dis = np.mean((c_bar_t1 - r_l) ** 2)

#         window_start = max(0, t - w + 1)
#         eps_rm = epsilon / 2 - eps2_history[window_start:t].sum()
#         eps_t2 = eps_rm / 2
#         c_bar_t2 = oue_perturb_vector(c_t, eps_t2)
#         err = np.mean((c_bar_t2 - c_t) ** 2)

#         if dis > err:
#             r_t = c_bar_t2
#             eps2_history[t] = eps_t2
#         else:
#             r_t = r_l
#             eps2_history[t] = 0

#         released[t] = r_t
#         r_l = r_t

#     out = pd.DataFrame(released, columns=bit_cols, index=df.index)
#     out['eps_t2'] = eps2_history
#     return out


def ldp_budget_absorption(df: pd.DataFrame, eps_cols: list, epsilon: float, window_size: int) -> pd.DataFrame:
    bit_cols = [c for c in df.columns if c not in eps_cols]
    w = window_size
    d = len(bit_cols)
    n = len(df)
    eps_base = epsilon / (2 * w)

    binary_vector = df[bit_cols].values.astype(float)

    released = np.zeros((n, d))
    eps2_history = np.zeros(n)
    r_l = np.zeros(d)
    l = 0
    eps_l2 = 0.0

    for t in range(n):
        c_t = binary_vector[t]

        c_bar_t1 = oue_perturb_vector(c_t, eps_base)
        dis = np.mean((c_bar_t1 - r_l) ** 2)

        t_N = (eps_l2 / eps_base) - 1 if eps_base > 0 else 0

        if t - l <= t_N:
            r_t = r_l
        else:
            t_A = t - (l + t_N)
            eps_t2 = eps_base * min(t_A, w)
            c_bar_t2 = oue_perturb_vector(c_t, eps_t2)
            err = np.mean((c_bar_t2 - c_t) ** 2)

            if dis > err:
                r_t = c_bar_t2
                l = t
                eps_l2 = eps_t2
                eps2_history[t] = eps_t2
            else:
                r_t = r_l

        released[t] = r_t
        r_l = r_t

    out = pd.DataFrame(released, columns=bit_cols, index=df.index)
    out['eps_t2'] = eps2_history
    return out


# def concatenate_bits(df: pd.DataFrame, eps_cols: list) -> pd.Series:
#     bit_cols = [c for c in df.columns if c not in eps_cols]
#     return df[bit_cols].astype(str).agg(''.join, axis=1).rename('ldp_energy_noise')

def concatenate_bits(df: pd.DataFrame) -> pd.Series:
    arr = df.to_numpy(dtype=np.uint8) + ord('0')
    n_cols = arr.shape[1]
    raw = arr.tobytes()
    strings = np.frombuffer(raw, dtype=f'S{n_cols}')
    return pd.Series(strings, index=df.index, name='ldp_energy_noise')

def ldp_energy_mechanism(mechanism: str, df: pd.DataFrame, epsilon: float, window_size: int):
    bin_vector_arr = df['ldp_energy_noise'].to_numpy()

    match mechanism:
        case 'LBU':
            epsilon_arr = ldp_budget_uniform(bin_vector_arr, epsilon, window_size)
            df['ldp_energy_noise'] = ldp_energy_randomise(bin_vector_arr, epsilon_arr)
            return df
        # case 'LSP':
        #     df, excess = ldp_sampling_method(df, epsilon, window_size, excess=excess)
        #     return ldp_energy_randomise(df, epsilon_column='LSP', eps_cols=EPSILON_COLUMNS), excess
        # case 'LBD':
        #     return ldp_budget_distribution(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
        # case 'LBA':
        #     return ldp_budget_absorption(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
        case _:
            raise ValueError(f"Unknown mechanism: {mechanism}")


def ldp_energy_server(perturbed_list: list, epsilon: float, num_appliances: int, num_levels: int) -> list:

    chunks = [
        pd.DataFrame({
            i: perturbed_list.str[i*num_levels:(i+1)*num_levels]
            for i in range(num_appliances)
        })
        for series in perturbed_list
    ]

    p = 0.5
    q = 1 / (1 + np.exp(epsilon))
    d = len(chunks)  # number of users

    results = []
    for i in range(num_appliances):
        user_bits = np.stack([
            np.array(chunks[u][i].apply(lambda s: [int(b) for b in s]).tolist())
            for u in range(d)
        ])  # shape (d, n_rows, num_levels)

        y = user_bits.sum(axis=0)  # shape (n_rows, num_levels)
        estimated = (y - d * q) / (p - q)

        results.append(pd.DataFrame(estimated, index=chunks[0].index))

    return results
