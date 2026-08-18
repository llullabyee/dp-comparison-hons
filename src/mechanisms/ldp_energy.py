import math
import numpy as np
import pandas as pd
from src.utils.appliance_map import ApplianceList

EPSILON_COLUMNS = ['LBU', 'LSP', 'eps_t2']

def return_bounds_snap(column, intervals, interval_size):
    """Always assigns to the interval [intervals[idx-1], intervals[idx]],
    even for values landing exactly on a boundary — no zero-width exact case."""
    idx = np.searchsorted(intervals, column, side="left")
    idx = np.clip(idx, 1, interval_size - 1)  # ensure idx-1 is valid, and idx never falls on the last point alone
    # note: Shanmugarasa never mention what level a value of 0 holds. This assumes a value of 0 is given the first level.
    lower = intervals[idx - 1]
    upper = intervals[idx]
    return lower, upper


def label_intervals(boundaries: list) -> dict:
    return {
        i + 1: (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    }


def return_levels(lower, boundaries):
    idx_of_lower = np.searchsorted(boundaries, lower, side="left")
    return idx_of_lower + 1


def ldp_energy_discretise(df: pd.DataFrame, boundary_list: list) -> tuple[pd.DataFrame, int]:
    boundaries = np.asarray(boundary_list)
    num_levels = boundaries.size - 1
    level_df = pd.DataFrame(0, index=df.index, columns=[e.name for e in ApplianceList])

    common_cols = level_df.columns.intersection(df.columns)
    for column in common_cols:
        col = df[column].to_numpy()
        lower, _ = return_bounds_snap(col,boundaries,num_levels)
        levels = return_levels(lower, boundaries)
        level_df[column] = levels
    return level_df, num_levels


def ldp_energy_encode_vectorise(df: pd.DataFrame, num_levels: int) -> pd.DataFrame:
    for column in df.columns:
        df[column] = pd.Categorical(df[column], categories=range(1, num_levels + 1))
    return pd.get_dummies(df, columns=list(df.columns)).astype(int)


def ldp_energy_randomise(df: pd.DataFrame, epsilon_column: str, eps_cols: list) -> pd.DataFrame:
    bit_cols = [c for c in df.columns if c not in eps_cols]
    eps_col = df[epsilon_column].values
    binary_vector = df[bit_cols].values

    p_prob = 1/2
    q_prob = (1 / (1 + np.exp(eps_col)))[:, None]

    rand = np.random.random(binary_vector.shape)
    perturbed = np.where(
        binary_vector == 1,
        (rand < p_prob).astype(int),
        (rand < q_prob).astype(int)
    )

    df[bit_cols] = perturbed
    return df

def ldp_budget_uniform(df: pd.DataFrame, epsilon: float, window_size: int) -> pd.DataFrame:
    df['LBU'] = epsilon / window_size
    return df


def ldp_sampling_method(df: pd.DataFrame, epsilon: float, window_size: int, excess: int = 0) -> tuple[pd.DataFrame, int]:
    start = excess or 0
    sampling_timestamps = np.arange(start, 86400, window_size)

    df['LSP'] = np.where(
        np.isin(np.arange(len(df)), sampling_timestamps),
        epsilon,
        0
    )

    if len(sampling_timestamps) > 0: # where would the next sample have landed, if the window continued past 86400?
        next_sample = sampling_timestamps[-1] + window_size
    else:
        next_sample = start  # window_size itself exceeds remaining rows in the day

    new_excess = next_sample - 86400
    return df, new_excess


def oue_perturb_vector(c, eps):
    p_prob = 0.5
    q_prob = 1 / (1 + np.exp(eps))
    rand = np.random.random(len(c))
    return np.where(c == 1, (rand < p_prob).astype(int), (rand < q_prob).astype(int))


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


def concatenate_bits(df: pd.DataFrame, eps_cols: list) -> pd.Series:
    bit_cols = [c for c in df.columns if c not in eps_cols]
    return df[bit_cols].astype(str).agg(''.join, axis=1).rename('ldp_energy_noise')


def ldp_energy_mechanism(mechanism: str, df: pd.DataFrame, epsilon: float, window_size: int, boundary_list: list, excess: int = 0) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df, num_levels = ldp_energy_discretise(df, boundary_list)
    df = ldp_energy_encode_vectorise(df, num_levels)

    match mechanism:
        case 'LBU':
            df = ldp_budget_uniform(df, epsilon, window_size)
            return ldp_energy_randomise(df, epsilon_column='LBU', eps_cols=EPSILON_COLUMNS), excess
        case 'LSP':
            df, excess = ldp_sampling_method(df, epsilon, window_size, excess=excess)
            return ldp_energy_randomise(df, epsilon_column='LSP', eps_cols=EPSILON_COLUMNS), excess
        case 'LBD':
            return ldp_budget_distribution(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
        case 'LBA':
            return ldp_budget_absorption(df, eps_cols=EPSILON_COLUMNS, epsilon=epsilon, window_size=window_size), excess
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
