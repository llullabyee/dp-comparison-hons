# %%
import gc
import sys
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, r"C:\Users\lullaby\Documents\dp-comparison-hons")

from analysis.utility import mae
from src.mechanisms.ldp_energy import (
    ldp_probability,
    return_bounds_snap,
    return_levels,
)
from src.utils.appliance_map import ApplianceList
from src.utils.config import PREPROCESS_DIR, PREPROCESS_MAINS_DIR

# %% variable creation
mechanism = "LBU"
epsilon = 10
domain_max = 1500
num_subintervals = 10
window_size = 3
boundary_list = np.linspace(0, domain_max, num_subintervals + 1)


preprocess_mains = PREPROCESS_MAINS_DIR
preprocess_apps = PREPROCESS_DIR / f"sens={domain_max:g}"


last_published_idx = 0
last_published_epsilon = 0
budget_spent = deque(maxlen=window_size - 1)
last_published_release = None
global_idx = 0
excess = 0


home_ids = sorted(
    {p.name.split("_")[0] for p in preprocess_mains.glob("*_*.parquet")}
)
days = sorted(
    {p.stem.split("_")[1] for p in preprocess_mains.glob("*_*.parquet")}
)


# %% test functions
def test_energy_discretise(
    tbl: pa.Table, boundaries: np.ndarray
) -> tuple[pa.Table, int, int]:
    num_levels = boundaries.size - 1
    dtype = np.min_scalar_type(
        num_levels
    )  # smallest uint dtype that fits num_levels
    num_rows = tbl.num_rows

    appliance_names = [member.value for member in ApplianceList]
    common_cols = [name for name in tbl.column_names if name in appliance_names]
    # appliance_names = ["mains"]
    # common_cols = appliance_names  # testing with main
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


def test_energy_encode_vectorise_v1(
    tbl: pa.Table, num_levels: int, num_rows: int
) -> pa.Table:
    result = {}
    for column in tbl.column_names:
        levels = tbl.column(
            column
        ).to_numpy()  # 0 = missing/not-present, 1..num_levels = valid level
        one_hot = np.zeros((num_rows, num_levels), dtype=np.int8)

        valid = levels > 0
        row_idx = np.arange(num_rows)[valid]
        col_idx = levels[valid] - 1
        one_hot[row_idx, col_idx] = 1

        for level in range(num_levels):
            result[f"{column}_{level + 1}"] = one_hot[:, level]
    return pa.table(result)


def test_energy_encode_vectorise_v2(
    tbl: pa.Table,
    num_levels: int,
    num_rows: int,
    timestamps: pa.Array | pa.ChunkedArray,
) -> tuple[pa.Table, pa.Array | pa.ChunkedArray]:
    result = {}
    for column in tbl.column_names:
        levels = tbl.column(
            column
        ).to_numpy()  # 0 = missing/not-present, 1..num_levels = valid level
        one_hot = np.zeros((num_rows, num_levels), dtype=np.int8)

        valid = levels > 0
        row_idx = np.arange(num_rows)[valid]
        col_idx = levels[valid] - 1
        one_hot[row_idx, col_idx] = 1

        for level in range(num_levels):
            result[f"{column}_{level + 1}"] = one_hot[:, level]

    # stack all one-hot columns to find rows that are entirely zero
    stacked = np.column_stack(list(result.values()))
    keep_mask = stacked.any(
        axis=1
    )  # True where at least one 1 exists in the row

    result = {name: arr[keep_mask] for name, arr in result.items()}
    timestamps = timestamps.filter(pa.array(keep_mask))

    return pa.table(result), timestamps


def test_budget_uniform(
    binary_vector: np.ndarray, epsilon: float, window_size: int
) -> np.ndarray:
    n = binary_vector.shape[0]
    actual_eps = epsilon / window_size
    epsilon_arr = np.full(n, actual_eps, dtype=np.float64)
    return epsilon_arr[:, None]


def test_energy_randomise(
    binary_vector: np.ndarray, epsilon_arr: np.ndarray
) -> np.ndarray:
    p_prob = 0.5
    n = binary_vector.shape[0]
    epsilon_col = epsilon_arr.reshape(n, 1)

    sampled_mask = (epsilon_col != None).ravel()

    output = np.empty(binary_vector.shape, dtype=np.int8)

    sampled_epsilon = epsilon_col[sampled_mask].astype(
        float
    )  # (k,1) — broadcasts over columns
    q_prob = ldp_probability(sampled_epsilon)  # (k,1)
    rand = np.random.random((sampled_epsilon.shape[0], binary_vector.shape[1]))
    sampled_bits = binary_vector[sampled_mask]  # (k,m)

    output[sampled_mask] = np.where(
        sampled_bits == 1,
        rand < p_prob,
        rand < q_prob,
    ).astype(np.int8)

    last_idx = np.where(sampled_mask, np.arange(n), 0)
    np.maximum.accumulate(last_idx, out=last_idx)
    output = output[last_idx]

    return output


# def test_raw_aggregate(tables: np.ndarray, n_homes: int, m: int):
#     mask = tables != -1

#     # reshape from (t, n_homes * m) -> (t, n_homes, m)
#     tables_r = tables.reshape(tables.shape[0], n_homes, m)
#     mask_r = mask.reshape(mask.shape[0], n_homes, m)

#     sums = np.where(mask_r, tables_r, 0).sum(axis=1)  # (t, m)
#     return sums  # (t, m)


def test_raw_aggregate(tables: np.ndarray, n_homes: int, m: int):
    tables_r = tables.reshape(tables.shape[0], n_homes, m).astype(float)

    mask_r = ~np.isnan(tables_r)
    sums = np.nansum(tables_r, axis=1)  # (t, m) — NaNs treated as 0
    n_valid = mask_r.sum(
        axis=1
    )  # (t, m) — homes reporting per timestep/appliance

    return sums, n_valid


def test_ldp_aggregate(
    tables: np.ndarray, epsilon_vals: np.ndarray, n_homes: int, m: int
):
    epsilon = np.nanmax(epsilon_vals, axis=1)
    epsilon = pd.Series(epsilon).replace(0, np.nan).ffill().to_numpy()[:, None]
    q_prob = ldp_probability(epsilon)

    tables_r = tables.reshape(tables.shape[0], n_homes, m)
    mask_r = tables_r != -1

    n_reports = mask_r.sum(axis=1)  # (t, m)
    sums = np.where(mask_r, tables_r, 0).sum(axis=1)  # (t, m)

    aggregated_perturbed_values = (sums - n_reports * q_prob) / (0.5 - q_prob)
    aggregated_perturbed_values[np.isnan(aggregated_perturbed_values)] = 0
    return aggregated_perturbed_values, epsilon.squeeze()


# %%
day = days[0]

true_tables = {}
perturbed_tables = {}

for home_id in home_ids:
    fp = preprocess_mains / f"{home_id}_{day}.parquet"
    if not fp.exists():
        continue
    # tbl = pq.read_table(fp, columns=["timestamp", "mains"])
    tbl = pq.read_table(fp)
    timestamps = tbl.column("timestamp")
    tbl = tbl.drop_columns("timestamp")
    tbl, num_levels, num_rows = test_energy_discretise(
        tbl=tbl, boundaries=boundary_list
    )
    tbl, timestamps = test_energy_encode_vectorise_v2(
        tbl=tbl, num_levels=num_levels, num_rows=num_rows, timestamps=timestamps
    )
    true_tables[home_id] = tbl.add_column(0, "timestamp", timestamps)

    print(tbl)
    table_names = tbl.column_names
    binary_vector = np.column_stack(
        [
            tbl.column(i).to_numpy(zero_copy_only=False)
            for i in range(tbl.num_columns)
        ]
    )

    epsilon_arr = test_budget_uniform(
        binary_vector=binary_vector, epsilon=epsilon, window_size=window_size
    )
    perturbed = test_energy_randomise(
        binary_vector=binary_vector, epsilon_arr=epsilon_arr
    )
    perturbed_tbl = pa.table(
        {
            "timestamp": timestamps,
            **{
                name: perturbed[:, i] for i, name in enumerate(tbl.column_names)
            },
            "epsilon": epsilon_arr.ravel(),
        }
    )
    perturbed_tables[home_id] = perturbed_tbl

# %%
true_merged = None
for key, tbl in true_tables.items():
    new_names = [
        col if col == "timestamp" else f"{key}_{col}"
        for col in tbl.column_names
    ]
    renamed = tbl.rename_columns(new_names)

    if true_merged is None:
        true_merged = renamed
    else:
        true_merged = true_merged.join(
            renamed, keys="timestamp", join_type="full outer"
        )
true_merged = true_merged.sort_by("timestamp")
print(true_merged.shape)
true_merged.to_pandas()


# %%
perturbed_merged = None
for key, tbl in perturbed_tables.items():
    new_names = [
        col if col == "timestamp" else f"{key}_{col}"
        for col in tbl.column_names
    ]
    renamed = tbl.rename_columns(new_names)

    if perturbed_merged is None:
        perturbed_merged = renamed
    else:
        perturbed_merged = perturbed_merged.join(
            renamed, keys="timestamp", join_type="full outer"
        )
perturbed_merged = perturbed_merged.sort_by("timestamp")
print(perturbed_merged.shape)
perturbed_merged.to_pandas()


# %%
true_merged = true_merged.drop_columns("timestamp")
true_arr = np.column_stack(
    [
        true_merged.column(i).to_numpy(zero_copy_only=False)
        for i in range(true_merged.num_columns)
    ]
)
pd.DataFrame(true_arr)

# %%
epsilon_table = perturbed_merged.select(
    [c for c in perturbed_merged.column_names if c.endswith("epsilon")]
)
epsilons = np.column_stack(
    [
        epsilon_table.column(c).to_numpy(zero_copy_only=False)
        for c in epsilon_table.column_names
    ]
)
epsilons = np.where(epsilons == None, 0, epsilons).astype(float)
pd.DataFrame(epsilons)

# %%
perturbed_merged = perturbed_merged.drop_columns("timestamp")
perturbed_merged = perturbed_merged.drop_columns(
    [c for c in perturbed_merged.column_names if c.endswith("epsilon")]
)
perturbed_arr = np.column_stack(
    [
        perturbed_merged.column(i).to_numpy(zero_copy_only=False)
        for i in range(perturbed_merged.num_columns)
    ]
)
pd.DataFrame(perturbed_arr)


# true_arr = np.nan_to_num(true_arr, nan=-1)
# perturbed_arr = np.nan_to_num(perturbed_arr, nan=-1)
# %%
n_homes = len(true_tables)
m = len(next(iter(true_tables.values())).column_names) - 1
print(n_homes, m)

# %%
del (
    perturbed_tables,
    true_tables,
    perturbed_merged,
    true_merged,
    epsilon_table,
)
gc.collect()

# %%
# perturbed_sum, epsilon_array = test_ldp_aggregate(
#     perturbed_arr, test_epsilon, n_homes, m
# )
# %%
# true_sum = test_raw_aggregate(true_arr, n_homes, m)
# pd.DataFrame(true_sum)

##################################### LDP AGGREGATE TESTING
# %%
agg_eps = np.nanmax(epsilons, axis=1)
pd.DataFrame(agg_eps)

# %%
q_prob = ldp_probability(epsilon)
q_prob

# %%
agg_test_perturbed_arr = perturbed_arr.reshape(
    perturbed_arr.shape[0], n_homes, m
)
print(agg_test_perturbed_arr[0])
print(agg_test_perturbed_arr[0].shape)
# %%
reporting_mask = ~np.isnan(agg_test_perturbed_arr)
reporting_mask
# %%
n_reports = reporting_mask.sum(axis=1)
with np.printoptions(threshold=np.inf):
    print(n_reports)
# %%
perturbed_sums = np.nansum(agg_test_perturbed_arr, axis=1)
perturbed_sums
# %%
aggregated_perturbed_values = (perturbed_sums - n_reports * q_prob) / (
    0.5 - q_prob
)
aggregated_perturbed_values
##################################### RAW AGGREGATE TESTING
# %%
agg_test_true_arr = true_arr.reshape(true_arr.shape[0], n_homes, m)
print(agg_test_true_arr[0])
print(agg_test_true_arr[0].shape)

# %%
true_sum = np.nansum(agg_test_true_arr, axis=1)
true_sum
# %%
mae(true_sum, aggregated_perturbed_values)
