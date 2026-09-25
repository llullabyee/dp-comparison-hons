# %%
import sys
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import kruskal

sys.path.insert(0, "/Users/lullaby/Projects/University/dp-comparison-hons")

from analysis.utility import mae
from src.mechanisms.ldp_energy import (
    ldp_energy_discretise,
    ldp_energy_encode_vectorise,
    ldp_energy_randomise,
    ldp_probability,
)
from src.utils.config import PREPROCESS_DIR, PREPROCESS_MAINS_DIR


def ldp_aggregate(tables: np.ndarray, epsilon: float, n_homes: int, m: int):
    q_prob = ldp_probability(epsilon)
    mask = (tables != -1) & (~np.isnan(tables))

    # reshape from (t, n_homes * m) -> (t, n_homes, m)
    tables_r = tables.reshape(tables.shape[0], n_homes, m)
    mask_r = mask.reshape(mask.shape[0], n_homes, m)

    n_reports = mask_r.sum(
        axis=1
    )  # (t, m) — users reporting, per appliance-bin
    sums = np.where(mask_r, tables_r, 0).sum(axis=1)  # (t, m)

    aggregated_perturbed_values = (sums - n_reports * q_prob) / (0.5 - q_prob)
    return aggregated_perturbed_values  # (t, m)


def raw_aggregate(tables: np.ndarray, epsilon: float, n_homes: int, m: int):
    mask = (tables != -1) & (~np.isnan(tables))

    # reshape from (t, n_homes * m) -> (t, n_homes, m)
    tables_r = tables.reshape(tables.shape[0], n_homes, m)
    mask_r = mask.reshape(mask.shape[0], n_homes, m)

    sums = np.where(mask_r, tables_r, 0).sum(axis=1)  # (t, m)
    return sums  # (t, m)


def test_energy_randomise(
    binary_vector: np.ndarray, epsilon: float | np.ndarray
) -> np.ndarray:
    p_prob = 0.5
    q_prob = ldp_probability(epsilon)
    rand = np.random.random((binary_vector.shape[0], binary_vector.shape[1]))

    perturbed = np.where(
        binary_vector == 1,
        rand < p_prob,
        rand < q_prob,
    ).astype(np.int8)
    return perturbed


# %%


epsilon = 10
domain_max = 76000  # ~95th percentile value from one of the homes. Determined by pasting into claude and asking it to determine it lowkey.
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

# %%
perturbed = {}
true_vals = {}
for home_id in home_ids:
    data = {}
    for day in days:
        filepath = preprocess_mains / f"{home_id}_{day}.parquet"
        if not filepath.exists():
            continue
        table = pq.read_table(filepath)
        table = table.drop_columns(["timestamp", "mains"])
        sums = {
            col: pc.sum(table[col], min_count=0).as_py()
            for col in table.column_names
        }
        data[day] = sums
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "day"
    df = df.reset_index()
    result = pa.Table.from_pandas(df)
    values, num_levels, num_rows = ldp_energy_discretise(result, boundary_list)
    values = ldp_energy_encode_vectorise(values, num_levels, num_rows)
    true_vals[home_id] = values.add_column(0, "day", result.column("day"))

    # %%
    arr = values.to_pandas().to_numpy()
    lbu_eps = epsilon / window_size
    arr = ldp_energy_randomise(arr, lbu_eps)
    values = pa.table(
        {name: arr[:, i] for i, name in enumerate(values.column_names)}
    )
    values = values.add_column(0, "day", result.column("day"))
    perturbed[home_id] = values


# %%
perturbed_dfs = []
for home_id, tbl in perturbed.items():
    df = tbl.to_pandas()
    df = df.rename(
        columns={c: f"{home_id}_{c}" for c in tbl.column_names if c != "day"}
    )
    perturbed_dfs.append(df.set_index("day"))

perturbed_df = pd.concat(perturbed_dfs, axis=1, join="outer").reset_index()
perturbed_df = perturbed_df.fillna(-1)
perturbed_arr = perturbed_df.drop(columns=["day"]).to_numpy()

true_dfs = []
for home_id, tbl in true_vals.items():
    df = tbl.to_pandas()
    df = df.rename(
        columns={c: f"{home_id}_{c}" for c in tbl.column_names if c != "day"}
    )
    true_dfs.append(df.set_index("day"))

true_df = pd.concat(true_dfs, axis=1, join="outer").reset_index()
true_df = true_df.fillna(-1)
true_arr = true_df.drop(columns=["day"]).to_numpy()

perturbed_arr

# %%
n_homes = len(perturbed)
m = len(next(iter(perturbed.values())).column_names) - 1

eps = epsilon / window_size
perturbed_agg = ldp_aggregate(perturbed_arr, eps, n_homes, m)
true_agg = raw_aggregate(true_arr, eps, n_homes, m)

df = pd.DataFrame(perturbed_agg)
df

# %%
trues = pd.DataFrame(true_agg)

trues

# %%
true_appliance_sums = np.sum(true_agg, axis=0)
perturbed_appliance_sums = np.sum(perturbed_agg, axis=0)

true_appliance_sums
# %%
perturbed_appliance_sums

# %%
val_mae = mae(true_agg, perturbed_agg)

val_mae

# %%
stat, p = kruskal(
    true_appliance_sums.flatten(), perturbed_appliance_sums.flatten()
)
stat, p
