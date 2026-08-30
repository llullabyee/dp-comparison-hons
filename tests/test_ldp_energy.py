# %%
# --- synthetic test setup: one-hot true bits + real perturbation functions ---
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.stats import kruskal
from src.data.load_data import load_data
from src.mechanisms.ldp_energy import ldp_aggregate, ldp_probability, ldp_energy_randomise, ldp_budget_uniform
from src.utils.appliance_map import ApplianceList
from src.utils.config import EPSILON, WINDOW_SIZE

test_dir = Path('/tmp/ldp_test')
n_meters = 1000
n_timestamps = 4000
appliance_list = list(ApplianceList)[:15]
n_appliances = len(appliance_list)  # 21
test_num_levels = 10
n_cols = n_appliances * test_num_levels  # 189
test_mechanism = 'LBU'
test_day = '2026-08-26'

perturbed_appliance_dirs_test = {
    f'meter_{m}': test_dir / f'meter_{m}' / 'appliances'
    for m in range(n_meters)
}

rng = np.random.default_rng(42)
round_epsilon = ldp_budget_uniform(10, WINDOW_SIZE)

col_names = [
    f'{a.value}_level_{l}'
    for a in appliance_list
    for l in range(test_num_levels)
]

# accumulate true counts across ALL meters, per (appliance, level, timestamp)
true_count_histogram = np.zeros((n_timestamps, n_cols), dtype=int)

for meter_id, app_dir in perturbed_appliance_dirs_test.items():
    mech_dir = app_dir / test_mechanism
    mech_dir.mkdir(parents=True, exist_ok=True)

    true_levels = rng.integers(0, test_num_levels, size=(n_timestamps, n_appliances))
    true_bits = np.zeros((n_timestamps, n_cols), dtype='uint8')
    for a_idx in range(n_appliances):
        rows = np.arange(n_timestamps)
        cols = a_idx * test_num_levels + true_levels[:, a_idx]
        true_bits[rows, cols] = 1

    true_count_histogram += true_bits  # accumulate ground truth across meters

    bits = ldp_energy_randomise(true_bits, round_epsilon)
    tbl = pa.table({
        'timestamp': np.arange(n_timestamps),
        **{name: bits[:, i] for i, name in enumerate(col_names)}
    })
    pq.write_table(tbl, mech_dir / f'{test_day}.parquet')

print(f'Synthetic data written to {test_dir}, round_epsilon={round_epsilon:.4f}')

# %%
loaded_appliances = {}
aggregate_filepath = test_dir / 'appliances' / test_mechanism / f'{test_day}.parquet'
aggregate_filepath.parent.mkdir(parents=True, exist_ok=True)

for meter_id, app_dir in perturbed_appliance_dirs_test.items():
    appliances_meter_filepath = app_dir / test_mechanism / f'{test_day}.parquet'
    if not appliances_meter_filepath.exists():
        continue
    app_tbl, _, _ = load_data(appliances_meter_filepath)
    app_meter = appliances_meter_filepath.parent.parent.parent.name
    level_cols = [c for c in app_tbl.column_names if c != 'timestamp']
    loaded_appliances[app_meter] = np.column_stack([
        app_tbl.column(c).to_numpy() for c in level_cols
    ])
    timestamp_col = app_tbl.column('timestamp')

print(len(loaded_appliances), list(loaded_appliances.keys())[:5])

tables = list(loaded_appliances.values())
aggregated_perturbed_values = ldp_aggregate(tables, round_epsilon, n_meters)

aggregated_tbl = pa.table({
    'timestamp': timestamp_col,
    **{
        f'{appliance_list[i // test_num_levels].value}_level_{i % test_num_levels}': aggregated_perturbed_values[:, i]
        for i in range(aggregated_perturbed_values.shape[1])
    }
})
pq.write_table(aggregated_tbl, aggregate_filepath)

print('round_epsilon:', round_epsilon)
q = ldp_probability(round_epsilon)
print('q:', q, 'p-q:', 0.5 - q)

# %%
# --- Kruskal-Wallis utility check: true counts vs debiased estimates, per appliance ---
# For each appliance, compare its true per-level counts against its estimated
# per-level counts, across timestamps (gives test_num_levels x n_timestamps
# samples per group per appliance — small, but matches the paper's per-appliance
# averaging methodology).

p_values = []
for a_idx, appliance in enumerate(appliance_list):
    col_start = a_idx * test_num_levels
    col_end = col_start + test_num_levels

    true_group = true_count_histogram[:, col_start:col_end].flatten()
    est_group = aggregated_perturbed_values[:, col_start:col_end].flatten()

    stat, p = kruskal(true_group, est_group)
    p_values.append(p)
    print(f'{appliance.value:20s} stat={stat:8.3f}  p={p:.4e}')

avg_p_value = np.mean(p_values)
print(f'\naverage p-value across all {n_appliances} appliances: {avg_p_value:.4f}')
