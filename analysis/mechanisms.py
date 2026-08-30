'''
this code needs to be redone a bit to not do the meter dirs all at once.
'''
# %%
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
from pathlib import Path
from collections import deque

from src.data.load_data import load_data
from src.mechanisms.ldp_energy import concatenate_bits, ldp_aggregate, ldp_energy_mechanism
from src.mechanisms.gai import gai_aggregate, gai_mechanism, build_boundary_list
from src.mechanisms.dplm import dplm_mechanism

from src.utils.appliance_map import ApplianceList
from src.utils.config import (
    ANALYSIS_DIR,
    PREPROCESS_DIR,
    RESULT_DIR,
    EPSILON,
    DOMAIN_MAX,
    WINDOW_SIZE,
    PEAK
)


# %%
mains_dir = PREPROCESS_DIR / 'mains'
meter_dirs = sorted(d for d in mains_dir.iterdir() if d.is_dir())
run_dir = (RESULT_DIR / 'perturbed' / f'eps{EPSILON:g}' / f'w{WINDOW_SIZE}' /
           f'peak{PEAK:g}' / f'sens{DOMAIN_MAX:g}') ## investigate Hive naming thing for pyarrow dataset usage

perturbed_mains_dirs = {d.name: run_dir / d.name / 'mains' for d in meter_dirs}
for p in perturbed_mains_dirs.values():
    p.mkdir(parents=True, exist_ok=True)

perturbed_appliance_dirs = {d.name: run_dir / d.name / 'appliances' for d in meter_dirs}
for p in perturbed_appliance_dirs.values():
    p.mkdir(parents=True, exist_ok=True)

# %%
boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
appliance_list = list(ApplianceList)
num_levels = boundary_list.size - 1
mechanisms = ['LBU','LSP','LBD','LBA']
days = sorted(fp.stem for fp in meter_dirs[0].glob('*.parquet'))

# %%
# for meter_dir in meter_dirs:
#     last_published_release = None
#     last_published_idx = 0
#     last_published_epsilon = 0
#     dplm_excess = 0
#     budget_spent = deque(maxlen=WINDOW_SIZE - 1)
#     for day in days:
#         filepath = meter_dir / f'{day}.parquet'
#         if not filepath.exists():
#             continue
#         tbl, meter_id, _ = load_data(filepath)

#         appliance_meter_dir = PREPROCESS_DIR / 'appliances' / meter_dir.name
#         appliance_filepath = appliance_meter_dir / f'{day}.parquet'
#         app_tbl, app_meter, _ = load_data(appliance_filepath)
# # %%
#         mains_readings = tbl.column('mains').to_numpy(zero_copy_only=False).astype(float)
#         gai_values = gai_mechanism(mains_readings, boundary_list, EPSILON)
#         dplm_values, dplm_excess = dplm_mechanism(mains_readings, 0.98, EPSILON, DOMAIN_MAX, PEAK, excess=dplm_excess)
#         # important: find a way to pass every excess value through for specific smart meters.
#         perturbed_tbl = pa.table({
#             'timestamp': tbl.column('timestamp'),
#             'gai_noise': gai_values,
#             'dplm_noise': dplm_values,
#         })
#         perturbed_fp = perturbed_mains_dirs[meter_id] / f'{day}.parquet'
#         pq.write_table(perturbed_tbl, perturbed_fp)
# # %%
#         for mechanism in mechanisms:
#             ldp_noise, last_published_idx, last_published_epsilon, budget_spent, \
#             last_published_release = ldp_energy_mechanism(mechanism, app_tbl, EPSILON, WINDOW_SIZE,
#                 last_published_idx, last_published_epsilon, budget_spent,
#                 last_published_release
#             )
#             perturbed_app_fp = perturbed_appliance_dirs[meter_id] / mechanism / f'{day}.parquet'
#             perturbed_app_fp.parent.mkdir(parents=True,exist_ok=True)
#             pq.write_table(ldp_noise, perturbed_app_fp)
#         print(f'{day} completed for meter {meter_id}')

# %%
for day in days:
    loaded_mains = {}
    for meter_dir, meter_dir_path in perturbed_mains_dirs.items():
        mains_filepath = meter_dir_path / f'{day}.parquet'
        if not mains_filepath.exists():
            continue
        tbl, meter, _ = load_data(mains_filepath)
        loaded_mains[meter] = tbl

    combined_mains_noise = np.column_stack([
        noise.column("gai_noise").to_numpy()
        for noise in loaded_mains.values()
    ])
    gai_freq_arr, debiased_freq_arr , total_power_consumption = gai_aggregate(combined_mains_noise,boundary_list,EPSILON)
    # nothing being done with gai_freq_arr rn, do something with it?
    gai_freq_tbl = pa.table({
        'timestamp': next(iter(loaded_mains.values())).column('timestamp'),
        **{f'{b:g}': debiased_freq_arr[:, j] for j, b in enumerate(boundary_list)},
    })
    gai_freq_fp = run_dir / 'gai' / f'{day}_gai.parquet'
    Path(gai_freq_fp).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(gai_freq_tbl, gai_freq_fp)

# %%
    for mechanism in mechanisms:
        loaded_appliances = {}
        aggregate_filepath = run_dir / 'appliances' / mechanism / f'{day}.parquet'
        aggregate_filepath.parent.mkdir(parents=True, exist_ok=True)
        for meter_id, app_dir in perturbed_appliance_dirs.items():
            appliances_meter_filepath = app_dir / mechanism / f'{day}.parquet'
            if not appliances_meter_filepath.exists():
                continue
            app_tbl, app_meter, _ = load_data(appliances_meter_filepath)
            level_cols = [c for c in app_tbl.column_names if c != 'timestamp']
            loaded_appliances[app_meter] = np.column_stack([
                app_tbl.column(c).to_numpy() for c in level_cols
            ])
            timestamp_col = app_tbl.column('timestamp')
        tables = list(loaded_appliances.values())
        aggregated_perturbed_values = ldp_aggregate(tables, EPSILON, num_levels)
        aggregated_tbl = pa.table({
            'timestamp': timestamp_col,
            **{
                f'{appliance_list[i // num_levels].value}_level_{i % num_levels}': aggregated_perturbed_values[:, i]
                for i in range(aggregated_perturbed_values.shape[1])
            }
        })
        pq.write_table(aggregated_tbl, aggregate_filepath)
    # ldp mechanisms

        # ldp mechanism goes here

        # ldp estimator goes here

    # table writing goes here

# mains_gen = load_data(PREPROCESS_DIR / 'mains')
# appliances_gen = load_data(PREPROCESS_DIR / 'appliances')



# for (tbl, meter_id, day), \
#     (ldp_tbl, ldp_meter_id, ldp_day) \
#     in  zip(mains_gen, appliances_gen, strict=True
# ):
#     print(day)
#     assert (meter_id, day) == (ldp_meter_id, ldp_day), \
#             f"Misaligned: mains=({meter_id}, {day}) vs appliances=({ldp_meter_id}, {ldp_day})"
#     if meter_id != prev_meter or (prev_day is not None and day != prev_day + np.timedelta64(1, 'D')):
#         dplm_excess = 0
#         ldp_energy_excess = 0
#     prev_day = day
#     prev_meter = meter_id

#     print(f"\rProcessing file for meter {meter_id}", end='', flush=True)
#     perturbed_meter_dir = RESULT_DIR / 'perturbed' / meter_id / 'mains'
#     perturbed_result_fp = perturbed_meter_dir / f'{str(day)}.parquet'
#     ldp_dir = RESULT_DIR / 'perturbed' / meter_id / 'appliances'
#     os.makedirs(perturbed_meter_dir, exist_ok=True)
#     os.makedirs(ldp_dir, exist_ok=True)

#     mains_readings = tbl.column('mains').to_numpy(zero_copy_only=False).astype(float)

#     gai_noise = gai_mechanism(mains_readings, boundary_list, EPSILON)
#     dplm_noise, dplm_excess = dplm_mechanism(mains_readings, 0.98, EPSILON, DOMAIN_MAX, 1500, excess=dplm_excess)

#     print('surely')

#     perturbed_tbl = pa.table({
#         'timestamp': tbl.column('timestamp'),
#         'gai_noise': gai_noise,
#         'dplm_noise': dplm_noise,
#     })
#     pq.write_table(perturbed_tbl, perturbed_result_fp)

#     for mechanism in mechanisms:
#         ldp_fp = ldp_dir / f'{str(day)}_{mechanism}.parquet'
#         ldp_noise, ldp_energy_excess = ldp_energy_mechanism(mechanism, ldp_tbl, EPSILON, WINDOW_SIZE, excess=ldp_energy_excess)
#         ldp_noise = ldp_noise.add_column(0, 'timestamp', tbl.column('timestamp'))
#         pq.write_table(ldp_noise, ldp_fp)
