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
    PERTURBED_APPS_DIR,
    PERTURBED_MAINS_DIR,
    PREPROCESS_APPS_DIR,
    PREPROCESS_MAINS_DIR,
    EPSILON,
    DOMAIN_MAX,
    WINDOW_SIZE,
    PEAK
)


# %%
mains_dir = (PERTURBED_MAINS_DIR / f'eps={EPSILON:g}' / f'w={WINDOW_SIZE}' /
           f'peak={PEAK:g}' / f'sens={DOMAIN_MAX:g}') ## investigate Hive naming thing for pyarrow dataset usage
apps_dir = (PERTURBED_APPS_DIR / f'eps={EPSILON:g}' / f'w={WINDOW_SIZE}' /
           f'peak={PEAK:g}' / f'sens={DOMAIN_MAX:g}') ## investigate Hive naming thing for pyarrow dataset usage
mains_dir.mkdir(parents=True, exist_ok=True)
apps_dir.mkdir(parents=True, exist_ok=True)

boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
appliance_list = list(ApplianceList)
num_levels = boundary_list.size - 1
mechanisms = ['LBU','LSP','LBD','LBA']

# %%
home_ids = sorted({p.name.split("_")[0] for p in PREPROCESS_MAINS_DIR.glob("*_*.parquet")})
days = sorted({p.stem.split("_")[1] for p in PREPROCESS_MAINS_DIR.glob("*_*.parquet")})

for home_id in home_ids:
    # %%
    dplm_excess = 0
    mechanism_state = {
        m: dict(last_published_idx=0, last_published_epsilon=0,
                budget_spent=deque(maxlen=WINDOW_SIZE - 1), last_published_release=None)
        for m in mechanisms
    }
    fps = sorted(list(PREPROCESS_MAINS_DIR.glob(f"{home_id}_*.parquet")))

    # %%
    for fp in fps:
        app_fp = PREPROCESS_APPS_DIR / fp.name
        tbl = pq.read_table(fp)
        app_tbl= pq.read_table(app_fp)
        mains_readings = tbl.column('mains').to_numpy(zero_copy_only=False).astype(float)
        gai_values = gai_mechanism(mains_readings, boundary_list, EPSILON)
        dplm_values, dplm_excess = dplm_mechanism(mains_readings, 0.98, EPSILON, DOMAIN_MAX, PEAK, excess=dplm_excess)
        perturbed_tbl = pa.table({
            'timestamp': tbl.column('timestamp'),
            'gai_noise': gai_values,
            'dplm_noise': dplm_values,
        })
        perturbed_fp = mains_dir / fp.name
        pq.write_table(perturbed_tbl, perturbed_fp)

        # %%
        for mechanism in mechanisms:
            state = mechanism_state[mechanism]
            ldp_noise, state['last_published_idx'], state['last_published_epsilon'], \
            state['budget_spent'], state['last_published_release'] = ldp_energy_mechanism(
                mechanism, app_tbl, EPSILON, WINDOW_SIZE,
                state['last_published_idx'], state['last_published_epsilon'],
                state['budget_spent'], state['last_published_release']
            )
            perturbed_app_fp = apps_dir / f"{fp.stem}_{mechanism}.parquet"
            pq.write_table(ldp_noise,perturbed_app_fp)




# %%
# for day in days:
#     loaded_mains = {}
#     for meter_dir, meter_dir_path in perturbed_mains_dirs.items():
#         mains_filepath = meter_dir_path / f'{day}.parquet'
#         if not mains_filepath.exists():
#             continue
#         tbl, meter, _ = load_data(mains_filepath)
#         loaded_mains[meter] = tbl

#     combined_mains_noise = np.column_stack([
#         noise.column("gai_noise").to_numpy()
#         for noise in loaded_mains.values()
#     ])
#     gai_freq_arr, debiased_freq_arr , total_power_consumption = gai_aggregate(combined_mains_noise,boundary_list,EPSILON)
#     # nothing being done with gai_freq_arr rn, do something with it?
#     gai_freq_tbl = pa.table({
#         'timestamp': next(iter(loaded_mains.values())).column('timestamp'),
#         **{f'{b:g}': debiased_freq_arr[:, j] for j, b in enumerate(boundary_list)},
#     })
#     gai_freq_fp = run_dir / 'gai' / f'{day}_gai.parquet'
#     Path(gai_freq_fp).parent.mkdir(parents=True, exist_ok=True)
#     pq.write_table(gai_freq_tbl, gai_freq_fp)

# # %%
#     for mechanism in mechanisms:
#         loaded_appliances = {}
#         aggregate_filepath = run_dir / 'appliances' / mechanism / f'{day}.parquet'
#         aggregate_filepath.parent.mkdir(parents=True, exist_ok=True)
#         for meter_id, app_dir in perturbed_appliance_dirs.items():
#             appliances_meter_filepath = app_dir / mechanism / f'{day}.parquet'
#             if not appliances_meter_filepath.exists():
#                 continue
#             app_tbl, app_meter, _ = load_data(appliances_meter_filepath)
#             level_cols = [c for c in app_tbl.column_names if c != 'timestamp']
#             loaded_appliances[app_meter] = np.column_stack([
#                 app_tbl.column(c).to_numpy() for c in level_cols
#             ])
#             timestamp_col = app_tbl.column('timestamp')
#         tables = list(loaded_appliances.values())
#         aggregated_perturbed_values = ldp_aggregate(tables, EPSILON, num_levels)
#         aggregated_tbl = pa.table({
#             'timestamp': timestamp_col,
#             **{
#                 f'{appliance_list[i // num_levels].value}_level_{i % num_levels}': aggregated_perturbed_values[:, i]
#                 for i in range(aggregated_perturbed_values.shape[1])
#             }
#         })
#         pq.write_table(aggregated_tbl, aggregate_filepath)
#     # ldp mechanisms

#         # ldp mechanism goes here

#         # ldp estimator goes here

#     # table writing goes here
