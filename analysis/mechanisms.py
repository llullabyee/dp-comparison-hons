'''

'''
# %%
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
from pathlib import Path

from src.data.load_data import load_data
from src.mechanisms.ldp_energy import concatenate_bits, ldp_energy_mechanism
from src.mechanisms.gai import gai_aggregate, gai_mechanism, build_boundary_list
from src.mechanisms.dplm import dplm_mechanism

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
boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
excess = 0
prev_day = None
prev_meter = None
mechanisms = ['LBU','LSP','LBD','LBA']

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
days = sorted(fp.stem for fp in meter_dirs[0].glob('*.parquet'))


for day in days:
    loaded_mains = {}
    loaded_appliances = {}
    perturbed_mains = {}
    for meter_dir in meter_dirs:
        filepath = meter_dir / f'{day}.parquet'
        if not filepath.exists():
            continue
        tbl, meter, _ = load_data(filepath)
        loaded_mains[meter] = tbl

        appliance_meter_dir = PREPROCESS_DIR / 'appliances' / meter_dir.name
        appliance_filepath = appliance_meter_dir / f'{day}.parquet'
        if appliance_filepath.exists():
            app_tbl, app_meter, _ = load_data(appliance_filepath)
            loaded_appliances[meter] = app_tbl
# %%
    for meter_id, tbl in loaded_mains.items():
        dplm_excess = 0
        mains_readings = tbl.column('mains').to_numpy(zero_copy_only=False).astype(float)
        gai_values = gai_mechanism(mains_readings, boundary_list, EPSILON)
        dplm_values, dplm_excess = dplm_mechanism(mains_readings, 0.98, EPSILON, DOMAIN_MAX, 1500, excess=dplm_excess)
        # important: find a way to pass every excess value through for specific smart meters.
        perturbed_tbl = pa.table({
            'timestamp': tbl.column('timestamp'),
            'gai_noise': gai_values,
            'dplm_noise': dplm_values,
        })
        perturbed_mains[meter_id] = perturbed_tbl

    # gai estimator code goes here, taking in perturbed_mains
    print(meter_dirs)
    print(loaded_mains.keys())
    print(perturbed_mains.keys())
    combined_noise = np.column_stack([
        table.column("gai_noise").to_numpy()
        for table in perturbed_mains.values()
    ])
    print(combined_noise)
    freq_arr, debiased_freq_arr , total_power_consumption = gai_aggregate(combined_noise,boundary_list,EPSILON)
    freq_tbl = pa.table({
        'timestamp': next(iter(perturbed_mains.values())).column('timestamp'),
        **{f'{b:g}': debiased_freq_arr[:, j] for j, b in enumerate(boundary_list)},
    })
    freq_fp = run_dir / f'{day}_gai.parquet'
    pq.write_table(freq_tbl, freq_fp)
    # ldp mechanisms
    for mechanism in mechanisms:
        pass
        # ldp mechanism goes here

        # ldp estimator goes here

    # table writing goes here
    print(freq_arr, debiased_freq_arr, total_power_consumption)

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
