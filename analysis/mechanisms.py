# %%

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from src.data.load_data import load_data, make_dirs
from src.mechanisms.ldp_energy import concatenate_bits, ldp_energy_mechanism
from src.utils.config import load_config
from src.mechanisms.gai import gai_mechanism, build_boundary_list
from src.mechanisms.dplm import dplm_mechanism

from src.utils.config import (
    ANALYSIS_DIR,
    PREPROCESS_DIR,
    RESULT_DIR,
    EPSILON,
    DOMAIN_MAX,
    WINDOW_SIZE
)


# %%
# |X| ~= 3 * e^eps + 2 intervals spread across [0, DOMAIN_MAX].
boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
excess = 0
prev_day = None
prev_meter = None

mains_gen = load_data(PREPROCESS_DIR / 'mains')
appliances_gen = load_data(PREPROCESS_DIR / 'appliances')

for (tbl, meter_id, day), \
    (ldp_tbl, ldp_meter_id, ldp_day) \
    in  zip(mains_gen, appliances_gen, strict=True
):
    print(day)
    assert (meter_id, day) == (ldp_meter_id, ldp_day), \
            f"Misaligned: mains=({meter_id}, {day}) vs appliances=({ldp_meter_id}, {ldp_day})"
    if meter_id != prev_meter or (prev_day is not None and day != prev_day + np.timedelta64(1, 'D')):
        dplm_excess = 0
        ldp_energy_excess = 0
    prev_day = day
    prev_meter = meter_id

    print(f"\rProcessing file for meter {meter_id}", end='', flush=True)
    perturbed_meter_dir, perturbed_result_fp = make_dirs((RESULT_DIR / 'perturbed' / meter_id), f'{str(day)}.parquet')

    mains_readings = tbl.column('mains').to_numpy(zero_copy_only=False).astype(float)

    #dplm_result_dir, dplm_result_fp = make_dirs((RESULT_DIR / 'csvs', meter_id, 'dplm'), filename)
    gai_noise = gai_mechanism(mains_readings, boundary_list, EPSILON)
    dplm_noise, dplm_excess = dplm_mechanism(mains_readings, 0.98, EPSILON, DOMAIN_MAX, 1500, excess=dplm_excess)
    ldp_energy_LBU, ldp_energy_excess = ldp_energy_mechanism('LBU', df, EPSILON, WINDOW_SIZE)
    # ldp_energy_LSP = ldp_energy_mechanism('LBU', df, EPSILON, WINDOW_SIZE)
    # ldp_energy_LBD = ldp_energy_mechanism('LBU', df, EPSILON, WINDOW_SIZE)
    # ldp_energy_LBA = ldp_energy_mechanism('LBU', df, EPSILON, WINDOW_SIZE)

    print('surely')
    # ldp_energy_series = concatenate_bits(ldp_energy_df)

    perturbed_tbl = pa.table({
        'timestamp': tbl.column('timestamp'),
        'gai_noise': gai_noise,
        'dplm_noise': dplm_noise,
        'ldp_energy_LBU': ldp_energy_LBU,
        # 'ldp_energy_LSP': ldp_energy_LSP,
        # 'ldp_energy_LBD': ldp_energy_LBD,
        # 'ldp_energy_LBA': ldp_energy_LBA
    })
    pq.write_table(perturbed_tbl, perturbed_result_fp)
#     perturbed_df = pd.concat([timestamps, days, gai_series, dplm_series], axis = 1)
#     df[['gai_noise']].to_csv(gai_result_fp, index=False)
#     perturbed_df.to_parquet(perturbed_result_fp, index=False, compression='snappy')
