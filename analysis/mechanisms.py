# %%
import os
import math
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

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

for df, meter_id, filename in load_data(PREPROCESS_DIR / 'mains'):
    # if meter_id != prev_meter or (prev_day is not None and day != prev_day + timedelta(days=1)):
    #     dplm_excess = 0
    #     ldp_energy_excess = 0
    # prev_day = day
    # prev_meter = meter_id

    print(f"\rProcessing file for meter {meter_id}", end='', flush=True)
    perturbed_meter_dir, perturbed_result_fp = make_dirs((RESULT_DIR / 'perturbed'), filename)

    #dplm_result_dir, dplm_result_fp = make_dirs((RESULT_DIR / 'csvs', meter_id, 'dplm'), filename)
    gai_series = gai_mechanism(df, boundary_list, EPSILON)
    dplm_series = dplm_mechanism(df, 0.98, EPSILON, DOMAIN_MAX, 1500)
    timestamps = df['timestamp']
    days = df['day']
    print('gai dplm done fr')
    df = pd.read_parquet(PREPROCESS_DIR / 'appliances' / filename)
    print('yo?')
    ldp_energy_df = ldp_energy_mechanism('LBU', df, EPSILON, WINDOW_SIZE)
    print('surely')
    ldp_energy_series = concatenate_bits(ldp_energy_df)
    perturbed_df = pd.concat([timestamps, days, gai_series, dplm_series], axis = 1)
#    df[['gai_noise']].to_csv(gai_result_fp, index=False)
    perturbed_df.to_parquet(perturbed_result_fp, index=False, compression='snappy')
