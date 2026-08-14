# %%
import os
import math
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.load_data import load_data, make_meter_dir
from src.utils.config import load_config
from src.mechanisms.gai import gai_mechanism, build_interval_list
from src.mechanisms.dplm import dplm_mechanism

from src.utils.config import (
    ANALYSIS_DIR,
    PREPROCESS_DIR,
    RESULT_DIR,
    EPSILON,
    DOMAIN_MAX
)


# %%
# |X| ~= 3 * e^eps + 2 intervals spread across [0, DOMAIN_MAX].
interval_list = build_interval_list(EPSILON, DOMAIN_MAX)
excess = 0
prev_day = None
prev_meter = None

for df, meter_id, filename, day in load_data(PREPROCESS_DIR):
    if meter_id != prev_meter or (prev_day is not None and day != prev_day + timedelta(days=1)):
        excess = 0
    prev_day = day
    prev_meter = meter_id

    gai_meter_dir, gai_result_fp = make_meter_dir((RESULT_DIR / 'csvs', meter_id, 'gai'), filename)
    dplm_result_dir, dplm_result_fp = make_meter_dir((RESULT_DIR / 'csvs', meter_id, 'dplm'), filename)
    gai_mechanism(df, interval_list, EPSILON)
    df[['gai_noise']].to_csv(gai_result_fp, index=False)
    excess = dplm_mechanism(df, 0.98, 0.10, 1500, excess)
    df[['dplm_noise']].to_csv(dplm_result_fp, index=False)
