# %%
import os
import math
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.load_data import load_data, make_meter_dir
from src.utils.config import load_config
from src.mechanisms.gai import gai_mechanism
from src.mechanisms.dplm import dplm_mechanism

from src.utils.config import (
    ANALYSIS_DIR,
    PREPROCESS_DIR,
    RESULT_DIR
)


# %%
interval_list = list(range(0, 11101, 100))

# for meter_dir in sorted(glob.glob(os.path.join(PROCESS_DIR, 'sm_*'))):
#     excess = 0
#     prev_date = None
#     for filepath in sorted(glob.glob(os.path.join(meter_dir, '*.csv'))):
#         meter_id = os.path.basename(os.path.dirname(filepath))
#         filename = os.path.basename(filepath)

#         day = datetime.strptime(filename, '%Y-%m-%d.csv').date()
#         if prev_date is not None and day != prev_date + timedelta(days=1):
#             excess = 0  # gap in dates (e.g. 06-02 missing after 06-01)
#         prev_date = day

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
    gai_mechanism(df, interval_list, 1.0)
    df[['perturbed']].to_csv(gai_result_fp, index=False)
    excess = dplm_mechanism(df, 0.98, 0.10, 1500, excess)
    df[['power_noisy']].to_csv(dplm_result_fp, index=False)
