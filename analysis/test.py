# %%
import os
import math
import glob
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.functions import make_meter_dir
from src.mechanisms.gai import gai_mechanism
# %%
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_DIR = os.path.normpath(os.path.join(ANALYSIS_DIR, '..', 'data', 'eco', 'preprocess'))
RESULT_DIR = os.path.normpath(os.path.join(ANALYSIS_DIR, '..', 'results'))
RESULT_DIR_GAI = os.path.normpath(os.path.join(RESULT_DIR, 'gai'))
RESULT_DIR_DPLM = os.path.normpath(os.path.join(RESULT_DIR, 'dplm'))

# %%
for meter_dir in glob.glob(os.path.join(PROCESS_DIR, 'sm_*')):
    for filepath in glob.glob(os.path.join(meter_dir, '*.csv')):
        meter_id = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        gai_meter_dir, gai_result_fp  = make_meter_dir(RESULT_DIR_GAI, meter_id, filename)
        dplm_result_dir, dplm_result_fp = make_meter_dir(RESULT_DIR_DPLM, meter_id, filename)

# %%
        df = pd.read_csv(filepath)

        interval_list = list(range(0, 10001, 100))
        gai_df = gai_mechanism(df, interval_list, 1.0)
        gai_df.to_csv(gai_result_fp)
