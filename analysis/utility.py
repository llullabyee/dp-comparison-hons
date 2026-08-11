import pandas as pd
from src.data.load_data import load_data, make_meter_dir
from src.utils.config import (
    ANALYSIS_DIR,
    PREPROCESS_DIR,
    RESULT_DIR
)


sum_list = []
prev_meter = None
meter_rows = []

gai_gen = load_data(RESULT_DIR, subfolder='gai')
dplm_gen = load_data(RESULT_DIR, subfolder='dplm')

for (gai_df, meter_id, filename, day), (dplm_df, dplm_meter_id, dplm_filename, dplm_day) in zip(gai_gen, dplm_gen):
    # sanity check the two streams are actually aligned
    assert meter_id == dplm_meter_id and day == dplm_day, f"Mismatch: {meter_id}/{day} vs {dplm_meter_id}/{dplm_day}"

    if meter_id != prev_meter:
        if prev_meter is not None:
            _, utility_fp = make_meter_dir((RESULT_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
            pd.DataFrame(meter_rows, columns=['day', 'gai_sum', 'dplm_sum']).to_csv(utility_fp, index=False)
        prev_meter = meter_id
        meter_rows = []

    gai_sum = gai_df['perturbed'].sum()
    dplm_sum = dplm_df['power_noisy'].sum()
    meter_rows.append((day, gai_sum, dplm_sum))

# flush the last meter
if prev_meter is not None:
    _, utility_fp = make_meter_dir((RESULT_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
    pd.DataFrame(meter_rows, columns=['day', 'gai_sum', 'dplm_sum']).to_csv(utility_fp, index=False)
