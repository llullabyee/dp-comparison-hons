import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_data import load_data
from src.utils.config import CSV_DIR, PREPROCESS_DIR

# for df, meter_id, filename in load_data(CSV_DIR, subfolder='utility'):
#     df.plot()
#     plt.show()

dplm_gen = load_data(CSV_DIR, subfolder="dplm")
raw_gen = load_data(PREPROCESS_DIR)

for (dplm_df, dplm_meter_id, dplm_filename, dplm_day), (
    raw_df,
    raw_meter_id,
    raw_filename,
    raw_day,
) in zip(dplm_gen, raw_gen, strict=True):
    assert dplm_meter_id == raw_meter_id and dplm_day == raw_day, (
        f"Mismatch: {dplm_meter_id}/{dplm_day} vs {raw_meter_id}/{raw_day}"
    )

    frames = [raw_df["power"], dplm_df]
    df = pd.concat(frames)
    df.plot()
    plt.show()
