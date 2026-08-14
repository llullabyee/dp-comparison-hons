import pandas as pd
from pathlib import Path

from src.data.load_data import load_data, make_meter_dir
from src.mechanisms.gai import build_interval_list, gai_estimate
from src.utils.config import CSV_DIR, PREPROCESS_DIR, EPSILON, DOMAIN_MAX


COLUMNS = ['day', 'raw_sum', 'dplm_sum']


def main():
    prev_meter = None
    meter_rows = []

    # Grid must match the one used to perturb the stored gai files.
    interval_list = build_interval_list(EPSILON, DOMAIN_MAX)

    gai_gen = load_data(CSV_DIR, subfolder='gai')
    dplm_gen = load_data(CSV_DIR, subfolder='dplm')
    raw_gen = load_data(PREPROCESS_DIR)

    for (gai_df, meter_id, filename, day), \
        (dplm_df, dplm_meter_id, dplm_filename, dplm_day), \
        (raw_df, raw_meter_id, raw_filename, raw_day) in zip(
            gai_gen, dplm_gen, raw_gen, strict=True
    ):
        # sanity check the three streams are actually aligned
        assert meter_id == dplm_meter_id == raw_meter_id and day == dplm_day == raw_day, \
            f"Mismatch: {meter_id}/{day} vs {dplm_meter_id}/{dplm_day} vs {raw_meter_id}/{raw_day}"

        if meter_id != prev_meter:
            if prev_meter is not None:
                _, util_sum_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
                pd.DataFrame(meter_sum_rows, columns=COLUMNS).to_csv(util_sum_fp, index=False)
                _, util_smape_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_SMAPE_mean.csv")
                pd.DataFrame(smape_mean_rows, columns=['day', 'mean_smape']).to_csv(util_smape_fp, index=False)
            prev_meter = meter_id
            meter_sum_rows = []
            smape_mean_rows = []
            # frames = [raw_df['power'],dplm_df['dplm_noise']]

# %%
        raw_sum = raw_df['power'].sum()
        # gai_sum = gai_df['gai_noise'].sum()  # naive (biased) total, for reference
        # gai_est = gai_estimate(gai_df['gai_noise'], interval_list, EPSILON)  # debiased (paper Sec 5.4)
        dplm_sum = dplm_df['dplm_noise'].sum()
        meter_sum_rows.append((day, raw_sum, dplm_sum))

# %% Calculate Symmetric Mean Absolute Percentage Error
        smape = abs(raw_df['power'] - dplm_df['dplm_noise']) / \
                ((abs(raw_df['power']) + abs(dplm_df['dplm_noise'])) / 2)
        smape_df = smape.to_frame(name='smape')

        _, util_smape_fp = make_meter_dir(
            (CSV_DIR, meter_id, 'utility', 'smape'),
            f"{Path(dplm_filename).stem}_SMAPE.csv"
        )
        smape_df.to_csv(util_smape_fp, index=False)

        mean_smape = smape_df['smape'].mean(numeric_only=True)
        smape_mean_rows.append((day, mean_smape))

    # flush the last meter
    if prev_meter is not None:
        _, util_sum_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
        pd.DataFrame(meter_sum_rows, columns=COLUMNS).to_csv(util_sum_fp, index=False)
        _, util_smape_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_SMAPE_mean.csv")
        pd.DataFrame(smape_mean_rows, columns=['day', 'mean_smape']).to_csv(util_smape_fp, index=False)


if __name__ == "__main__":
    main()
