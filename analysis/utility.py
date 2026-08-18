import pandas as pd
import numpy as np
from pathlib import Path

from src.data.load_data import load_data, make_meter_dir
from src.mechanisms.gai import build_boundary_list, gai_estimate
from src.utils.config import CSV_DIR, PREPROCESS_DIR, EPSILON, DOMAIN_MAX


COLUMNS = ['day', 'raw_sum', 'dplm_sum']


def main():
    prev_meter = None
    meter_rows = []

    # Grid must match the one used to perturb the stored gai files.
    boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)

    perturb_gen = load_data(CSV_DIR, subfolder='perturbed')
    raw_gen = load_data(PREPROCESS_DIR)

    for (perturb_df, perturb_meter_id, perturb_filename, perturb_day), \
        (raw_df, raw_meter_id, raw_filename, raw_day) in zip(
            perturb_gen, raw_gen, strict=True
    ):
        # sanity check the three streams are actually aligned
        assert perturb_meter_id == raw_meter_id and perturb_day == raw_day, \
            f"Mismatch: {perturb_meter_id}/{perturb_day} vs {raw_meter_id}/{raw_day}"

        if perturb_meter_id != prev_meter:
            if prev_meter is not None:
                _, util_sum_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
                pd.DataFrame(meter_sum_rows, columns=COLUMNS).to_csv(util_sum_fp, index=False)
                _, util_smape_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_SMAPE_mean.csv")
                pd.DataFrame(smape_mean_rows, columns=['day', 'mean_smape']).to_csv(util_smape_fp, index=False)
            prev_meter = perturb_meter_id
            meter_sum_rows = []
            smape_mean_rows = []
            # frames = [raw_df['power'],dplm_df['dplm_noise']]

# %%
        raw_sum = raw_df['power'].sum()
        # gai_sum = gai_df['gai_noise'].sum()  # naive (biased) total, for reference
        # gai_est = gai_estimate(gai_df['gai_noise'], boundary_list, EPSILON)  # debiased (paper Sec 5.4)
        dplm_sum = perturb_df['dplm_noise'].sum()
        meter_sum_rows.append((perturb_day, raw_sum, dplm_sum))

# %% Calculate Symmetric Mean Absolute Percentage Error
        # smape = abs(raw_df['power'] - perturb_df['dplm_noise']) / \
        #         ((abs(raw_df['power']) + abs(perturb_df['dplm_noise'])) / 2)
        # smape_df = smape.to_frame(name='smape')

        denom = (raw_df['power'].abs() + perturb_df['dplm_noise'].abs()) / 2
        numer = (raw_df['power'] - perturb_df['dplm_noise']).abs()
        smape = np.where(denom == 0, 0.0, numer / denom)
        smape_df = pd.Series(smape, name='smape').to_frame()


        _, util_smape_fp = make_meter_dir(
            (CSV_DIR, prev_meter, 'utility', 'smape'),
            f"{Path(perturb_filename).stem}_SMAPE.csv"
        )
        smape_df.to_csv(util_smape_fp, index=False)

        mean_smape = smape_df['smape'].mean(numeric_only=True)
        smape_mean_rows.append((perturb_day, mean_smape))

    # flush the last meter
    if prev_meter is not None:
        _, util_sum_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
        pd.DataFrame(meter_sum_rows, columns=COLUMNS).to_csv(util_sum_fp, index=False)
        _, util_smape_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_SMAPE_mean.csv")
        pd.DataFrame(smape_mean_rows, columns=['day', 'mean_smape']).to_csv(util_smape_fp, index=False)


if __name__ == "__main__":
    main()
