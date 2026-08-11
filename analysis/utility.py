import pandas as pd
from src.data.load_data import load_data, make_meter_dir
from src.utils.config import CSV_DIR, PREPROCESS_DIR


def main():
    prev_meter = None
    meter_rows = []

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
                _, utility_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
                pd.DataFrame(meter_rows, columns=['day', 'raw_sum', 'gai_sum', 'dplm_sum']).to_csv(utility_fp, index=False)
            prev_meter = meter_id
            meter_rows = []

        raw_sum = raw_df['power'].sum()
        gai_sum = gai_df['perturbed'].sum()
        dplm_sum = dplm_df['power_noisy'].sum()
        meter_rows.append((day, raw_sum, gai_sum, dplm_sum))

    # flush the last meter
    if prev_meter is not None:
        _, utility_fp = make_meter_dir((CSV_DIR, prev_meter, 'utility'), f"{prev_meter}_total.csv")
        pd.DataFrame(meter_rows, columns=['day', 'raw_sum', 'gai_sum', 'dplm_sum']).to_csv(utility_fp, index=False)


if __name__ == "__main__":
    main()
