import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv

from src.data.load_data import make_run_dir
from src.mechanisms.gai import gai_counts_to_power
from src.utils.config import (
    MECHANISMS,
    PREPROCESS_DIR,
    PREPROCESS_MAINS_DIR,
    load_params,
)
from src.utils.utility_helpers import (
    dplm_aggregate,
    monthly_power_consumption,
    process_stats,
)

params = load_params()
peak = params["peak"]
epsilon = params["epsilon"]
window_size = params["window_size"]
domain_max = params["domain_max"]

preprocess_mains = PREPROCESS_MAINS_DIR
preprocess_apps = PREPROCESS_DIR / f"sens={domain_max:g}" / "appliances"
preprocess_bin_mains = PREPROCESS_DIR / f"sens={domain_max:g}" / "mains"
preprocess_agg = PREPROCESS_DIR / f"sens={domain_max:g}" / "ldp_energy"
(
    run_dir,
    mains_dir,
    apps_dir,
    bin_main_dir,
    gai_dir,
    ldp_apps_dir,
    ldp_mains_dir,
) = make_run_dir(epsilon, window_size, peak, domain_max)
utils_dir = run_dir / "utility"
billing_dir = run_dir / "billing_metrics"

utils_dir.mkdir(parents=True, exist_ok=True)
billing_dir.mkdir(parents=True, exist_ok=True)

home_ids = sorted({p.name.split("_")[0] for p in preprocess_mains.glob("*_*.parquet")})
days = sorted({p.stem.split("_")[1] for p in preprocess_apps.glob("*_*.parquet")})
stats = {}
billing_stats = {}


def process_ldp(bin_type, day):
    raw_fp = preprocess_agg / bin_type / f"{day}.parquet"
    raw_tbl = pq.read_table(raw_fp).drop_columns("timestamp")
    cols = raw_tbl.column_names
    raw_counts = np.column_stack([raw_tbl[n].to_numpy() for n in cols])
    del raw_tbl
    ldp_dir = ldp_apps_dir if bin_type == "appliances" else ldp_mains_dir
    for mechanism in MECHANISMS:
        file = ldp_dir / f"{day}_{mechanism}.parquet"
        ldp_tbl = pq.read_table(file, columns=[*cols, "epsilon"])
        mean_eps = np.nanmean(ldp_tbl.column("epsilon").to_numpy())
        ldp_counts = np.column_stack([ldp_tbl[c].to_numpy() for c in cols])
        del ldp_tbl
        label = f"{mechanism}_{bin_type}"
        process_stats(stats, day, raw_counts, ldp_counts, label, bin_type)
        stats[day]["avg_epsilon"][label] = mean_eps


def metric_table(stats: dict, metric: str, col_name) -> pa.Table:
    days = sorted(stats)
    labels = sorted({l for d in days for l in stats[d].get(metric, {})})
    cols = {col_name: pa.array(days, pa.string())}
    for label in labels:
        cols[label] = pa.array(
            [
                (lambda v: None if v is None else float(v))(
                    stats[d].get(metric, {}).get(label)
                )
                for d in days
            ],
            pa.float64(),
        )
    return pa.table(cols)


def main():
    for day in days:
        # dplm utility - power comparison
        _, mains_sum = dplm_aggregate(home_ids, day, PREPROCESS_MAINS_DIR, "mains")
        _, dplm_sum = dplm_aggregate(home_ids, day, mains_dir, "dplm_noise")
        process_stats(stats, day, mains_sum, dplm_sum, "dplm")

        # gai utility - power comparison
        gai_fp = gai_dir / f"{day}_gai.parquet"
        gai_agg_tbl = pq.read_table(gai_fp)
        gai_power_sum = gai_counts_to_power(gai_agg_tbl)
        process_stats(stats, day, mains_sum, gai_power_sum, "gai_power")

        # gai utility - counts comparison
        preprocess_gai_fp = PREPROCESS_MAINS_DIR / "gai" / f"{day}.parquet"
        preprocess_gai_tbl = pq.read_table(preprocess_gai_fp).drop_columns("timestamp")
        gai_agg_tbl = gai_agg_tbl.drop_columns("timestamp")
        preprocess_gai_aggregate_counts = np.column_stack(
            [preprocess_gai_tbl[n].to_numpy() for n in preprocess_gai_tbl.column_names]
        )
        gai_agg_arr = np.column_stack(
            [gai_agg_tbl[n].to_numpy() for n in gai_agg_tbl.column_names]
        )

        del preprocess_gai_tbl, gai_agg_tbl
        process_stats(
            stats,
            day,
            preprocess_gai_aggregate_counts,
            gai_agg_arr,
            "gai_counts",
            "gai",
        )

        # ldp energy - appliances utility
        process_ldp("appliances", day)
        process_ldp("mains", day)

    metrics = sorted({k for day_stats in stats.values() for k in day_stats})
    tables = {metric: metric_table(stats, metric, "day") for metric in metrics}
    for metric, table in tables.items():
        out = utils_dir / f"{metric}.csv"
        csv.write_csv(table, out)

    # billing
    for home_id in home_ids:
        true_monthly_profile = monthly_power_consumption(
            preprocess_mains, home_id, "mains"
        )
        dplm_monthly_profile = monthly_power_consumption(
            mains_dir, home_id, "dplm_noise"
        )
        process_stats(
            billing_stats,
            home_id,
            true_monthly_profile,
            dplm_monthly_profile,
            "billing",
        )
    metrics = sorted({k for day_stats in billing_stats.values() for k in day_stats})
    tables = {
        metric: metric_table(billing_stats, metric, "home_id") for metric in metrics
    }
    for metric, table in tables.items():
        out = billing_dir / f"{metric}.csv"
        csv.write_csv(table, out)


if __name__ == "__main__":
    main()
