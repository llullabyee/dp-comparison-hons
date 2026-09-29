from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv
from scipy.stats import kruskal, norm

from src.data.load_data import make_run_dir
from src.mechanisms.gai import gai_counts_to_power
from src.utils.config import (
    MECHANISMS,
    PREPROCESS_DIR,
    PREPROCESS_MAINS_DIR,
    load_params,
)


def smape(true_values: np.ndarray, perturbed_values: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error, ignoring NaNs."""
    numerator = np.abs(true_values - perturbed_values)
    denominator = (np.abs(true_values) + np.abs(perturbed_values)) / 2

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denominator == 0, 0, numerator / denominator)

    smape_val = np.nanmean(ratio)
    return float(smape_val)


def rmse(true_values: np.ndarray, perturbed_values: np.ndarray) -> float:
    """Root Mean Squared Error, ignoring NaNs."""
    squared_error = (true_values - perturbed_values) ** 2
    rmse_val = np.sqrt(np.nanmean(squared_error))
    return float(rmse_val)


# def mae(true_values: np.ndarray, perturbed_values: np.ndarray) -> float:
#     """Mean Absolute Error."""
#     abs_error = np.abs(true_values - perturbed_values)
#     mae_val = np.mean(abs_error)
#     return float(mae_val)


def mae(true_values: np.ndarray, perturbed_values: np.ndarray) -> float:
    """Mean Absolute Error, ignoring NaNs."""
    abs_error = np.abs(true_values - perturbed_values)
    mae_val = np.nanmean(abs_error)
    return float(mae_val)


def kruskal_wallis(true_values: np.ndarray, perturbed_values: np.ndarray):
    h_statistic, p_value = kruskal(
        true_values.ravel(), perturbed_values.ravel()
    )
    return h_statistic, p_value


def daily_power_consumption(
    vals: np.ndarray,
    sample_interval_seconds: float = 1.0,
) -> float:
    """
    Total power consumption in Wh over the period (sum across all timestamps).
    Note: This reports Wh consumed during observed periods.
    """
    sample_interval_hours = sample_interval_seconds / 3600
    power_consumption = np.nansum(vals) * sample_interval_hours
    return float(power_consumption)


def max_error_rate(
    sensitivity: int,
    epsilon: float,
    peak: int,
    norm_prob: float = 0.98,
    num_readings: int = 1,
):
    """
    Computes DPLM's maximum error bound (watts and as a fraction of peak).
    z_score uses the two-sided normal interval for `norm_prob` confidence.
    """
    z_score = norm.interval(norm_prob)[1]
    max_error_percent = (sensitivity * z_score * np.sqrt(2 * num_readings)) / (
        epsilon * peak
    )
    max_error_watts = max_error_percent * peak
    return max_error_watts, max_error_percent


def process_stats(
    stats: dict,
    day: str,
    true_vals: np.ndarray,
    perturbed_vals: np.ndarray,
    prefix: str,
) -> None:
    """
    prefix is column name if DPLM or Gai (e.g. DPLM_rmse), or the
    mechanism name if LDP-Energy (e.g. LBA_rmse).
    """
    is_mechanism = prefix in MECHANISMS
    prefix = f"{prefix}_" if prefix else ""

    if day not in stats:
        stats[day] = {
            "mae": {},
            "rmse": {},
            "smape": {},
            "power": {},
            "mains_counts": {},
            "appliance_counts": {},
            "counts_differences": {},
        }

    if "true_power_consumption" not in stats[day]["power"]:
        stats[day]["power"]["true_power_consumption"] = daily_power_consumption(
            true_vals
        )
    if not is_mechanism:
        stats[day]["power"][f"{prefix}perturbed_power_consumption"] = (
            daily_power_consumption(perturbed_vals)
        )
    else:
        if "total_true_appliance_count" not in stats[day]["appliance_counts"]:
            true_val_count = np.nansum(true_vals)
            stats[day]["appliance_counts"]["total_true_appliance_count"] = (
                true_val_count
            )
        perturbed_val_count = np.nansum(perturbed_vals)
        difference = (
            stats[day]["appliance_counts"]["total_true_appliance_count"]
            - perturbed_val_count
        )
        stats[day]["appliance_counts"][f"{prefix}total_appliance_count"] = (
            perturbed_val_count
        )
        stats[day]["counts_differences"][f"{prefix}counts_difference"] = (
            difference
        )

    stats[day]["rmse"][f"{prefix}rmse"] = rmse(true_vals, perturbed_vals)
    stats[day]["smape"][f"{prefix}smape"] = smape(true_vals, perturbed_vals)
    stats[day]["mae"][f"{prefix}mae"] = mae(true_vals, perturbed_vals)


def create_table(days: list[str], stats: dict) -> pa.Table:
    rows = [{"day": d, **stats[d]} for d in days]
    return pa.Table.from_pylist(rows)


def run_utils(home_id: str, days: list[str], perturbed_dir: Path) -> pa.Table:
    stats = {}
    for day in days:
        true_file = PREPROCESS_MAINS_DIR / "mains" / f"{home_id}_{day}.parquet"
        perturbed_file = perturbed_dir / f"{home_id}_{day}.parquet"
        true_vals = (
            pq.read_table(true_file)
            .column("mains")
            .to_numpy(zero_copy_only=False)
            .astype(float)
        )
        tbl = pq.read_table(perturbed_file)
        perturbed_vals = (
            tbl.column("dplm_noise")
            .to_numpy(zero_copy_only=False)
            .astype(float)
        )
        process_stats(stats, day, true_vals, perturbed_vals, "dplm_noise")
    table = create_table(days, stats)
    return table


def dplm_aggregate(home_ids, day, directory, col_name):
    """
    Joins all homes' readings for one day on 'timestamp' (selecting col_name
    per home), then sums across homes to produce a population-level aggregate.
    Valid for zero-mean-noise MECHANISMS (e.g. DPLM) or true/unperturbed
    mains, where a plain sum needs no debiasing correction. Join gaps
    (missing home/timestamp pairs) are treated as contributing 0.

    Returns the joined per-home table and the (t,) aggregated power array.
    """
    fp = directory / f"{home_ids[0]}_{day}.parquet"
    joined = pq.read_table(fp, columns=["timestamp", col_name])
    joined = joined.rename_columns(["timestamp", home_ids[0]])

    for home_id in home_ids[1:]:
        fp = directory / f"{home_id}_{day}.parquet"
        if not fp.exists():
            continue
        tbl = pq.read_table(fp, columns=["timestamp", col_name])
        tbl = tbl.rename_columns(["timestamp", home_id])
        joined = joined.join(tbl, keys="timestamp", join_type="full outer")
    joined = joined.sort_by("timestamp")
    home_cols = [c for c in joined.column_names if c != "timestamp"]
    values = np.column_stack(
        [joined.column(c).to_numpy(zero_copy_only=False) for c in home_cols]
    )
    values = np.nan_to_num(values, nan=0.0)
    return joined, np.sum(values, axis=1)


def main():
    params = load_params()
    peak = params["peak"]
    epsilon = params["epsilon"]
    window_size = params["window_size"]
    domain_max = params["domain_max"]

    preprocess_mains = PREPROCESS_MAINS_DIR
    preprocess_apps = (
        PREPROCESS_DIR / f"eps={epsilon:g}" / f"sens={domain_max:g}"
    )
    preprocess_apps = PREPROCESS_DIR / f"sens={domain_max:g}"
    run_dir, mains_dir, apps_dir, gai_dir, ldp_energy_dir = make_run_dir(
        epsilon, window_size, peak, domain_max
    )
    utils_dir = run_dir / "utility"
    utils_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "utility.csv"
    if out.exists():
        return
    home_ids = sorted(
        {p.name.split("_")[0] for p in preprocess_mains.glob("*_*.parquet")}
    )
    days = sorted(
        {p.stem.split("_")[1] for p in preprocess_apps.glob("*_*.parquet")}
    )
    stats = {}

    for day in days:
        # dplm utility
        _, mains_sum = dplm_aggregate(
            home_ids, day, PREPROCESS_MAINS_DIR, "mains"
        )
        _, dplm_sum = dplm_aggregate(home_ids, day, mains_dir, "dplm_noise")
        process_stats(stats, day, mains_sum, dplm_sum, "dplm")

        # 45.52 only useful for dplm.
        if epsilon == 45.52:
            table = create_table(days, stats)
            csv.write_csv(table, out)
            return

        # gai utility - power comparison
        gai_fp = gai_dir / f"{day}_gai.parquet"
        gai_agg_tbl = pq.read_table(gai_fp)
        gai_power_sum = gai_counts_to_power(gai_agg_tbl)
        process_stats(stats, day, mains_sum, gai_power_sum, "gai_power")

        # gai utility - counts comparison
        preprocess_gai_fp = PREPROCESS_MAINS_DIR / "gai" / f"{day}.parquet"
        preprocess_gai_tbl = pq.read_table(preprocess_gai_fp).drop_columns(
            "timestamp"
        )
        gai_agg_tbl = gai_agg_tbl.drop_columns("timestamp")
        preprocess_gai_aggregate_counts = np.column_stack(
            [
                preprocess_gai_tbl[n].to_numpy()
                for n in preprocess_gai_tbl.column_names
            ]
        )
        gai_agg_arr = np.column_stack(
            [gai_agg_tbl[n].to_numpy() for n in gai_agg_tbl.column_names]
        )
        print(gai_agg_tbl.schema.types)
        print(preprocess_gai_tbl.schema.types)
        print(
            set(preprocess_gai_tbl.column_names) - set(gai_agg_tbl.column_names)
        )
        print(f"preprocess: {preprocess_gai_tbl.column_names}")
        print(f"after: {gai_agg_tbl.column_names}")

        del preprocess_gai_tbl, gai_agg_tbl
        process_stats(
            stats,
            day,
            preprocess_gai_aggregate_counts,
            gai_agg_arr,
            "gai_counts",
        )

        # ldp energy - utility
        raw_apps_fp = preprocess_apps / "ldp_energy" / f"{day}.parquet"
        raw_appliance_tbl = pq.read_table(raw_apps_fp).drop_columns("timestamp")
        raw_appliance_counts = np.column_stack(
            [
                raw_appliance_tbl[n].to_numpy()
                for n in raw_appliance_tbl.column_names
            ]
        )
        for mechanism in MECHANISMS:
            file = ldp_energy_dir / f"{day}_{mechanism}.parquet"
            ldp_appliance_counts_tbl = pq.read_table(file)
            ldp_appliance_counts = np.column_stack(
                [
                    ldp_appliance_counts_tbl[col].to_numpy()
                    for col in ldp_appliance_counts_tbl.column_names
                    if col != "timestamp"
                ]
            )
            del ldp_appliance_counts_tbl
            process_stats(
                stats,
                day,
                raw_appliance_counts,
                ldp_appliance_counts,
                mechanism,
            )

    table = create_table(days, stats)
    csv.write_csv(table, out)


if __name__ == "__main__":
    main()
