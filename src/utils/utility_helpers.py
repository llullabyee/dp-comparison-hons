from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kruskal, norm

from src.utils.config import (
    PREPROCESS_MAINS_DIR,
)

METRIC_KEYS = (
    "mae",
    "rmse",
    "smape",
    "bias",
    "rmse_mae_ratio",
    "wape",
    "nrmse",
    "p95_abs_err",
    "max_abs_err",
    "corr",
    "power",
    "counts_differences",
    "avg_epsilon",
)


SCALE_DEPENDENT_METRICS = [
    "mae",
    "rmse",
    "bias",
    "p95_abs_err",
    "max_abs_err",
]


SCALE_FREE_METRICS = ["smape", "wape", "nrmse", "rmse_mae_ratio", "corr"]


METRIC_DISPLAY_NAMES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "SMAPE",
    "wape": "WAPE",
    "nrmse": "NRMSE",
    "bias": "Bias",
    "rmse_mae_ratio": "RMSE/MAE Ratio",
    "p95_abs_err": "95th Percentile Absolute Error",
    "max_abs_err": "Max Absolute Error",
    "corr": "Correlation",
    "avg_epsilon": "Average Epsilon",
    "power": "Power Consumption",
    "gai_counts": "Gai Counts",
    "appliances_counts": "Appliance Counts",
    "mains_counts": "Mains Counts",
    "counts_differences": "Counts Differences",
}


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
    bin_type: str | None = None,
) -> None:
    """
    prefix is the column name for DPLM/Gai (e.g. dplm), or
    f"{mechanism}_{bin_type}" for LDP-Energy (e.g. LBA_appliances).
    bin_type only affects LDP-Energy calls (selects the {bin_type}_counts dict).
    """
    assert true_vals.shape == perturbed_vals.shape, (
        f"{prefix}: shape {perturbed_vals.shape} vs true {true_vals.shape}"
    )
    is_mechanism = bin_type is not None

    day_stats = stats.setdefault(day, {})
    for key in METRIC_KEYS:
        day_stats.setdefault(key, {})

    if not is_mechanism:
        if "true_power_consumption" not in day_stats["power"]:
            day_stats["power"]["true_power_consumption"] = (
                daily_power_consumption(true_vals)
            )
        day_stats["power"][prefix] = daily_power_consumption(perturbed_vals)
    else:
        counts = day_stats.setdefault(f"{bin_type}_counts", {})
        if "total_true" not in counts:
            counts["total_true"] = np.nansum(true_vals)
        perturbed_count = np.nansum(perturbed_vals)
        counts[prefix] = perturbed_count
        day_stats["counts_differences"][prefix] = (
            counts["total_true"] - perturbed_count
        )

    err = perturbed_vals - true_vals
    abs_err = np.abs(err)

    day_rmse = rmse(true_vals, perturbed_vals)
    day_mae = mae(true_vals, perturbed_vals)
    day_stats["rmse"][prefix] = day_rmse
    day_stats["mae"][prefix] = day_mae
    day_stats["smape"][prefix] = smape(true_vals, perturbed_vals)
    day_stats["bias"][prefix] = np.nanmean(err)
    day_stats["rmse_mae_ratio"][prefix] = (
        day_rmse / day_mae if day_mae else np.nan
    )

    # 1. scale-free error
    abs_true_sum = np.nansum(np.abs(true_vals))
    mean_true = np.nanmean(true_vals)
    day_stats["wape"][prefix] = (
        np.nansum(abs_err) / abs_true_sum if abs_true_sum else np.nan
    )
    day_stats["nrmse"][prefix] = day_rmse / mean_true if mean_true else np.nan

    # 2. tail error
    day_stats["p95_abs_err"][prefix] = np.nanpercentile(abs_err, 95)
    day_stats["max_abs_err"][prefix] = np.nanmax(abs_err)

    # 3. shape preservation
    mask = ~(np.isnan(true_vals) | np.isnan(perturbed_vals))
    t, p = true_vals[mask].ravel(), perturbed_vals[mask].ravel()
    if t.size > 1 and t.std() > 0 and p.std() > 0:
        day_stats["corr"][prefix] = np.corrcoef(t, p)[0, 1]
    else:
        day_stats["corr"][prefix] = np.nan


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
