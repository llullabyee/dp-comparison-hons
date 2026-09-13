import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
from scipy.stats import kruskal, norm

from src.data.load_data import make_run_dir
from src.utils.config import PREPROCESS_MAINS_DIR


def smape(
    true_values: np.ndarray, perturbed_values: np.ndarray
) -> tuple[np.ndarray, float]:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(true_values - perturbed_values)
    denominator = (np.abs(true_values) + np.abs(perturbed_values)) / 2

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denominator == 0, 0, numerator / denominator)

    smape_per_row = np.mean(ratio, axis=1)
    avg_smape = np.mean(ratio)
    return smape_per_row, avg_smape


def mse(
    true_values: np.ndarray, perturbed_values: np.ndarray
) -> tuple[np.ndarray, float]:
    """Mean Squared Error."""
    squared_error = (true_values - perturbed_values) ** 2
    mse_per_row = np.mean(squared_error, axis=1)
    avg_mse = np.mean(
        squared_error
    )  # direct overall mean, not mean-of-row-means
    return mse_per_row, avg_mse


def kruskal_wallis(true_values: np.ndarray, perturbed_values: np.ndarray):
    h_statistic, p_value = kruskal(
        true_values.ravel(), perturbed_values.ravel()
    )
    return h_statistic, p_value


def daily_power_consumption(
    true_vals: np.ndarray,
    perturbed_vals: np.ndarray,
    sample_interval_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Total power consumption in Wh over the period (sum across all timestamps).
    Note: This reports Wh consumed during observed periods.
    """
    sample_interval_hours = sample_interval_seconds / 3600
    true_power_consumption = np.sum(true_vals, axis=0) * sample_interval_hours
    perturbed_power_consumption = (
        np.sum(perturbed_vals, axis=0) * sample_interval_hours
    )
    return true_power_consumption, perturbed_power_consumption


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


def aggregation_counts(tables: np.ndarray):
    mask = tables != -1
    sums = np.where(mask, tables, 0).sum(axis=1)  # (t, m)
    return sums


def main():
    # params = load_params()
    # peak = params["peak"]
    # epsilon = params["epsilon"]
    # window_size = params["window_size"]
    # domain_max = params["domain_max"]
    peak, epsilon, window_size, domain_max = 1500, 1.0, 3, 3000

    mains_dir, apps_dir, gai_dir, ldp_energy_dir = make_run_dir(
        epsilon, window_size, peak, domain_max
    )
    for file in mains_dir.iterdir():
        preprocess_mains_file = PREPROCESS_MAINS_DIR / file.name
        tbl = pq.read_table(file)
        true_tbl = pq.read_table(preprocess_mains_file)
        true_vals = (
            true_tbl.column("mains")
            .to_numpy(zero_copy_only=False)
            .astype(float)
        )
        timestamps = (
            tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(float)
        )
        gai_noise = (
            tbl.column("gai_noise").to_numpy(zero_copy_only=False).astype(float)
        )
        dplm_noise = (
            tbl.column("dplm_noise")
            .to_numpy(zero_copy_only=False)
            .astype(float)
        )
        del tbl
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=true_vals, mode="lines", name="Raw mains"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=gai_noise, mode="lines", name="Gai noise"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=dplm_noise, mode="lines", name="DPLM noise"
            )
        )
        fig.show()
        break


if __name__ == "__main__":
    main()
