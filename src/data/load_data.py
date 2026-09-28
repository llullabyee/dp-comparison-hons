from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import RESULT_DIR


def make_run_dir(epsilon, window_size, peak, domain_max):
    run_dir = (
        RESULT_DIR
        / f"eps={epsilon:g}"
        / f"w={window_size}"
        / f"peak={peak:g}"
        / f"sens={domain_max:g}"
    )
    mains_dir = run_dir / "mains"
    apps_dir = run_dir / "appliances"
    bin_main_dir = run_dir / "binary_mains"
    gai_dir = run_dir / "gai_agg"
    ldp_apps_dir = run_dir / "ldp_appliances_agg"
    ldp_mains_dir = run_dir / "ldp_mains_agg"
    return run_dir, mains_dir, apps_dir, bin_main_dir, gai_dir, ldp_apps_dir, ldp_mains_dir
