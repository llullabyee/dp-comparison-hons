"""
this code needs to be redone a bit to not do the meter dirs all at once.
"""

# %% import statements
from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from functools import reduce
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.load_data import make_run_dir
from src.mechanisms.dplm import dplm_mechanism
from src.mechanisms.gai import build_boundary_list, gai_aggregate, gai_mechanism
from src.mechanisms.ldp_energy import (
    ldp_aggregate,
    ldp_energy_mechanism,
)
from src.utils.appliance_map import ApplianceList
from src.utils.config import (
    MECHANISMS,
    PREPROCESS_DIR,
    PREPROCESS_MAINS_DIR,
    load_params,
)

# %% global variable and list initialisation
params = load_params()
peak = params["peak"]
epsilon = params["epsilon"]
window_size = params["window_size"]
domain_max = params["domain_max"]

# boundary_list = build_boundary_list(epsilon, domain_max)
boundary_list = np.linspace(0, domain_max, 10 + 1)
appliance_list = list(ApplianceList)
num_boundaries = len(boundary_list)

preprocess_mains = PREPROCESS_MAINS_DIR
# preprocess_apps = PREPROCESS_DIR / f"eps={epsilon:g}" / f"sens={domain_max:g}"
preprocess_apps = PREPROCESS_DIR / f"sens={domain_max:g}" / "appliances"
preprocess_bin_mains = PREPROCESS_DIR / f"sens={domain_max:g}" / "mains"


home_ids = sorted(
    {p.name.split("_")[0] for p in preprocess_mains.glob("*_*.parquet")}
)
days = sorted(
    {p.stem.split("_")[1] for p in preprocess_apps.glob("*_*.parquet")}
)

# %% directory creation
run_dir, mains_dir, apps_dir, bin_main_dir, gai_dir, ldp_apps_dir, ldp_mains_dir = make_run_dir(
    epsilon, window_size, peak, domain_max
)
mains_dir.mkdir(parents=True, exist_ok=True)
apps_dir.mkdir(parents=True, exist_ok=True)
bin_main_dir.mkdir(parents=True, exist_ok=True)


# %% function definition — one home's worth of work, run inside a worker process
def process_home(home_id):
    np.random.seed(hash(home_id) & 0xFFFFFFFF)

    dplm_excess = 0
    mechanism_state = {
        m: dict(
            last_published_idx=0,
            last_published_epsilon=0,
            budget_spent=deque(maxlen=window_size - 1),
            last_published_release=None,
            global_idx=0,
            excess=0,
        )
        for m in MECHANISMS
    }
    mains_mechanism_state = {
        m: dict(
            last_published_idx=0,
            last_published_epsilon=0,
            budget_spent=deque(maxlen=window_size - 1),
            last_published_release=None,
            global_idx=0,
            excess=0,
        )
        for m in MECHANISMS
    }

    fps = sorted(preprocess_mains.glob(f"{home_id}_*.parquet"))

    try:
        for fp in fps:
            app_fp = preprocess_apps / fp.name
            bin_main_fp = preprocess_bin_mains / fp.name
            print(f"[{home_id}] Processing mains: {fp}")
            tbl = pq.read_table(fp)
            mains_readings = (
                tbl.column("mains").to_numpy(zero_copy_only=False).astype(float)
            )

            # %% gai perturbation
            gai_values = gai_mechanism(mains_readings, boundary_list, epsilon)
            # %% dplm perturbation
            dplm_values, dplm_excess = dplm_mechanism(
                mains_readings, 0.98, epsilon, domain_max, peak, excess=dplm_excess
            )
            # %% table writing
            perturbed_tbl = pa.table(
                {
                    "timestamp": tbl.column("timestamp"),
                    "gai_noise": gai_values,
                    "dplm_noise": dplm_values,
                }
            )
            pq.write_table(perturbed_tbl, mains_dir / fp.name)

            # %% LDP-Energy perturbation — sequential within the process.
            # State must carry file-to-file for a given mechanism, and this
            # process already gets its parallelism from running alongside
            # other homes' processes, so no inner thread/process pool here
            # (avoids oversubscribing cores).
            if app_fp.exists():
                print(f"[{home_id}] Processing appliances: {app_fp}")
                app_tbl = pq.read_table(app_fp)
                for mechanism in MECHANISMS:
                    try:
                        state = mechanism_state[mechanism]
                        (
                            ldp_noise,
                            state["last_published_idx"],
                            state["last_published_epsilon"],
                            state["last_published_release"],
                            state["global_idx"],
                            state["excess"],
                        ) = ldp_energy_mechanism(
                            mechanism,
                            app_tbl,
                            epsilon,
                            window_size,
                            state["last_published_idx"],
                            state["last_published_epsilon"],
                            state["budget_spent"],
                            state["last_published_release"],
                            state["global_idx"],
                            state["excess"],
                        )
                        perturbed_app_fp = apps_dir / f"{fp.stem}_{mechanism}.parquet"
                        pq.write_table(ldp_noise, perturbed_app_fp)
                        del ldp_noise
                    except Exception as e:
                            raise RuntimeError(f"Failed processing home {home_id}, file {fp.name} and mechanism {mechanism} (apps)") from e

            if bin_main_fp.exists():
                print(f"[{home_id}] Processing binary mains: {bin_main_fp}")
                bin_main_tbl = pq.read_table(bin_main_fp)
                for mechanism in MECHANISMS:
                    try:
                        state = mains_mechanism_state[mechanism]
                        (
                            ldp_noise,
                            state["last_published_idx"],
                            state["last_published_epsilon"],
                            state["last_published_release"],
                            state["global_idx"],
                            state["excess"],
                        ) = ldp_energy_mechanism(
                            mechanism,
                            bin_main_tbl,
                            epsilon,
                            window_size,
                            state["last_published_idx"],
                            state["last_published_epsilon"],
                            state["budget_spent"],
                            state["last_published_release"],
                            state["global_idx"],
                            state["excess"],
                        )
                        perturbed_bin_main_fp = bin_main_dir / f"{fp.stem}_{mechanism}.parquet"
                        pq.write_table(ldp_noise, perturbed_bin_main_fp)
                    except Exception as e:
                            raise RuntimeError(f"Failed processing home {home_id}, file {fp.name} and mechanism {mechanism} (mains)") from e
        return home_id
    except Exception as e:
            raise RuntimeError(f"Failed processing home {home_id} (gai/dplm mains)") from e


# %% outer loop for perturbation — one process per home
def main():
    mains_ready = mains_dir.exists() and any(mains_dir.iterdir())
    apps_ready = apps_dir.exists() and any(apps_dir.iterdir())
    if mains_ready and apps_ready:
        return
    else:
        with ProcessPoolExecutor(max_workers=12) as ex:  # tune to core count
            futures = {ex.submit(process_home, h): h for h in home_ids}
            for fut in as_completed(futures):
                completed_home = (
                    fut.result()
                )  # raises here if that home's work errored
                print(f"Finished home {completed_home}")


if __name__ == "__main__":
    main()
