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

boundary_list = build_boundary_list(epsilon, domain_max)
appliance_list = list(ApplianceList)
num_boundaries = len(boundary_list)

preprocess_mains = PREPROCESS_MAINS_DIR
preprocess_apps = PREPROCESS_DIR / f"eps={epsilon:g}" / f"sens={domain_max:g}"

home_ids = sorted(
    {p.name.split("_")[0] for p in preprocess_mains.glob("*_*.parquet")}
)
days = sorted(
    {p.stem.split("_")[1] for p in preprocess_apps.glob("*_*.parquet")}
)

# %% directory creation
run_dir, mains_dir, apps_dir, gai_dir, ldp_energy_dir = make_run_dir(
    epsilon, window_size, peak, domain_max
)
mains_dir.mkdir(parents=True, exist_ok=True)
apps_dir.mkdir(parents=True, exist_ok=True)
gai_dir.mkdir(parents=True, exist_ok=True)
ldp_energy_dir.mkdir(parents=True, exist_ok=True)


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
    fps = sorted(preprocess_mains.glob(f"{home_id}_*.parquet"))

    for fp in fps:
        app_fp = preprocess_apps / fp.name
        print(f"[{home_id}] Processing {app_fp}")
        tbl = pq.read_table(fp)
        app_tbl = pq.read_table(app_fp)
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
        for mechanism in MECHANISMS:
            state = mechanism_state[mechanism]
            (
                ldp_noise,
                state["last_published_idx"],
                state["last_published_epsilon"],
                state["budget_spent"],
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

    return home_id


def process_day(day):
    mains_fps = sorted(mains_dir.glob(f"*_{day}.parquet"))
    agg_home_ids = [fp.stem.split("_")[0] for fp in mains_fps]
    tables = [
        pq.read_table(fp, columns=["timestamp", "gai_noise"]).rename_columns(
            ["timestamp", f"g{i}"]
        )
        for i, fp in enumerate(mains_fps)
    ]
    joined = reduce(
        lambda l, r: l.join(r, keys="timestamp", join_type="full outer"), tables
    ).sort_by("timestamp")
    del tables
    timestamps = joined.column("timestamp")
    joined = joined.drop_columns("timestamp")
    joined = pa.table(
        {
            c: pc.fill_null(pc.cast(joined.column(c), pa.int16()), -1)
            for c in joined.column_names
        }
    )
    arr = np.column_stack(
        [
            joined.column(c).to_numpy(zero_copy_only=False)
            for c in joined.column_names
        ]
    )
    del joined

    # %% gai aggregation
    aggregated_perturbed_values = gai_aggregate(
        arr, epsilon, num_boundaries, boundary_list
    )
    del arr
    gai_freq_tbl = pa.table(
        {
            "timestamp": timestamps,
            **{
                f"{b:g}": aggregated_perturbed_values[:, j]
                for j, b in enumerate(boundary_list)
            },
        }
    )
    del aggregated_perturbed_values
    gai_freq_fp = gai_dir / f"{day}_gai.parquet"
    Path(gai_freq_fp).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(gai_freq_tbl, gai_freq_fp)
    del gai_freq_tbl
    print(f"Gai Debiasing — homes reported {day}: {agg_home_ids}")

    # %% LDP-Energy aggregation
    for mechanism in MECHANISMS:
        appliance_fps = sorted(apps_dir.glob(f"*_{day}_{mechanism}.parquet"))
        agg_home_ids = [fp.stem.split("_")[0] for fp in appliance_fps]
        raw_tables = [pq.read_table(fp) for fp in appliance_fps]
        mech_tables = []
        col_groups = []
        for i, t in enumerate(raw_tables):
            value_cols = [c for c in t.column_names if c != "timestamp"]
            mech_tables.append(
                t.rename_columns(
                    ["timestamp"] + [f"{c}_{i}" for c in value_cols]
                )
            )
            col_groups.append([f"{c}_{i}" for c in value_cols])
        joined = reduce(
            lambda l, r: l.join(r, keys="timestamp", join_type="full outer"),
            mech_tables,
        ).sort_by("timestamp")
        del raw_tables, mech_tables
        timestamps = joined.column("timestamp")
        joined = joined.drop_columns("timestamp")
        joined = pa.table(
            {
                c: pc.fill_null(pc.cast(joined.column(c), pa.int8()), -1)
                for c in joined.column_names
            }
        )

        mech_arrs = [
            np.column_stack(
                [joined.column(c).to_numpy(zero_copy_only=False) for c in group]
            )
            for group in col_groups
        ]
        del joined
        mech_arrs = np.hstack(mech_arrs)
        mech_arrs = mech_arrs.reshape(mech_arrs.shape[0], len(col_groups), -1)

        aggregated_perturbed_values = ldp_aggregate(mech_arrs, epsilon)
        del mech_arrs
        appliance_names = [c.rsplit("_", 1)[0] for c in col_groups[0]]

        ldp_agg_tbl = pa.table(
            {
                "timestamp": timestamps,
                **{
                    name: aggregated_perturbed_values[:, j]
                    for j, name in enumerate(appliance_names)
                },
            }
        )
        del aggregated_perturbed_values
        mechanism_freq_fp = ldp_energy_dir / f"{day}_{mechanism}.parquet"
        Path(mechanism_freq_fp).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(ldp_agg_tbl, mechanism_freq_fp)
        del ldp_agg_tbl
        print(f"Appliance Debiasing — homes reported {day}: {agg_home_ids}")

    return day


# %% outer loop for perturbation — one process per home
def main():
    mains_ready = mains_dir.exists() and any(mains_dir.iterdir())
    apps_ready = apps_dir.exists() and any(apps_dir.iterdir())
    gai_ready = gai_dir.exists() and any(gai_dir.iterdir())
    ldp_ready = ldp_energy_dir.exists() and any(ldp_energy_dir.iterdir())
    if mains_ready and apps_ready and ldp_ready and gai_ready:
        return
    if not mains_ready or not apps_ready:
        with ProcessPoolExecutor(max_workers=12) as ex:  # tune to core count
            futures = {ex.submit(process_home, h): h for h in home_ids}
            for fut in as_completed(futures):
                completed_home = (
                    fut.result()
                )  # raises here if that home's work errored
                print(f"Finished home {completed_home}")
    if not gai_ready or not ldp_ready:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_day, day): day for day in days}
            for future in as_completed(futures):
                day = futures[future]
                future.result()


if __name__ == "__main__":
    main()
