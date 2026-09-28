import concurrent.futures
import time
from functools import reduce
from pathlib import Path
import pandas as pd

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.load_data import make_run_dir
from src.mechanisms.gai import (
    build_boundary_list,
    gai_aggregate,
    gai_discretise,
    return_bounds,
    true_aggregate,
)
from src.mechanisms.ldp_energy import ldp_aggregate
from src.utils.appliance_map import ApplianceList
from src.utils.config import (
    MECHANISMS,
    PREPROCESS_DIR,
    PREPROCESS_MAINS_DIR,
    load_params,
)

params = load_params()
peak = params["peak"]
epsilon = params["epsilon"]
window_size = params["window_size"]
domain_max = params["domain_max"]

# boundary_list = build_boundary_list(epsilon, domain_max)
boundary_list = np.linspace(0, domain_max, 10 + 1)
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
run_dir, mains_dir, apps_dir, bin_mains_dir, gai_dir, ldp_apps_dir, ldp_mains_dir = make_run_dir(
    epsilon, window_size, peak, domain_max
)
mains_dir.mkdir(parents=True, exist_ok=True)
apps_dir.mkdir(parents=True, exist_ok=True)
bin_mains_dir.mkdir(parents=True, exist_ok=True)
gai_dir.mkdir(parents=True, exist_ok=True)
ldp_apps_dir.mkdir(parents=True, exist_ok=True)
ldp_mains_dir.mkdir(parents=True, exist_ok=True)


def raw_aggregate(home_ids, day, directory, col_name):
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
    timestamps = joined.column("timestamp")
    joined = joined.drop_columns("timestamp")
    home_cols = [c for c in joined.column_names]
    values = np.column_stack(
        [joined.column(c).to_numpy(zero_copy_only=False) for c in home_cols]
    )
    values = np.nan_to_num(values, nan=0.0)
    lower, upper = return_bounds(values, boundary_list, boundary_list.size)
    values = gai_discretise(values, upper, lower)
    counts, _, _, _ = true_aggregate(
        values, epsilon, boundary_list.size, boundary_list
    )
    counts = pd.DataFrame(counts, columns=boundary_list)
    tbl = pa.Table.from_pandas(counts)
    tbl = tbl.add_column(0, "timestamp", timestamps)
    return tbl


def aggregation_counts(home_ids, day, directory):
    fp = directory / f"{home_ids[0]}_{day}.parquet"
    joined = pq.read_table(fp)
    column_names = joined.column_names
    new_names = [
        f"{home_ids[0]}_{name}" if name != "timestamp" else name
        for name in joined.column_names
    ]
    joined = joined.rename_columns(new_names)
    loaded_home_ids = [home_ids[0]]
    for home_id in home_ids[1:]:
        fp = directory / f"{home_id}_{day}.parquet"
        if not fp.exists():
            continue
        tbl = pq.read_table(fp)

        new_names = [
            f"{home_id}_{name}" if name != "timestamp" else name
            for name in tbl.column_names
        ]
        tbl = tbl.rename_columns(new_names)
        joined = joined.join(tbl, keys="timestamp", join_type="full outer")
        loaded_home_ids.append(home_id)
    joined = joined.sort_by("timestamp")
    timestamps = joined.column("timestamp")
    joined = joined.drop_columns("timestamp")

    n_homes = len(loaded_home_ids)
    n_appliance_bins = len(column_names) - 1
    home_cols = [c for c in joined.column_names]
    values = np.column_stack(
        [joined.column(c).to_numpy(zero_copy_only=False) for c in home_cols]
    )
    values = values.reshape(values.shape[0], n_homes, n_appliance_bins)
    true_sum = np.nansum(values, axis=1)
    del column_names[0]
    tbl = pa.table({name: true_sum[:, i] for i, name in enumerate(column_names)})
    tbl = tbl.add_column(0, "timestamp", timestamps)
    return tbl


def ldp_loop(day, in_dir, out_dir, datatype):
    for mechanism in MECHANISMS:
        appliance_fps = sorted(in_dir.glob(f"*_{day}_{mechanism}.parquet"))
        agg_home_ids = [fp.stem.split("_")[0] for fp in appliance_fps]
        raw_tables = [pq.read_table(fp) for fp in appliance_fps]
        n_homes = len(agg_home_ids)
        n_appliance_bins = raw_tables[0].num_columns - 2
        mech_tables = []
        col_groups = []
        for i, t in enumerate(raw_tables):
            value_cols = [c for c in t.column_names if c != "timestamp"]
            mech_tables.append(
                t.rename_columns(
                    ["timestamp"] + [f"{c}_{i}" for c in value_cols]
                )
            )
            col_groups.append([f"{c}_{i}" for c in value_cols if c != "epsilon"])
        for idx, tbl in enumerate(mech_tables):
            for i, field in enumerate(tbl.schema):
                if pa.types.is_null(field.type):
                    tbl = tbl.set_column(i, field.name, pa.array([None] * tbl.num_rows, type=pa.float64()))
            mech_tables[idx] = tbl
        joined = reduce(
            lambda l, r: l.join(r, keys="timestamp", join_type="full outer"),
            mech_tables,
        ).sort_by("timestamp")
        epsilon_table = joined.select([c for c in joined.column_names if c.startswith("epsilon_")])
        epsilons = np.column_stack([epsilon_table.column(c).to_numpy(zero_copy_only=False) for c in epsilon_table.column_names])
        epsilons = np.where(epsilons == None, 0, epsilons).astype(float)

        joined = joined.drop_columns([c for c in joined.column_names if c.startswith("epsilon_")])

        del raw_tables, mech_tables, epsilon_table
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

        aggregated_perturbed_values, eps_arrs = ldp_aggregate(mech_arrs, np.array(epsilons), n_homes, n_appliance_bins)
        del mech_arrs
        appliance_names = [c.rsplit("_", 1)[0] for c in col_groups[0]]

        ldp_agg_tbl = pa.table(
            {
                "timestamp": timestamps,
                **{
                    name: aggregated_perturbed_values[:, j]
                    for j, name in enumerate(appliance_names)
                },
                "epsilon": eps_arrs,
            }
        )
        del aggregated_perturbed_values
        mechanism_freq_fp = out_dir / f"{day}_{mechanism}.parquet"
        Path(mechanism_freq_fp).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(ldp_agg_tbl, mechanism_freq_fp)
        del ldp_agg_tbl
        print(f"{datatype} Debiasing — homes reported {day}: {agg_home_ids}")


def process_day(day, boundary_list, num_boundaries, gai_dir, ldp_apps_dir, ldp_mains_dir):
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
    del joined  # %% gai aggregation
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
    ldp_loop(day, apps_dir, ldp_apps_dir, "Appliances")
    ldp_loop(day, bin_mains_dir, ldp_mains_dir, "Binary Mains")

    return day


def _run_day(
    day, home_ids, preprocess_aggregate_gai_dir, preprocess_agg_ldp_mains_dir, preprocess_agg_ldp_apps_dir
):
    """
    Per-day unit of work extracted from aggregate_all's loop body, so it can
    be dispatched to a worker (process or thread) unchanged.
    """
    start = time.perf_counter()
    print(f"[{day}] starting — {len(home_ids)} homes")

    preprocess_aggregate_gai_fp = (
        preprocess_aggregate_gai_dir / f"{day}.parquet"
    )
    preprocess_agg_ldp_mains_fp = (
        preprocess_agg_ldp_mains_dir / f"{day}.parquet"
    )
    preprocess_agg_ldp_apps_fp = (
        preprocess_agg_ldp_apps_dir / f"{day}.parquet"
    )
    preprocess_main_ready = preprocess_aggregate_gai_dir.exists() and (len(list(preprocess_aggregate_gai_dir.glob("*.parquet"))) == 30)
    preprocess_app_ready = preprocess_agg_ldp_apps_dir.exists() and (len(list(preprocess_agg_ldp_apps_dir.glob("*.parquet"))) == 30)
    preprocess_bin_mains_ready = preprocess_agg_ldp_mains_dir.exists() and (len(list(preprocess_agg_ldp_mains_dir.glob("*.parquet"))) == 30)

    gai_ready = gai_dir.exists() and (len(list(gai_dir.glob("*.parquet"))) == 30)
    ldp_mains_ready = ldp_mains_dir.exists() and (len(list(ldp_mains_dir.glob("*.parquet"))) == 120)
    ldp_apps_ready = ldp_apps_dir.exists() and (len(list(ldp_apps_dir.glob("*.parquet"))) == 120)

    if not preprocess_main_ready:
        print(f"[{day}] raw_aggregate (mains)...")
        mains_agg_tbl = raw_aggregate(
            home_ids, day, PREPROCESS_MAINS_DIR, "mains"
        )
        print(f"[{day}] writing aggregate parquet preprocess mains file...")
        pq.write_table(mains_agg_tbl, preprocess_aggregate_gai_fp)
        del mains_agg_tbl
    else:
        print(f"[{day}] raw_aggregate (mains) already exists. Skipping...")

    if not preprocess_app_ready:
        print(f"[{day}] aggregation_counts (appliances)...")
        apps_agg_tbl = aggregation_counts(home_ids, day, preprocess_apps)
        print(f"[{day}] writing aggregate parquet preprocess apps file...")
        pq.write_table(apps_agg_tbl, preprocess_agg_ldp_apps_fp)
        del apps_agg_tbl
    if not preprocess_bin_mains_ready:
        print(f"[{day}] aggregation_counts (binary mains)...")
        bin_mains_agg_tbl = aggregation_counts(home_ids, day, preprocess_bin_mains)
        print(f"[{day}] writing aggregate parquet preprocess bin_mains file...")
        pq.write_table(bin_mains_agg_tbl, preprocess_agg_ldp_mains_fp)
        del bin_mains_agg_tbl
    else:
        print(
            f"[{day}] aggregation_counts (appliances) already exists. Skipping..."
        )
    if not gai_ready or not ldp_mains_ready or not ldp_apps_ready:
        print(f"[{day}] process_day (GAI + LDP-Energy mechanisms)...")
        _ = process_day(
            day, boundary_list, num_boundaries, gai_dir, ldp_apps_dir, ldp_mains_dir
        )
    else:
        print(
            f"[{day}] process_day (GAI + LDP-Energy mechanisms) already exists. Skipping..."
        )

    elapsed = time.perf_counter() - start
    print(f"[{day}] done in {elapsed:.1f}s")
    return day, elapsed


def main(home_ids, max_workers=2, use_processes=True):
    preprocess_aggregate_gai_dir = preprocess_mains / "gai"
    preprocess_agg_ldp_apps_dir = PREPROCESS_DIR / f"sens={domain_max:g}" / "ldp_energy" / "appliances"
    preprocess_agg_ldp_mains_dir = PREPROCESS_DIR / f"sens={domain_max:g}" / "ldp_energy" / "mains"
    preprocess_aggregate_gai_dir.mkdir(parents=True, exist_ok=True)
    preprocess_agg_ldp_apps_dir.mkdir(parents=True, exist_ok=True)
    preprocess_agg_ldp_mains_dir.mkdir(parents=True, exist_ok=True)

    Executor = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )
    mode = "processes" if use_processes else "threads"

    print(
        f"Starting aggregation over {len(days)} days "
        f"using {max_workers} {mode} ({len(home_ids)} homes, epsilon={epsilon:g})"
    )
    overall_start = time.perf_counter()

    with Executor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_day,
                day,
                home_ids,
                preprocess_aggregate_gai_dir,
                preprocess_agg_ldp_mains_dir,
                preprocess_agg_ldp_apps_dir,
            ): day
            for day in days
        }
        completed = 0
        failed = 0
        for future in concurrent.futures.as_completed(futures):
            day = futures[future]
            try:
                _, elapsed = future.result()
                completed += 1
                print(
                    f"Progress: {completed + failed}/{len(days)} days "
                    f"({day} took {elapsed:.1f}s)"
                )
            except Exception:
                failed += 1
                print(f"Day {day} failed:")
                raise

    overall_elapsed = time.perf_counter() - overall_start
    print(
        f"Finished: {completed} succeeded, {failed} failed, "
        f"total time {overall_elapsed:.1f}s "
        f"(avg {overall_elapsed / max(completed, 1):.1f}s/day)"
    )


if __name__ == "__main__":
    main(home_ids)
