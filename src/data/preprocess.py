"""
Preprocesses the ECO dataset. Reads sm_* / plug_* files from data/eco/raw/ and
writes one parquet per household per day to data/eco/preprocess/mains/sm_*/,
containing mains power, per-appliance power, and timestamp/day columns.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

from src.mechanisms.gai import build_boundary_list
from src.mechanisms.ldp_energy import ldp_energy_discretise, ldp_energy_encode_vectorise
from src.utils.config import (
    APPLIANCES_DIR,
    MAINS_DIR,
    PREPROCESS_DIR,
    PREPROCESS_MAINS_DIR,
    load_params,
)

START_TIMESTAMP = datetime(2018, 4, 1)
END_TIMESTAMP = datetime(2018, 5, 1)
DAY_SECONDS = 86_400

_READ_OPTS = pv.ReadOptions(autogenerate_column_names=True)
_CONV_OPTS = pv.ConvertOptions(
    include_columns=["f0"], column_types={"f0": pa.float32()}
)

params = load_params()
epsilon = params["epsilon"]
domain_max = params["domain_max"]

# %% directory creation
mains_dir = PREPROCESS_MAINS_DIR
# apps_dir = PREPROCESS_DIR / f"eps={epsilon:g}" / f"sens={domain_max:g}"
apps_dir = PREPROCESS_DIR / f"sens={domain_max:g}"


# --- stage 1: per-household mains/appliance merge -------------------------


def _process_home(file, start_ts, end_ts, appliances_dir, out_dir):
    home_id = file.stem.split("_")[0]
    appliances = list(appliances_dir.glob(f"{home_id}_*.csv"))
    if not appliances:
        return

    tbls = []
    tbl = pv.read_csv(file, read_options=_READ_OPTS)
    tbl = tbl.rename_columns(["timestamp", "mains"])
    tbls.append(tbl)

    grouped = defaultdict(list)
    for appliance in appliances:
        appliance_name = appliance.stem.split("electric-appliance_")[-1]
        tbl = pv.read_csv(appliance, read_options=_READ_OPTS)
        tbl = tbl.rename_columns(["timestamp", appliance_name])
        grouped[appliance_name].append(tbl)

    for name, group in grouped.items():
        merged = pa.concat_tables(group) if len(group) > 1 else group[0]
        tbls.append(merged)

    combined = tbls[0]
    for t in tbls[1:]:
        combined = combined.join(t, keys="timestamp")

    combined = combined.set_column(
        combined.column_names.index("timestamp"),
        "timestamp",
        pc.cast(combined.column("timestamp"), pa.timestamp("ms")),
    )

    if start_ts is not None or end_ts is not None:
        mask = None
        if start_ts is not None:
            mask = pc.greater_equal(
                combined.column("timestamp"),
                pa.scalar(start_ts, type=pa.timestamp("ms")),
            )
        if end_ts is not None:
            end_mask = pc.less(
                combined.column("timestamp"),
                pa.scalar(end_ts, type=pa.timestamp("ms")),
            )
            mask = end_mask if mask is None else pc.and_(mask, end_mask)
        combined = combined.filter(mask)
        if combined.num_rows == 0:
            return

    day_col = pc.cast(combined.column("timestamp"), pa.date32())
    combined = combined.append_column("day", day_col)

    unique_days = pc.unique(combined.column("day")).to_pylist()
    for day in unique_days:
        day_tbl = combined.filter(pc.equal(combined.column("day"), day))
        day_tbl = day_tbl.drop_columns(["day"])
        out_fp = out_dir / f"{home_id}_{day}.parquet"
        pq.write_table(day_tbl, out_fp, compression="snappy")
        print(f"table written to {out_fp}")


def preprocess_data(start_ts, end_ts, max_workers=16):
    files = list(MAINS_DIR.iterdir())
    worker = partial(
        _process_home,
        start_ts=start_ts,
        end_ts=end_ts,
        appliances_dir=APPLIANCES_DIR,
        out_dir=mains_dir,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(worker, files))  # list() forces exceptions to surface


# --- stage 2: per-household-day discretise + encode ------------------------


def _process_binary_file(file, boundary_list, out_dir):
    tbl = pq.read_table(file)
    timestamps = tbl.column("timestamp")
    dropped = tbl.drop_columns(["timestamp"])
    values = pa.table(
        {name: pc.fill_null(dropped.column(name), 0) for name in dropped.column_names}
    )
    values, num_levels, num_rows = ldp_energy_discretise(values, boundary_list)
    values = ldp_energy_encode_vectorise(values, num_levels, num_rows)

    out = pa.table(
        {
            "timestamp": timestamps,
            **{name: values.column(name) for name in values.column_names},
        }
    )
    pq.write_table(out, out_dir / file.name, compression="snappy")
    print(f"{file.name}: {out.num_columns - 1} bit columns -> {out_dir}")


def preprocess_binary_vector(boundary_list: np.ndarray, max_workers=16):
    files = list(mains_dir.iterdir())
    worker = partial(
        _process_binary_file, boundary_list=boundary_list, out_dir=apps_dir
    )
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(worker, files))


def main(max_workers=12):
    mains_ready = mains_dir.exists() and any(mains_dir.iterdir())
    apps_ready = apps_dir.exists() and any(apps_dir.iterdir())
    if apps_ready:
        return

    mains_dir.mkdir(parents=True, exist_ok=True)
    apps_dir.mkdir(parents=True, exist_ok=True)

    if not mains_ready:
        preprocess_data(
            start_ts=START_TIMESTAMP, end_ts=END_TIMESTAMP, max_workers=max_workers
        )

    # boundary_list = build_boundary_list(epsilon, domain_max)
    boundary_list = np.linspace(0, domain_max, 10 + 1) # 10 is number of intervals
    preprocess_binary_vector(boundary_list, max_workers=max_workers)


if __name__ == "__main__":
    main()
