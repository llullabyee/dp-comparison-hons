'''
Preprocesses the ECO dataset. Reads sm_* / plug_* files from data/eco/raw/ and
writes one parquet per household per day to data/eco/preprocess/mains/sm_*/,
containing mains power, per-appliance power, and timestamp/day columns.
'''
import os
import glob
import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.utils.appliance_map import get_appliance_name, APPLIANCE_MAP
from src.utils.config import RAW_DIR, PREPROCESS_DIR, EPSILON, DOMAIN_MAX
from src.mechanisms.ldp_energy import ldp_energy_discretise, ldp_energy_encode_vectorise
from src.mechanisms.gai import build_boundary_list
from src.data.load_data import load_data

DAY_SECONDS = 86_400

_READ_OPTS = pv.ReadOptions(autogenerate_column_names=True)
_CONV_OPTS = pv.ConvertOptions(include_columns=['f0'],
                               column_types={'f0': pa.float32()})

def read_column(filepath: str, col_name: str) -> np.ndarray:
    """First column of a headerless ECO csv, padded/truncated to one day."""
    if not os.path.exists(filepath):
        return np.full(DAY_SECONDS, np.nan, dtype='float32')
    tbl = pv.read_csv(filepath, read_options=_READ_OPTS, convert_options=_CONV_OPTS)
    arr = tbl.column('f0').to_numpy(zero_copy_only=False).astype('float32')
    if len(arr) >= DAY_SECONDS:
        return arr[:DAY_SECONDS]
    return np.pad(arr, (0, DAY_SECONDS - len(arr)), constant_values=np.nan)


def get_plug_data(filepath: str, appliance_no: str, meter_id: str) -> tuple[str, np.ndarray]:
    dirname, filename = os.path.split(filepath)
    parts = dirname.split(os.sep)
    parts = [p.replace('sm', 'plug') if p.startswith('sm_') else p for p in parts]
    new_filepath = os.path.join(os.sep.join(parts), appliance_no, filename)
    col_name = get_appliance_name(meter_id, appliance_no)
    return col_name, read_column(new_filepath, col_name)


def load_day(raw_fp: str, meter_id: str) -> tuple[pa.Table, str]:
    """One day-file for one household: mains + all appliance plugs, gap-free 1 Hz."""
    day = os.path.splitext(os.path.basename(raw_fp))[0]      # '2012-06-01'

    start = np.datetime64(day, 's')
    timestamps = start + np.arange(DAY_SECONDS, dtype='timedelta64[s]')

    columns = {
        'timestamp': pa.array(timestamps),
        'mains': pa.array(read_column(raw_fp, 'mains'), from_pandas=True),
    }
    for no in APPLIANCE_MAP[meter_id].values():
        col_name, values = get_plug_data(raw_fp, no, meter_id)
        columns[col_name] = pa.array(values, from_pandas=True)

    return pa.table(columns), day

def preprocess_user(meter_dir: str) -> float:
    meter_id = os.path.basename(meter_dir)
    day_fps = sorted(glob.glob(os.path.join(meter_dir, '*.csv')))
    if not day_fps:
        return 0.0

    out_dir = PREPROCESS_DIR / 'mains' / meter_id
    os.makedirs(out_dir, exist_ok=True)

    meter_max = 0.0
    for raw_fp in day_fps:
        tbl, day = load_day(raw_fp, meter_id)
        pq.write_table(tbl, out_dir / f'{day}.parquet', compression='snappy')

        day_max = pc.max(tbl.column('mains')).as_py()
        meter_max = max(meter_max, day_max or 0.0)

    print(f'{meter_id}: {len(day_fps):,} days -> {out_dir}')
    return meter_max

def preprocess_binary_vector(boundary_list: np.ndarray):
    for tbl, meter_id, day in load_data(PREPROCESS_DIR / 'mains'):
        out_dir = PREPROCESS_DIR / 'appliances' / meter_id
        os.makedirs(out_dir, exist_ok=True)

        timestamps = tbl.column('timestamp')

        dropped = tbl.drop_columns(['timestamp'])
        values = pa.table({
            name: pc.fill_null(dropped.column(name), 0)
            for name in dropped.column_names
        })
        values, num_levels, num_rows = ldp_energy_discretise(values, boundary_list)
        values = ldp_energy_encode_vectorise(values, num_levels, num_rows)

        out = pa.table({
            'timestamp': timestamps,
            **{name: values.column(name) for name in values.column_names}
        })
        pq.write_table(out, out_dir / f'{day}.parquet', compression='snappy')
        print(f'{meter_id}/{day}: {out.num_columns - 1} bit columns -> {out_dir}')

def main():
    global_max = 0.0
    for meter_dir in sorted(glob.glob(os.path.join(RAW_DIR, 'sm_*'))):
        global_max = max(global_max, preprocess_user(meter_dir))
    print('global mains max:', global_max)

    boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
    preprocess_binary_vector(boundary_list)

if __name__ == "__main__":
    main()
