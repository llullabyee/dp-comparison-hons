'''
Preprocesses the ECO dataset. Reads sm_* / plug_* files from data/eco/raw/ and
writes one parquet per household to data/eco/preprocess/, containing mains power,
per-appliance power, and timestamp/day columns across all days.
'''
import os
import glob
import pandas as pd

from src.utils.appliance_map import get_appliance_name, APPLIANCE_MAP
from src.utils.config import RAW_DIR, PREPROCESS_DIR, EPSILON, DOMAIN_MAX
from src.mechanisms.ldp_energy import concatenate_bits, ldp_energy_discretise, ldp_energy_encode_vectorise
from src.mechanisms.gai import build_boundary_list
from src.data.load_data import load_data


def get_plug_data(filepath: str, appliance_no: str, meter_id: str) -> pd.DataFrame:
    dirname, filename = os.path.split(filepath)
    parts = dirname.split(os.sep)
    parts = [p.replace('sm', 'plug') if p.startswith('sm_') else p for p in parts]
    new_filepath = os.path.join(os.sep.join(parts), appliance_no, filename)
    col_name = get_appliance_name(meter_id, appliance_no)
    if not os.path.exists(new_filepath):
        return pd.DataFrame(columns=[col_name], dtype='float32')
    return pd.read_csv(new_filepath, usecols=[0], header=None,
                       names=[col_name], dtype='float32')


def load_day(raw_fp: str, meter_id: str) -> pd.DataFrame:
    """One day-file for one household: mains + all appliance plugs."""
    mains = pd.read_csv(raw_fp, usecols=[0], header=None,
                        names=['mains'], dtype='float32')

    plugs = [get_plug_data(raw_fp, no, meter_id)
             for no in APPLIANCE_MAP[meter_id].values()]

    # reindex to mains so a ragged plug file can't change the row count
    df = pd.concat([mains] + plugs, axis=1).reindex(mains.index)

    day = os.path.splitext(os.path.basename(raw_fp))[0]      # '2012-06-01'
    df.insert(0, 'timestamp', pd.date_range(start=day, periods=len(df), freq='s'))
    df.insert(1, 'day', day)
    return df


def preprocess_user(meter_dir: str) -> int:
    meter_id = os.path.basename(meter_dir)
    day_fps = sorted(glob.glob(os.path.join(meter_dir, '*.csv')))
    if not day_fps:
        return 0

    df = pd.concat([load_day(fp, meter_id) for fp in day_fps], ignore_index=True)
    df = df.sort_values('timestamp', ignore_index=True)

    # --- reindex to a gap-free 1 Hz calendar -------------------------------
    start = df['timestamp'].min().normalize()
    end   = df['timestamp'].max().normalize() + pd.Timedelta('1D')
    full  = pd.date_range(start, end, freq='s', inclusive='left')

    n_before = len(df)
    df = (df.set_index('timestamp')
            .reindex(full)
            .rename_axis('timestamp')
            .reset_index())
    n_inserted = len(df) - n_before

    # rebuild `day` for inserted rows, then compress
    df['day'] = df['timestamp'].dt.strftime('%Y-%m-%d').astype('category')
    # ----------------------------------------------------------------------
    os.makedirs(PREPROCESS_DIR / 'mains', exist_ok=True)
    out_fp = os.path.join(PREPROCESS_DIR / 'mains', f'{meter_id}.parquet')
    df.to_parquet(out_fp, index=False, compression='snappy')

    print(f'{meter_id}: {len(df):,} rows '
          f'({n_inserted:,} inserted, {n_inserted/86400:.1f} days missing) -> {out_fp}')
    return int(df['mains'].max())


def preprocess_binary_vector(boundary_list: list):
    os.makedirs(PREPROCESS_DIR / 'appliances', exist_ok=True)
    for df, _, filename in load_data(PREPROCESS_DIR / 'mains'):
        df = df.fillna(0)
        print(filename)
        timestamps = df['timestamp']
        days = df['day']
        print('ive gotten this far')
        df, num_levels = ldp_energy_discretise(df, boundary_list)
        print('we cooked')
        df = ldp_energy_encode_vectorise(df, num_levels)
        print('surely')
        # bit_vectors = concatenate_bits(df)
        print('aint no way')
        df = pd.concat([timestamps, days, df], axis=1)
        out_fp = PREPROCESS_DIR / 'appliances' / filename
        df.to_parquet(out_fp, index=False, compression='snappy')


def main():
    global_max = 0
    for meter_dir in sorted(glob.glob(os.path.join(RAW_DIR, 'sm_*'))):
        global_max = max(global_max, preprocess_user(meter_dir))
    print('global mains max:', global_max)
    boundary_list = build_boundary_list(EPSILON, DOMAIN_MAX)
    preprocess_binary_vector(boundary_list)


if __name__ == "__main__":
    main()
