'''
this script preprocesses the data from the eco dataset. It takes the the sm and occ files in `data/eco/raw/`,
and outputs to `data/eco/preprocess/`. For every csv in `/preprocess/`, it will contain one column of power
measurements (the `powerallphases` column in sm), and one column of occupancy data, with 86400 records in each.'''
import os
import glob
import pandas as pd
from src.utils.appliance_map import get_appliance_name, APPLIANCE_MAP


DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'eco'))
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PREPROCESS_DIR = os.path.join(DATA_DIR, 'preprocess')


def get_plug_data(filepath: str, appliance_no: str, meter_id: str):
    dirname, filename = os.path.split(filepath)
    parts = dirname.split(os.sep)
    parts = [p.replace('sm', 'plug') if p.startswith('sm_') else p for p in parts]
    print(parts)
    new_dir = os.sep.join(parts)
    new_filepath = os.path.join(new_dir, appliance_no, filename)
    col_name=get_appliance_name(meter_id, appliance_no)
    if not os.path.exists(new_filepath):
        print("hello")
        return pd.DataFrame(columns=[col_name], dtype=float)
    return pd.read_csv(new_filepath, usecols=[0], header=None, names=[col_name])



def preprocess(raw_fp: str):
    '''
    gotta probably put the for meter dir in main, will do later teehee
    '''
    meter_id = os.path.basename(os.path.dirname(raw_fp))
    filename = os.path.basename(raw_fp)
    meter_process = os.path.join(PREPROCESS_DIR, meter_id)
    os.makedirs(meter_process, exist_ok=True)
    process_fp = os.path.join(meter_process, filename)

    mains_df = pd.read_csv(raw_fp, usecols=[0], header=None, names=['power']) ### make list of read_csv and merge with concat?

    appliances = APPLIANCE_MAP[meter_id]
    appliance_df = []
    for appliance_no in appliances.values():
        appliance_df.append(get_plug_data(raw_fp,appliance_no,meter_id))

    df = pd.concat([mains_df] + appliance_df, axis=1)
    df.to_csv(process_fp, index=False)

def main():
    for meter_dir in glob.glob(os.path.join(RAW_DIR, 'sm_*')):
        for raw_fp in glob.glob(os.path.join(meter_dir, '*.csv')):
            preprocess(raw_fp)

if __name__ == "__main__":
    main()
