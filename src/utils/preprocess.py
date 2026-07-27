'''
this script preprocesses the data from the eco dataset. It takes the the sm and occ files in `data/eco/raw/`,
and outputs to `data/eco/preprocess/`. For every csv in `/preprocess/`, it will contain one column of power
measurements (the `powerallphases` column in sm), and one column of occupancy data, with 86400 records in each.'''
import os
import glob
import pandas as pd
from src.utils.appliance_map import get_appliance_name, APPLIANCE_MAP


def get_plug_data(filepath: str, appliance_no: str, meter_id: str):
    dirname, filename = os.path.split(filepath)
    parts = dirname.split(os.sep)
    parts = ['plug' if p.startswith('sm_') else p for p in parts]
    new_dir = os.sep.join(parts)
    new_filepath = os.path.join(new_dir, appliance_no, filename)
    col_name=get_appliance_name(meter_id, appliance_no)
    if not os.path.exists(new_filepath):
        return pd.DataFrame(columns=[col_name], dtype=float)
    return pd.read_csv(new_filepath, usecols=[0], header=None, names=[col_name])



def preprocess(raw_dir: str, process_dir: str):
    '''
    gotta probably put the for meter dir in main, will do later teehee
    '''
    for meter_dir in glob.glob(os.path.join(raw_dir, 'sm_*')):
        for raw_fp in glob.glob(os.path.join(meter_dir, '*.csv')):
            meter_id = os.path.basename(meter_dir)
            filename = os.path.basename(raw_fp)
            meter_process = os.path.join(process_dir, meter_id)
            os.makedirs(meter_process, exist_ok=True)
            process_fp = os.path.join(meter_process, filename)

            mains_df = pd.read_csv(raw_fp, usecols=[0], header=None, names=['power']) ### make list of read_csv and merge with concat?

            appliances = APPLIANCE_MAP[meter_id]
            appliance_df = []
            for appliance, appliance_no in appliances.items():
                appliance_df.append(get_plug_data(raw_fp,appliance_no,meter_id))

            df = pd.concat([mains_df] + appliance_df, axis=1)
            df.to_csv(process_fp, index=False)


def main():
    pass

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(UTILS_DIR, '..', 'data'))

# if __name__ == "__main__":
#    main()
