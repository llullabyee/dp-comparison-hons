'''
this script preprocesses the data from the eco dataset. It takes the the sm and occ files in `data/eco/raw/`,
and outputs to `data/eco/preprocess/`. For every csv in `/preprocess/`, it will contain one column of power
measurements (the `powerallphases` column in sm), and one column of occupancy data, with 86400 records in each.'''
import os, glob, datetime
import pandas as pd


def get_plug_data(filepath: str, appliance_no: str, existing_df: pd.DataFrame):
    dirname, filename = os.path.split(filepath)
    parts = dirname.split(os.sep)
    parts = ['plug' if p.startswith('sm_') else p for p in parts]
    new_dir = os.sep.join(parts)
    new_filepath = os.path.join(new_dir, appliance_no, filename)
    if not os.path.exists(new_filepath):
        series = pd.Series([])
        return pd.concat([existing_df, series], axis=1)
    return pd.concat([existing_df, pd.read_csv(new_filepath, usecols=[0], header=None)], axis=1) ### remove the pd.concat here and make it a list in the main function. calling it creates new dataframes which will suck up memory



def preprocess(raw_dir: str, process_dir: str, globstring: str, input_col: int, output_col: int):
    for dir in glob.glob(os.path.join(raw_dir, 'sm_*')):
        for raw_filepath in glob.glob(os.path.join(raw_dir, '*.csv')):
            filename = os.path.basename(raw_filepath)
            process_filepath = os.path.join(process_dir, filename)
            df = pd.read_csv(raw_filepath, usecols=[0])
            df = pd.concat


def main():
    pass

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(UTILS_DIR, '..', 'data'))

get_plug_data()

# if __name__ == "__main__":
#    main()
