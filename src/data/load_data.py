import os
import glob
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import re


def make_meter_dir(parts, filename):
    if isinstance(parts, (str, os.PathLike)):
        meter_dir = os.path.join(parts)
    else:
        meter_dir = os.path.join(*parts)
    os.makedirs(meter_dir, exist_ok=True)
    meter_fp = os.path.join(meter_dir, filename)
    return meter_dir, meter_fp


# def load_data(directory, subfolder=None):
#     pattern = re.compile(r'^\d{4}-\d{2}-\d{2}\.csv$')
#     if isinstance(directory, (str, os.PathLike)):
#         base = directory
#     else:
#         base = os.path.join(*directory)
#     for meter_dir in sorted(glob.glob(os.path.join(base, 'sm_*'))):
#         meter_id = os.path.basename(meter_dir)
#         search_dir = os.path.join(meter_dir, subfolder) if subfolder else meter_dir
#         for filepath in sorted(glob.glob(os.path.join(search_dir, '*.csv'))):
#             filename = os.path.basename(filepath)
#             if not pattern.match(filename):
#                 yield pd.read_csv(filepath), meter_id, filename
#                 continue
#             day = datetime.strptime(filename, '%Y-%m-%d.csv').date()
#             yield pd.read_csv(filepath), meter_id, filename, day

def load_data(preprocess_dir):
    for fp in sorted(Path(preprocess_dir).glob('*.parquet')):
        yield pd.read_parquet(fp), fp.stem, fp.name
