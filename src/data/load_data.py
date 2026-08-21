import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
import re


def make_dirs(parts, filename):
    if isinstance(parts, (str, os.PathLike)):
        dir = os.path.join(parts)
    else:
        dir = os.path.join(*parts)
    os.makedirs(dir, exist_ok=True)
    fp = os.path.join(dir, filename)
    return dir, fp

# def load_data(directory: str|Path) -> tuple[pd.DataFrame, str, str]:
#     for fp in sorted(Path(directory).glob('*.parquet')):
#         yield pd.read_parquet(fp), fp.stem, fp.name

def load_data(directory: str | Path) -> Iterator[tuple[pa.Table, str, str]]:
    """Yield (table, meter, day) — one day-file at a time."""
    for fp in sorted(Path(directory).glob('*/*.parquet')):
        yield pq.read_table(fp), fp.parent.name, fp.stem
