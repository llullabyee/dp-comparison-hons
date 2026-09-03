import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


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


def load_data(fp: Path) -> pa.Table:
    return pq.read_table(fp)
