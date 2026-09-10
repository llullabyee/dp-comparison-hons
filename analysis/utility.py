import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.load_data import load_data, make_dirs
from src.mechanisms.gai import build_boundary_list
from src.utils.config import RESULT_DIR


def 


leaf_dirs = []
for dirpath, dirnames, filenames in os.walk(RESULT_DIR):
    if not dirnames:  # no subdirectories -> this is a leaf
        leaf_dirs.append(Path(dirpath))
