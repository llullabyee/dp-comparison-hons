from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import RESULT_DIR


def make_run_dir(epsilon, window_size, peak, domain_max):
    run_dir = (
        RESULT_DIR
        / f"eps={epsilon:g}"
        / f"w={window_size}"
        / f"peak={peak:g}"
        / f"sens={domain_max:g}"
    )
    mains_dir = run_dir / "main"
    apps_dir = run_dir / "appliances"
    gai_dir = run_dir / "gai_agg"
    ldp_energy_dir = run_dir / "ldp_energy_agg"
    return mains_dir, apps_dir, gai_dir, ldp_energy_dir


# def load_data(directory: str|Path) -> tuple[pd.DataFrame, str, str]:
#     for fp in sorted(Path(directory).glob('*.parquet')):
#         yield pd.read_parquet(fp), fp.stem, fp.name


def load_data(fp: Path) -> pa.Table:
    return pq.read_table(fp)
