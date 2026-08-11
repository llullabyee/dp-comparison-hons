import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'eco'
RAW_DIR = DATA_DIR / 'raw'
PREPROCESS_DIR = DATA_DIR / 'preprocess'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
RESULT_DIR = PROJECT_ROOT / 'results'
SRC_DIR = PROJECT_ROOT / 'src'
UTIL_DIR = SRC_DIR / 'utils'

def load_config(config_path: str | Path = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
