from pathlib import Path

import yaml

PROJECT_ROOT = Path("/media/lullaby-nas")
DATA_DIR = PROJECT_ROOT / "IDEAL"
MAINS_DIR = DATA_DIR / "household_sensors" / "electric_mains"
APPLIANCES_DIR = DATA_DIR / "room_and_appliance_sensors" / "appliances_electric"
PREPROCESS_MAINS_DIR = DATA_DIR / "preprocess" / "mains"
PREPROCESS_APPS_DIR = DATA_DIR / "preprocess" / "appliances"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
RESULT_DIR = PROJECT_ROOT / "DP_results"
PERTURBED_MAINS_DIR = RESULT_DIR / "mains"
PERTURBED_APPS_DIR = RESULT_DIR / "appliances"
SRC_DIR = PROJECT_ROOT / "src"
UTIL_DIR = SRC_DIR / "utils"


EPSILON = 1.0
DOMAIN_MAX = 3000
WINDOW_SIZE = 3
PEAK = 1500


def load_config(config_path: str | Path = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
