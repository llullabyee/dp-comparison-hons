import json
import sys
from pathlib import Path


def load_params():
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        with open(sys.argv[idx + 1]) as f:
            return json.load(f)
    raise ValueError("No config provided")


DATA_DIR = Path(r"Y:\IDEAL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
MAINS_DIR = DATA_DIR / "household_sensors" / "electric_mains"
APPLIANCES_DIR = DATA_DIR / "room_and_appliance_sensors" / "appliances_electric"
PREPROCESS_DIR = DATA_DIR / "preprocess"
PREPROCESS_MAINS_DIR = PREPROCESS_DIR / "mains"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
RESULT_DIR = DATA_DIR / "DP_results"
PERTURBED_MAINS_DIR = RESULT_DIR / "mains"
PERTURBED_APPS_DIR = RESULT_DIR / "appliances"
SRC_DIR = PROJECT_ROOT / "src"
UTIL_DIR = SRC_DIR / "utils"
