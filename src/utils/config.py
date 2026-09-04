from pathlib import Path

PROJECT_ROOT = Path("/media/lullaby-nas")
CONFIG_DIR = PROJECT_ROOT / "config"
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
