import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: str | Path = None) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


