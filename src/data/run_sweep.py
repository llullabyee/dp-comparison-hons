import itertools
import json
import subprocess
from pathlib import Path

import yaml

from src.utils.config import CONFIG_DIR

SWEEP_CONFIG_PATH = CONFIG_DIR / "sweep.yaml"

with open(SWEEP_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

sweep = config["sweep"]
keys = list(sweep.keys())
combos = list(itertools.product(*sweep.values()))

configs_dir = Path("sweep_configs")
configs_dir.mkdir(exist_ok=True)

logs_dir = Path("sweep_logs")
logs_dir.mkdir(exist_ok=True)

print(f"Running {len(combos)} combinations")

for combo in combos:
    params = dict(zip(keys, combo))
    tag = "_".join(f"{k}{v}" for k, v in params.items())

    config_path = configs_dir / f"{tag}.json"
    with open(config_path, "w") as f:
        json.dump(params, f)

    print(f"[{tag}] {params}")
    log_path = logs_dir / f"{tag}.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["python", "-m", "analysis.mechanisms", "--config", str(config_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) — see {log_path}")
