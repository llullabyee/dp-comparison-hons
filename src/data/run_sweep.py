import itertools
import json
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import yaml

from src.utils.config import SWEEP_CONFIG_PATH

HEARTBEAT_URL = "https://health.llullabyee.dev/api/push/IZUbT8ukPP37i4XF8aSOBLfv26rBqx1R?status=up&msg=OK&ping="
PING_INTERVAL = 30
stop_event = threading.Event()


def heartbeat_loop():
    while not stop_event.is_set():
        try:
            requests.get(HEARTBEAT_URL, timeout=5)
        except requests.RequestException:
            pass
        stop_event.wait(PING_INTERVAL)


def send_failure_ping(msg: str):
    parsed = urlparse(HEARTBEAT_URL)
    query = parse_qs(parsed.query)
    query["status"] = ["down"]
    query["msg"] = [msg[:500]]  # keep it reasonably short
    new_query = urlencode(query, doseq=True)
    down_url = urlunparse(parsed._replace(query=new_query))
    try:
        requests.get(down_url, timeout=5)
    except requests.RequestException:
        pass


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

hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
hb_thread.start()
try:
    for combo in combos:
        params = dict(zip(keys, combo))
        tag = "_".join(f"{k}{v}" for k, v in params.items())
        config_path = configs_dir / f"{tag}.json"
        with open(config_path, "w") as f:
            json.dump(params, f)
        print(f"[{tag}] {params}")

        log_path = logs_dir / f"{tag}_preprocess.log"
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data.preprocess",
                    "--config",
                    str(config_path),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                msg = f"[{tag}] FAILED preprocess (exit {result.returncode})"
                print(f"  {msg} — see {log_path}")
                send_failure_ping(msg)
                continue  # skip mechanisms run if preprocessing failed

        # log_path = logs_dir / f"{tag}.log"
        # with open(log_path, "w") as log_file:
        #     result = subprocess.run(
        #         [
        #             sys.executable,
        #             "-m",
        #             "analysis.mechanisms",
        #             "--config",
        #             str(config_path),
        #         ],
        #         stdout=log_file,
        #         stderr=subprocess.STDOUT,
        #     )
        #     if result.returncode != 0:
        #         msg = f"[{tag}] FAILED mechanisms (exit {result.returncode})"
        #         print(f"  {msg} — see {log_path}")
        #         send_failure_ping(msg)
        #         continue

        # log_path = logs_dir / f"{tag}_utility.log"
        # with open(log_path, "w") as log_file:
        #     result = subprocess.run(
        #         [
        #             sys.executable,
        #             "-m",
        #             "analysis.utility",
        #             "--config",
        #             str(config_path),
        #         ],
        #         stdout=log_file,
        #         stderr=subprocess.STDOUT,
        #     )
        #     if result.returncode != 0:
        #         msg = f"[{tag}] FAILED preprocess (exit {result.returncode})"
        #         print(f"  {msg} — see {log_path}")
        #         send_failure_ping(msg)
        #         continue  # skip mechanisms run if preprocessing failed
finally:
    stop_event.set()
