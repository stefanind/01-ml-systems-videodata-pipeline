"""Small helpers shared by the standalone stage scripts."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_args(parser=None):
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    config_file = Path(args.config).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["_config_file"] = config_file
    return config, args


def path(config, key, *parts):
    return config["_config_file"].parent / config[key] / Path(*parts)


def save_json(destination, value):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def selection(config):
    return json.loads(path(config, "data_dir", "selection.json").read_text(encoding="utf-8"))


def shard_urls(config):
    lines = path(config, "shards_file").read_text(encoding="utf-8").splitlines()
    urls = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not urls:
        raise ValueError("shards.txt must contain at least one shard URL")
    return urls
