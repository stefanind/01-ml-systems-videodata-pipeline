"""Small helpers shared by the standalone stage scripts."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_args(parser=None):
    """Read the CLI configuration and remember its location for relative paths."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    args = parser.parse_args()
    config_file = Path(args.config).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["_config_file"] = config_file
    return config, args


def path(config, key, *parts):
    """Resolve a configured path relative to the config file, then add subpaths."""
    return config["_config_file"].parent / config[key] / Path(*parts)


def save_json(destination, value):
    """Write a readable JSON file, creating its parent directory if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(value, indent=2) + "\n"
    destination.write_text(json_text, encoding="utf-8")


def selection(config):
    """Load the ordered video records that decode wrote for later stages."""
    selection_file = path(config, "data_dir", "selection.json")
    return json.loads(selection_file.read_text(encoding="utf-8"))


def shard_urls(config):
    """Read shard URLs in file order, ignoring comments and empty lines."""
    lines = path(config, "shards_file").read_text(encoding="utf-8").splitlines()
    urls = []
    for line in lines:
        url = line.strip()
        if url and not url.startswith("#"):
            urls.append(url)

    if not urls:
        raise ValueError("shards.txt must contain at least one shard URL")
    return urls
