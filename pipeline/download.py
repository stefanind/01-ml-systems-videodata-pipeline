"""Download complete remote shards sequentially, without decoding anything."""

import argparse
from urllib.parse import urlparse

import requests
from huggingface_hub import get_token

from .common import path, read_args, shard_urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-shards", action="store_true", help="Skip existing shard files")
    config, args = read_args(parser)
    destination = path(config, "data_dir", "shards")
    destination.mkdir(parents=True, exist_ok=True)
    token = get_token()
    filenames = set()
    for url in shard_urls(config):
        parsed = urlparse(url)
        filename = parsed.path.rsplit("/", 1)[-1]
        if parsed.scheme not in {"http", "https"} or filename in {"", ".", ".."}:
            raise ValueError(f"Invalid shard URL: {url}")
        if filename in filenames:
            raise ValueError(f"Shard filenames must be unique: {filename}")
        filenames.add(filename)
        target = destination / filename
        if args.reuse_shards:
            if not target.exists():
                raise FileNotFoundError(f"Local shard missing: {target}. Run without --reuse-shards first.")
            print(f"Using local shard: {filename}", flush=True)
            continue
        headers = {"Authorization": f"Bearer {token}"} if token and parsed.hostname == "huggingface.co" else {}
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {filename}", flush=True)
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        temporary.replace(target)
        print(f"Saved {target.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
