"""Download complete remote shards sequentially, without decoding anything."""

import argparse
from urllib.parse import urlparse

import requests
from huggingface_hub import get_token

from .common import path, read_args, shard_urls


def download_shard(url, config, token=None, reuse_shards=False):
    """Download one complete shard, or return its existing local path."""
    parsed = urlparse(url)
    filename = parsed.path.rsplit("/", 1)[-1]
    if parsed.scheme not in {"http", "https"} or filename in {"", ".", ".."}:
        raise ValueError(f"Invalid shard URL: {url}")

    destination = path(config, "data_dir", "shards")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename

    # Reuse is explicit so a cached shard cannot silently change a download run.
    if reuse_shards:
        if not target.exists():
            raise FileNotFoundError(
                f"Local shard missing: {target}. Run without --reuse-shards first."
            )
        print(f"Using local shard: {filename}", flush=True)
        return target

    # Send the Hugging Face token only to Hugging Face, never to another host.
    headers = {}
    if token and parsed.hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {token}"

    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {filename}", flush=True)

    # Stream small chunks to disk instead of holding the whole shard in RAM.
    # Rename only after completion so the final path always contains a full file.
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    temporary.replace(target)
    print(f"Saved {target.stat().st_size / 1e6:.1f} MB", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-shards",
        action="store_true",
        help="Skip existing shard files",
    )
    config, args = read_args(parser)
    token = get_token()
    filenames = set()

    # V0 still downloads every shard in order before the decode stage starts.
    for url in shard_urls(config):
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        if filename in filenames:
            raise ValueError(f"Shard filenames must be unique: {filename}")
        filenames.add(filename)
        download_shard(url, config, token, args.reuse_shards)


if __name__ == "__main__":
    main()
