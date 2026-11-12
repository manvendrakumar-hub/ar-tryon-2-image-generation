"""fal client wrapper (adapted from AR Try On/scripts/fal_setup.py).

truststore must patch ssl at module level: the corporate proxy re-signs TLS and
Python's certifi bundle rejects it otherwise.
"""

import os
import pathlib
import urllib.request

import truststore

truststore.inject_into_ssl()

import fal_client  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SECRET_FILES = (ROOT / ".env",)


def ensure_key() -> None:
    if os.environ.get("FAL_KEY"):
        return
    for path in SECRET_FILES:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip().startswith("FAL_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    os.environ["FAL_KEY"] = value
                    return
    raise RuntimeError("FAL_KEY not set: export it, or put FAL_KEY=... in .env at the repo root")


def upload(path) -> str:
    ensure_key()
    return fal_client.upload_file(str(path))


def run(endpoint: str, arguments: dict) -> dict:
    ensure_key()
    return fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)


def download(url: str, dest) -> pathlib.Path:
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())
    return dest
