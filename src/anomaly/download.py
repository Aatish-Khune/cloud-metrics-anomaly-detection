"""Download the SMD subset and NAB AWS CloudWatch files into data/raw/ over HTTPS.

Only the individual data files are fetched (no git clone). Existing files are skipped.
"""

from __future__ import annotations

from pathlib import Path

import requests

from anomaly.config import (
    NAB_AWS_SERIES,
    NAB_BASE_URL,
    RAW_DIR,
    SMD_BASE_URL,
    SMD_MACHINES,
)


def file_list() -> list[tuple[str, Path]]:
    """Return (url, local_path) pairs for every file the project needs."""
    files: list[tuple[str, Path]] = []
    for machine in SMD_MACHINES:
        for split in ("train", "test", "test_label"):
            url = f"{SMD_BASE_URL}/{split}/{machine}.txt"
            files.append((url, RAW_DIR / "smd" / split / f"{machine}.txt"))
    for series in NAB_AWS_SERIES:
        url = f"{NAB_BASE_URL}/data/realAWSCloudwatch/{series}.csv"
        files.append((url, RAW_DIR / "nab" / f"{series}.csv"))
    files.append(
        (f"{NAB_BASE_URL}/labels/combined_windows.json", RAW_DIR / "nab" / "combined_windows.json")
    )
    return files


def download(url: str, dest: Path, timeout: int = 60) -> int:
    """Stream one file to disk. Returns bytes written (0 if it already existed)."""
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
    tmp.replace(dest)
    return written


def download_all(verbose: bool = True) -> None:
    """Download every file in `file_list()`."""
    total = 0
    for url, dest in file_list():
        n = download(url, dest)
        total += n
        if verbose:
            status = "skip (exists)" if n == 0 else f"{n / 1e6:.2f} MB"
            print(f"{status:>14}  {dest.relative_to(RAW_DIR)}")
    if verbose:
        print(f"Downloaded {total / 1e6:.1f} MB into {RAW_DIR}")


if __name__ == "__main__":
    download_all()
