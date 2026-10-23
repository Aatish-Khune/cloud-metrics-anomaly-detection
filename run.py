"""Cross-platform task runner (works on Windows without `make`).

    python run.py data    # download SMD subset + NAB AWS CloudWatch into data/raw/
    python run.py train   # train + evaluate both detectors, write reports/ and models/
    python run.py test    # lint + unit tests
    python run.py app     # launch the Streamlit demo
    python run.py all     # data + train
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main(cmd: str) -> int:
    if cmd in ("data", "all"):
        from anomaly.download import download_all

        download_all()
    if cmd in ("train", "all"):
        from anomaly.pipeline import run_all

        run_all()
    if cmd == "test":
        lint = subprocess.call([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT)
        return lint or subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if cmd == "app":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")],
            cwd=ROOT,
        )
    if cmd not in ("data", "train", "all"):
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
