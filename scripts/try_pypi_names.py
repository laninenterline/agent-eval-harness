"""Try distinct PyPI distribution names until one passes PyPI's similarity check."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["aehcli", "aehx", "aehctl", "aehkit", "runproof", "evalproof"]
PYPROJECT = ROOT / "pyproject.toml"


def set_name(name: str) -> None:
    text = PYPROJECT.read_text()
    text = re.sub(r'^name = ".*"$', f'name = "{name}"', text, count=1, flags=re.M)
    PYPROJECT.write_text(text)


def main() -> int:
    for name in CANDIDATES:
        set_name(name)
        for junk in ("dist", "build"):
            shutil.rmtree(ROOT / junk, ignore_errors=True)
        for egg in (ROOT / "src").glob("*.egg-info"):
            shutil.rmtree(egg, ignore_errors=True)

        build = subprocess.run([sys.executable, "-m", "build"], cwd=ROOT,
                               capture_output=True, text=True)
        if build.returncode != 0:
            print(f"[{name}] build failed: {build.stderr[-300:]}")
            continue

        upload = subprocess.run(
            [sys.executable, "-m", "twine", "upload", "--non-interactive", *map(str, (ROOT / "dist").glob("*"))],
            cwd=ROOT, capture_output=True, text=True,
        )
        combined = upload.stdout + upload.stderr
        if "View at" in combined or upload.returncode == 0:
            print(f"[{name}] UPLOADED")
            print(combined[-600:])
            (ROOT / ".pypi_name").write_text(name)
            return 0
        reason = [line for line in combined.splitlines() if "too similar" in line or "ERROR" in line]
        print(f"[{name}] rejected: {' | '.join(reason[:2])}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
