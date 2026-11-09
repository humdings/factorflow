"""Docstring coverage and the README quickstart stay in sync."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_embeds_quickstart():
    script = (ROOT / "examples" / "quickstart.py").read_text().strip()
    readme = (ROOT / "README.md").read_text()
    assert script in readme


def test_public_docstring_coverage():
    proc = subprocess.run(
        [sys.executable, "-m", "interrogate", "src/factorflow"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
