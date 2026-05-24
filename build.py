"""
Build the Blender extension zip for distribution.

Usage:
    python build.py

Requires `blender` to be on PATH (or edit BLENDER_EXE below).
Output zip is written to dist/.
"""

import subprocess
import sys
from pathlib import Path

BLENDER_EXE = "blender"
SOURCE_DIR = Path(__file__).parent / "source"
OUTPUT_DIR = Path(__file__).parent / "dist"

OUTPUT_DIR.mkdir(exist_ok=True)

result = subprocess.run(
    [
        BLENDER_EXE,
        "--command", "extension", "build",
        "--source-dir", str(SOURCE_DIR),
        "--output-dir", str(OUTPUT_DIR),
    ],
    check=False,
)

sys.exit(result.returncode)
