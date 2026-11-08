"""
Build the Blender extension zips for distribution.

Usage:
    python build.py

Output: dist/add-on-{id}-v{version}-{platform}.zip, one per platform listed in
blender_manifest.toml. The zips only differ in their manifest's `platforms`
entry; the add-on is pure Python and only needs the NumPy that ships with Blender.
"""

import pathlib
import re
import zipfile

SOURCE_DIR = pathlib.Path(__file__).parent / "source"
OUTPUT_DIR = pathlib.Path(__file__).parent / "dist"

SKIP_PARTS    = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".blend", ".zip", ".whl"}

_PLATFORMS_RE = re.compile(r'^platforms\s*=\s*\[(.*?)\]', re.MULTILINE | re.DOTALL)


def _get_version(manifest: str) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _get_id(manifest: str) -> str:
    m = re.search(r'^id\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    return m.group(1).replace("_", "-") if m else "extension"


def build() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    manifest  = (SOURCE_DIR / "blender_manifest.toml").read_text(encoding="utf-8")
    version   = _get_version(manifest)
    ext_id    = _get_id(manifest)
    platforms = re.findall(r'"([^"]+)"', _PLATFORMS_RE.search(manifest).group(1))

    files = []
    for f in sorted(SOURCE_DIR.rglob("*")):
        rel = f.relative_to(SOURCE_DIR)
        if (f.is_file() and not set(rel.parts) & SKIP_PARTS and f.suffix not in SKIP_SUFFIXES
                and rel != pathlib.Path("blender_manifest.toml")):
            files.append((f, rel))

    for platform in platforms:
        out = OUTPUT_DIR / f"add-on-{ext_id}-v{version}-{platform}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("blender_manifest.toml",
                        _PLATFORMS_RE.sub(f'platforms = ["{platform}"]', manifest))
            for f, rel in files:
                zf.write(f, rel)
        print(f"Created {out.name}")


if __name__ == "__main__":
    build()
