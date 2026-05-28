"""
Build platform-specific Blender extension zips for distribution.

Usage:
    python build.py

Output: dist/add-on-{id}-v{version}-{platform}.zip  (one per OS)
Each zip gets a patched blender_manifest.toml listing only that OS's wheels.
Wheels are auto-detected from source/wheels/ by filename pattern — just drop
in cp313 (or any future Python version) wheels and they will be included.
"""

import pathlib
import re
import zipfile

SOURCE_DIR = pathlib.Path(__file__).parent / "source"
OUTPUT_DIR = pathlib.Path(__file__).parent / "dist"
WHEELS_DIR = SOURCE_DIR / "wheels"

# Map Blender platform tag → wheel filename predicate
PLATFORMS: dict[str, object] = {
    "windows-x64": lambda n: "win_amd64" in n,
    "linux-x64":   lambda n: "manylinux" in n,
    "macos-arm64": lambda n: "macosx" in n and "arm64" in n,
}

SKIP_PARTS    = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".blend", ".zip"}


def _get_version(manifest: str) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _get_id(manifest: str) -> str:
    m = re.search(r'^id\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    return m.group(1).replace("_", "-") if m else "extension"


def _patch_manifest(manifest: str, platform: str, wheels: list[str]) -> str:
    manifest = re.sub(
        r'^platforms\s*=\s*\[.*?\]',
        f'platforms = ["{platform}"]',
        manifest, flags=re.MULTILINE,
    )
    wheel_lines = "\n".join(f'    "./wheels/{w}",' for w in wheels)
    manifest = re.sub(
        r'^wheels\s*=\s*\[.*?\]',
        f'wheels = [\n{wheel_lines}\n]',
        manifest, flags=re.MULTILINE | re.DOTALL,
    )
    return manifest


def build() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    manifest_src = (SOURCE_DIR / "blender_manifest.toml").read_text(encoding="utf-8")
    version      = _get_version(manifest_src)
    ext_id       = _get_id(manifest_src)
    all_wheels   = sorted(f.name for f in WHEELS_DIR.glob("*.whl"))

    if not all_wheels:
        print("WARNING: no .whl files found in source/wheels/ — zips will have no bundled dependency")

    for platform, predicate in PLATFORMS.items():
        wheels = [w for w in all_wheels if predicate(w)]
        if not wheels:
            print(f"WARNING: no wheels matched for {platform}, skipping")
            continue

        patched = _patch_manifest(manifest_src, platform, wheels)
        out     = OUTPUT_DIR / f"add-on-{ext_id}-v{version}-{platform}.zip"

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            # Patched manifest goes in at the zip root
            zf.writestr("blender_manifest.toml", patched)

            # All other source files (excluding wheels dir and original manifest)
            for f in SOURCE_DIR.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(SOURCE_DIR)
                if set(rel.parts) & SKIP_PARTS or f.suffix in SKIP_SUFFIXES:
                    continue
                if rel.parts[0] == "wheels" or rel == pathlib.Path("blender_manifest.toml"):
                    continue
                zf.write(f, rel)

            # Only this platform's wheels
            for wheel_name in wheels:
                zf.write(WHEELS_DIR / wheel_name, f"wheels/{wheel_name}")

        print(f"Created {out.name}")
        for w in wheels:
            print(f"  + {w}")


if __name__ == "__main__":
    build()
