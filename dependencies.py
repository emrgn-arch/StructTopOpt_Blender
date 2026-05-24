"""
Auto-install Python dependencies that Blender doesn't bundle.

Blender's Python excludes the Windows user site-packages directory
(AppData/Roaming/Python/...) from sys.path. When pip can't write to
Blender's own site-packages (no admin rights) it falls back there —
so we must add it to sys.path ourselves before checking / importing.
"""

import importlib
import os
import site
import subprocess
import sys


# (import_name, pip_install_name)
REQUIRED = [
    ("scipy", "scipy"),
]


def _add_user_site_to_path():
    try:
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
    except Exception:
        pass


def _python_exe():
    return os.path.join(sys.prefix, "bin", "python.exe")


def _all_available():
    _add_user_site_to_path()
    importlib.invalidate_caches()
    for import_name, _ in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            return False
    return True


def scipy_available():
    """Keep this name — called from UI checks."""
    return _all_available()


def ensure_dependencies():
    """Install any missing packages. Returns (already_ok, error_str)."""
    if _all_available():
        return True, ""

    missing_pips = []
    for import_name, pip_name in REQUIRED:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_pips.append(pip_name)

    python = _python_exe()
    try:
        subprocess.call([python, "-m", "ensurepip"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.call([python, "-m", "pip", "install", "--upgrade", "pip"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = subprocess.call(
            [python, "-m", "pip", "install", "--user"] + missing_pips,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if ret != 0:
            return False, f"pip exited with code {ret}"
        return False, ""
    except Exception as e:
        return False, str(e)


