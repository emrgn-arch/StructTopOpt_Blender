"""
Dependency availability check for the Blender Extensions build.

When distributed via extensions.blender.org (or any blender_manifest.toml
package), scipy is declared in the `wheels` field and Blender installs it
before the addon is loaded — no pip subprocess needed.

For GitHub source installs, users must ensure scipy is available in Blender's
Python environment separately.
"""

import importlib


def scipy_available():
    """Return True if scipy is importable in the current Python environment."""
    try:
        importlib.import_module("scipy")
        return True
    except ImportError:
        return False


def ensure_dependencies():
    """
    Check that scipy is available. Returns (ok: bool, error_str: str).

    With the Extensions build, scipy is pre-installed via bundled wheels.
    If missing, the extension was not installed correctly.
    """
    if scipy_available():
        return True, ""
    return False, (
        "scipy is not available. "
        "If installed from extensions.blender.org, try disabling and re-enabling the addon. "
        "For manual installs, ensure scipy is installed in Blender's Python environment."
    )
