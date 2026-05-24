"""Density result cache and preview color update."""

import bpy
import numpy as np

from .preview import PREVIEW_NAME, _COLOR_SUPPORT, _COLOR_LOAD, _COLOR_KEEP

_cached_density = None


def cache_density(density):
    global _cached_density
    _cached_density = density


def get_cached_density():
    return _cached_density
