"""Density result cache."""

_cached_density = None


def cache_density(density):
    global _cached_density
    _cached_density = density


def get_cached_density():
    return _cached_density
