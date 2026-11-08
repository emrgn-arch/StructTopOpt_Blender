"""Solve result cache — the latest density field and the problem it was solved on."""

_cached_density = None
_cached_problem = None


def cache_result(problem, density):
    global _cached_density, _cached_problem
    _cached_density = density
    _cached_problem = problem


def get_cached_density():
    return _cached_density


def get_cached_problem():
    return _cached_problem
