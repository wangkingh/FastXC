"""Reusable implementations for FastXC tool-style CLI commands."""

from importlib import import_module

_EXPORTS = {
    "convert_sac_dir": ("fastxc.tools.sac2dat", "convert_sac_dir"),
    "extract_bigsac": ("fastxc.tools.extract", "extract_bigsac"),
    "extract_bigsac_dir": ("fastxc.tools.extract", "extract_bigsac_dir"),
    "sac_to_dat": ("fastxc.tools.sac2dat", "sac_to_dat"),
}

__all__ = [
    "convert_sac_dir",
    "extract_bigsac",
    "extract_bigsac_dir",
    "sac_to_dat",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
