"""Reusable implementations for FastXC tool-style CLI commands."""

from importlib import import_module

_EXPORTS = {
    "SpackDecodeResult": ("fastxc.tools.decode_spack", "SpackDecodeResult"),
    "XCacheInspection": ("fastxc.tools.inspect_xcache", "XCacheInspection"),
    "convert_sac_dir": ("fastxc.tools.sac2dat", "convert_sac_dir"),
    "decode_spack": ("fastxc.tools.decode_spack", "decode_spack"),
    "extract_bigsac": ("fastxc.tools.extract", "extract_bigsac"),
    "extract_bigsac_dir": ("fastxc.tools.extract", "extract_bigsac_dir"),
    "inspect_xcache": ("fastxc.tools.inspect_xcache", "inspect_xcache"),
    "sac_to_dat": ("fastxc.tools.sac2dat", "sac_to_dat"),
}

__all__ = [
    "SpackDecodeResult",
    "XCacheInspection",
    "convert_sac_dir",
    "decode_spack",
    "extract_bigsac",
    "extract_bigsac_dir",
    "inspect_xcache",
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
