"""FastXC configuration parser public API."""

from importlib.metadata import PackageNotFoundError, version

from .loader import Config, ConfigError

try:
    __version__: str = version("fastxc")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "dev"

__all__ = ["Config", "ConfigError", "__version__"]
