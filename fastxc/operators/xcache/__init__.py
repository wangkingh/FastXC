"""Build timestamp-level XC input cache shards from SEGSPEC files."""

from .builder import (
    build_xcache,
)
from .async_builder import (
    AsyncXCacheMaterializer,
    XCacheBuildResult,
)
from fastxc.io.xcspec import (
    XCSPEC_DTYPE_COMPLEX64,
    XCSPEC_HEADER_SIZE,
    XCSPEC_LAYOUT_STEP_FILE_FREQ,
    XCSPEC_MAGIC,
    XCSPEC_SOURCE_ENTRY_SIZE,
    XCSPEC_VERSION,
)

__all__ = [
    "XCSPEC_DTYPE_COMPLEX64",
    "XCSPEC_HEADER_SIZE",
    "XCSPEC_LAYOUT_STEP_FILE_FREQ",
    "XCSPEC_MAGIC",
    "XCSPEC_SOURCE_ENTRY_SIZE",
    "XCSPEC_VERSION",
    "AsyncXCacheMaterializer",
    "XCacheBuildResult",
    "build_xcache",
]
