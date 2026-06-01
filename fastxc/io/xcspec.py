from __future__ import annotations

import struct

XCSPEC_MAGIC = b"FXCXSPEC"
XCSPEC_VERSION = 1
XCSPEC_ENDIAN_TAG = 0x01020304

XCSPEC_HEADER_SIZE = 256
XCSPEC_SOURCE_ENTRY_SIZE = 128
XCSPEC_PAYLOAD_ALIGNMENT = 4096

# payload[step][file][freq], where each value is float32 real/imag.
XCSPEC_LAYOUT_STEP_FILE_FREQ = 1
XCSPEC_DTYPE_COMPLEX64 = 1
XCSPEC_STRING_ENCODING_ASCII_NUL = 1

SEGSPEC_HEADER_STRUCT = struct.Struct("<ffiiffi")
COMPLEX64_BYTES = 8


def align_offset(value: int, alignment: int = XCSPEC_PAYLOAD_ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def pack_ascii(text: str, size: int) -> bytes:
    raw = text.encode("ascii")
    if len(raw) >= size:
        raise ValueError(f"fixed string is too long ({len(raw)} >= {size}): {text!r}")
    return raw + b"\0" * (size - len(raw))


def pack_header(
    *,
    timestamp: str,
    file_count: int,
    nstep: int,
    nspec: int,
    dt: float,
    df: float,
    source_table_offset: int,
    payload_offset: int,
    step_bytes: int,
    payload_bytes: int,
    manifest_hash_u64: int = 0,
) -> bytes:
    nfft = 2 * (nspec - 1)
    source_table_bytes = file_count * XCSPEC_SOURCE_ENTRY_SIZE
    header = bytearray(XCSPEC_HEADER_SIZE)

    struct.pack_into("<8s", header, 0, XCSPEC_MAGIC)
    struct.pack_into("<I", header, 8, XCSPEC_VERSION)
    struct.pack_into("<I", header, 12, XCSPEC_ENDIAN_TAG)
    struct.pack_into("<I", header, 16, XCSPEC_HEADER_SIZE)
    struct.pack_into("<I", header, 20, XCSPEC_SOURCE_ENTRY_SIZE)
    struct.pack_into("<Q", header, 24, source_table_offset)
    struct.pack_into("<I", header, 32, file_count)
    struct.pack_into("<I", header, 36, file_count)
    struct.pack_into("<Q", header, 40, payload_offset)
    struct.pack_into("<I", header, 48, XCSPEC_LAYOUT_STEP_FILE_FREQ)
    struct.pack_into("<I", header, 52, XCSPEC_DTYPE_COMPLEX64)
    struct.pack_into("<I", header, 56, XCSPEC_STRING_ENCODING_ASCII_NUL)
    struct.pack_into("<64s", header, 64, pack_ascii(timestamp, 64))
    struct.pack_into("<I", header, 128, nstep)
    struct.pack_into("<I", header, 132, nspec)
    struct.pack_into("<I", header, 136, nfft)
    struct.pack_into("<f", header, 144, float(dt))
    struct.pack_into("<f", header, 148, float(df))
    struct.pack_into("<Q", header, 152, step_bytes)
    struct.pack_into("<Q", header, 160, payload_bytes)
    struct.pack_into("<Q", header, 168, manifest_hash_u64)
    struct.pack_into("<Q", header, 176, source_table_bytes)
    return bytes(header)


def pack_source_entry(
    *,
    file_index: int,
    nsl_id: int,
    stla: float,
    stlo: float,
    network: str,
    station: str,
    location: str,
    component: str,
) -> bytes:
    entry = bytearray(XCSPEC_SOURCE_ENTRY_SIZE)
    struct.pack_into("<I", entry, 0, file_index)
    struct.pack_into("<I", entry, 4, nsl_id)
    struct.pack_into("<f", entry, 8, float(stla))
    struct.pack_into("<f", entry, 12, float(stlo))
    struct.pack_into("<16s", entry, 16, pack_ascii(network, 16))
    struct.pack_into("<32s", entry, 32, pack_ascii(station, 32))
    struct.pack_into("<16s", entry, 64, pack_ascii(location, 16))
    struct.pack_into("<16s", entry, 80, pack_ascii(component, 16))
    return bytes(entry)
