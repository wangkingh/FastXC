from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

SAC_HEADER_BYTES = 632
SAC_NUMERIC_BYTES = 440
SAC_FLOAT_COUNT = 70
SAC_INT_COUNT = 40

ITIME = 1
IXY = 4

_NATIVE_ENDIAN = "<" if sys.byteorder == "little" else ">"

_FLOAT_FIELDS: dict[str, int] = {
    "delta": 0,
    "depmin": 1,
    "depmax": 2,
    "scale": 3,
    "odelta": 4,
    "b": 5,
    "e": 6,
    "o": 7,
    "a": 8,
    "stla": 31,
    "stlo": 32,
    "stel": 33,
    "stdp": 34,
    "evla": 35,
    "evlo": 36,
    "evel": 37,
    "evdp": 38,
    "user0": 40,
    "user1": 41,
    "user2": 42,
    "user3": 43,
    "user4": 44,
    "user5": 45,
    "user6": 46,
    "user7": 47,
    "user8": 48,
    "user9": 49,
    "dist": 50,
    "az": 51,
    "baz": 52,
    "gcarc": 53,
    "depmen": 56,
    "cmpaz": 57,
    "cmpinc": 58,
    "scmpaz": 68,
    "scmpinc": 69,
}

_INT_FIELDS: dict[str, int] = {
    "nzyear": 0,
    "nzjday": 1,
    "nzhour": 2,
    "nzmin": 3,
    "nzsec": 4,
    "nzmsec": 5,
    "nvhdr": 6,
    "npts": 9,
    "iftype": 15,
    "leven": 35,
    "lpspol": 36,
    "lovrok": 37,
    "lcalda": 38,
    "unused27": 39,
}

_STR_FIELDS: dict[str, tuple[int, int]] = {
    "kstnm": (440, 8),
    "kevnm": (448, 16),
    "khole": (464, 8),
    "ko": (472, 8),
    "ka": (480, 8),
    "kt0": (488, 8),
    "kt1": (496, 8),
    "kt2": (504, 8),
    "kt3": (512, 8),
    "kt4": (520, 8),
    "kt5": (528, 8),
    "kt6": (536, 8),
    "kt7": (544, 8),
    "kt8": (552, 8),
    "kt9": (560, 8),
    "kf": (568, 8),
    "kuser0": (576, 8),
    "kuser1": (584, 8),
    "kuser2": (592, 8),
    "kcmpnm": (600, 8),
    "knetwk": (608, 8),
    "kdatrd": (616, 8),
    "kinst": (624, 8),
}


def _detect_endian(raw: bytes) -> str:
    if len(raw) != SAC_HEADER_BYTES:
        raise ValueError(f"SAC header must be {SAC_HEADER_BYTES} bytes, got {len(raw)}")

    nvhdr_off = SAC_FLOAT_COUNT * 4 + _INT_FIELDS["nvhdr"] * 4
    npts_off = SAC_FLOAT_COUNT * 4 + _INT_FIELDS["npts"] * 4
    little_nvhdr = struct.unpack_from("<i", raw, nvhdr_off)[0]
    big_nvhdr = struct.unpack_from(">i", raw, nvhdr_off)[0]
    if little_nvhdr in (6, 7):
        return "<"
    if big_nvhdr in (6, 7):
        return ">"

    little_npts = struct.unpack_from("<i", raw, npts_off)[0]
    big_npts = struct.unpack_from(">i", raw, npts_off)[0]
    little_delta = struct.unpack_from("<f", raw, 0)[0]
    big_delta = struct.unpack_from(">f", raw, 0)[0]

    little_ok = 0 < little_npts < 1_000_000_000 and np.isfinite(little_delta)
    big_ok = 0 < big_npts < 1_000_000_000 and np.isfinite(big_delta)
    if little_ok and not big_ok:
        return "<"
    if big_ok and not little_ok:
        return ">"
    return _NATIVE_ENDIAN


def _field_index(name_or_index: str | int, table: dict[str, int]) -> int:
    if isinstance(name_or_index, int):
        return name_or_index
    try:
        return table[name_or_index]
    except KeyError as exc:
        raise KeyError(f"Unknown SAC header field: {name_or_index}") from exc


@dataclass
class SacHeader:
    """Small mutable wrapper around a binary SAC header.

    The object preserves the original 632-byte header and only rewrites fields
    that callers explicitly modify. This keeps Python output close to the
    historical C `sacio.c` behavior while avoiding an ObsPy dependency.
    """

    raw: bytearray
    endian: str = _NATIVE_ENDIAN

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SacHeader":
        return cls(bytearray(raw), _detect_endian(raw))

    @classmethod
    def empty(cls, *, endian: str = _NATIVE_ENDIAN) -> "SacHeader":
        raw = bytearray(SAC_HEADER_BYTES)
        hd = cls(raw, endian)
        hd.set_int("nvhdr", 6)
        hd.set_int("iftype", ITIME)
        hd.set_int("leven", 1)
        return hd

    def copy(self) -> "SacHeader":
        return SacHeader(bytearray(self.raw), self.endian)

    def get_float(self, name_or_index: str | int) -> float:
        idx = _field_index(name_or_index, _FLOAT_FIELDS)
        return struct.unpack_from(f"{self.endian}f", self.raw, idx * 4)[0]

    def set_float(self, name_or_index: str | int, value: float) -> None:
        idx = _field_index(name_or_index, _FLOAT_FIELDS)
        struct.pack_into(f"{self.endian}f", self.raw, idx * 4, float(value))

    def get_int(self, name_or_index: str | int) -> int:
        idx = _field_index(name_or_index, _INT_FIELDS)
        off = SAC_FLOAT_COUNT * 4 + idx * 4
        return struct.unpack_from(f"{self.endian}i", self.raw, off)[0]

    def set_int(self, name_or_index: str | int, value: int) -> None:
        idx = _field_index(name_or_index, _INT_FIELDS)
        off = SAC_FLOAT_COUNT * 4 + idx * 4
        struct.pack_into(f"{self.endian}i", self.raw, off, int(value))

    def get_text(self, field: str) -> str:
        off, size = _STR_FIELDS[field]
        return bytes(self.raw[off : off + size]).decode("ascii", errors="ignore").strip()

    def set_text(self, field: str, value: str) -> None:
        off, size = _STR_FIELDS[field]
        encoded = value.encode("ascii", errors="replace")[:size]
        self.raw[off : off + size] = encoded.ljust(size, b" ")

    @property
    def npts(self) -> int:
        return self.get_int("npts")

    @property
    def iftype(self) -> int:
        return self.get_int("iftype")

    @property
    def data_count(self) -> int:
        npts = self.npts
        if npts <= 0:
            raise ValueError(f"Invalid SAC npts: {npts}")
        return npts * 2 if self.iftype == IXY else npts

    @property
    def data_dtype(self) -> np.dtype:
        return np.dtype(f"{self.endian}f4")

    def data_nbytes(self) -> int:
        return self.data_count * 4


def _read_exact(fp: BinaryIO, size: int, label: str) -> bytes:
    data = fp.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes for {label}, got {len(data)}")
    return data


def read_sac(path: str | Path) -> tuple[SacHeader, np.ndarray]:
    path = Path(path)
    with path.open("rb") as fp:
        header = SacHeader.from_bytes(_read_exact(fp, SAC_HEADER_BYTES, "SAC header"))
        data_bytes = _read_exact(fp, header.data_nbytes(), "SAC data")
    data = np.frombuffer(data_bytes, dtype=header.data_dtype).astype(np.float32, copy=False)
    return header, np.array(data, dtype=np.float32, copy=True)


def read_sac_record_from(
    fp: BinaryIO,
    path: str | Path,
    offset: int = 0,
    size: int | None = None,
) -> tuple[SacHeader, np.ndarray]:
    path = Path(path)
    offset = int(offset)
    if offset < 0:
        raise ValueError(f"{path}: SAC record offset must be >= 0, got {offset}")

    fp.seek(offset)
    header = SacHeader.from_bytes(_read_exact(fp, SAC_HEADER_BYTES, "SAC record header"))
    record_size = SAC_HEADER_BYTES + header.data_nbytes()
    if size is not None and int(size) != record_size:
        raise ValueError(f"{path}: expected SAC record size {size}, header implies {record_size}")
    data_bytes = _read_exact(fp, header.data_nbytes(), "SAC record data")

    data = np.frombuffer(data_bytes, dtype=header.data_dtype).astype(np.float32, copy=False)
    return header, np.array(data, dtype=np.float32, copy=True)


def read_sac_record(path: str | Path, offset: int = 0, size: int | None = None) -> tuple[SacHeader, np.ndarray]:
    path = Path(path)
    with path.open("rb") as fp:
        return read_sac_record_from(fp, path, offset, size)


def encode_sac_record(header: SacHeader, data: np.ndarray) -> bytes:
    arr = np.asarray(data, dtype=np.float32)
    expected = header.data_count
    if arr.size != expected:
        raise ValueError(f"SAC header expects {expected} float samples, got {arr.size}")
    out = arr.astype(header.data_dtype, copy=False)
    return bytes(header.raw) + out.tobytes(order="C")


def write_sac(path: str | Path, header: SacHeader, data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data, dtype=np.float32)
    expected = header.data_count
    if arr.size != expected:
        raise ValueError(f"{path}: header expects {expected} float samples, got {arr.size}")

    with path.open("wb") as fp:
        fp.write(encode_sac_record(header, arr))


def inspect_bigsac(path: str | Path) -> tuple[SacHeader, int, int]:
    """Return `(first_header, record_count, data_count)` for a concatenated SAC."""

    path = Path(path)
    size = os.path.getsize(path)
    if size < SAC_HEADER_BYTES:
        raise ValueError(f"{path}: file is smaller than one SAC header")

    with path.open("rb") as fp:
        header = SacHeader.from_bytes(_read_exact(fp, SAC_HEADER_BYTES, "BigSAC first header"))

    record_size = SAC_HEADER_BYTES + header.data_nbytes()
    if record_size <= SAC_HEADER_BYTES:
        raise ValueError(f"{path}: invalid SAC record size {record_size}")
    if size % record_size != 0:
        raise ValueError(
            f"{path}: file size {size} is not a multiple of SAC record size {record_size}"
        )

    return header, size // record_size, header.data_count
