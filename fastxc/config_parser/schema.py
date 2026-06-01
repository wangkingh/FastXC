# config_parser/schema.py
# ---------------------------------------------------------------------- #
"""Dataclass schemas for every INI section – *100 % dict-friendly*."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Any

from fastxc.system import (
    DEFAULT_EXECUTABLES,
    resolve_executable,
    resolve_executable_root,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# 共用工具
# ---------------------------------------------------------------------- #
def _as_bool(val: str | bool | Any, *, default: bool = False) -> bool:
    """将 'yes/true/1/on' 等字符串安全转换为 bool。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _band_count(bands: str) -> int:
    return len([token for token in bands.split() if token.strip()])


def _resolve_normalize(value: str | None, bands: str) -> str:
    raw = "AUTO" if value is None else str(value).strip().upper()
    raw = raw.replace("_", "-")
    aliases = {
        "AUTO": "AUTO",
        "DEFAULT": "AUTO",
        "OFF": "OFF",
        "NONE": "OFF",
        "RUNABS": "RUN-ABS",
        "RUN-ABS": "RUN-ABS",
        "RUNABSMF": "RUN-ABS-MF",
        "RUNABS-MF": "RUN-ABS-MF",
        "RUN-ABS-MF": "RUN-ABS-MF",
        "ONEBIT": "ONE-BIT",
        "ONE-BIT": "ONE-BIT",
    }
    resolved = aliases.get(raw, raw)
    if resolved == "AUTO":
        return "RUN-ABS-MF" if _band_count(bands) > 1 else "RUN-ABS"
    return resolved


def _resolve_shift_len(value: str | None, win_len: int) -> int:
    raw = "AUTO" if value is None else str(value).strip().upper()
    if raw in {"", "AUTO", "NONE"}:
        return win_len
    return int(float(raw))


def _normalize_tfpws_band(value: str | None) -> str:
    text = "FULL" if value is None else str(value).strip().upper()
    return "FULL" if text in {"", "NONE", "OFF", "FULL", "ALL"} else text


def _normalize_optional_path(value: str | None) -> str:
    if value is None:
        return "NONE"
    text = str(value).strip()
    return "NONE" if text == "" or text.upper() == "NONE" else text


def _resolve_rooted_executable(root: str, value: str | None, default_name: str) -> str:
    text = _normalize_optional_path(value if value is not None else default_name)
    if text == "NONE":
        return "NONE"
    if root == "NONE":
        return text
    path = Path(text).expanduser()
    if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return str(path)
    return str(Path(root).expanduser() / path)


# ---------------------------------------------------------------------- #
# 1. ArrayInfo / TimeFilter
# ---------------------------------------------------------------------- #
@dataclass
class ArrayInfo:
    sac_dir: str
    pattern: str
    sta_list: str = "NONE"
    component_list: List[str] = field(default_factory=list)
    group_id: str = "1"
    section_name: str = "seisarray1"
    source_name: str = ""

    # ---------- factory ----------
    @classmethod
    def from_cfg(
        cls,
        g: Mapping[str, str],
        *,
        group_id: str = "1",
        section_name: str = "seisarray1",
        source_name: str = "",
    ) -> "ArrayInfo":
        comps = [c.strip() for c in g.get("component_list", "").split(",") if c.strip()]
        return cls(
            sac_dir        = g.get("sac_dir", "NONE"),
            pattern        = g["pattern"],
            sta_list       = g.get("sta_list", "NONE"),
            component_list = comps,
            group_id       = str(group_id),
            section_name   = section_name,
            source_name    = source_name,
        )

    # ---------- validation ----------
    def validate(self) -> None:
        if self.sac_dir != "NONE" and not Path(self.sac_dir).exists():
            raise FileNotFoundError(f"SAC dir not found: {self.sac_dir}")

        if self.sta_list != "NONE" and not Path(self.sta_list).is_file():
            raise FileNotFoundError(f"sta_list not found: {self.sta_list}")

        if not re.fullmatch(r"[1-9]", str(self.group_id)):
            raise ValueError("seisarray group_id must be one digit from 1 to 9")

        if not 1 <= len(self.component_list) <= 3:
            raise ValueError("component_list must contain 1–3 items")

    # ---------- helpers ----------
    def to_dict(self): return asdict(self)


@dataclass
class TimeFilter:
    time_start: str
    time_end: str
    time_list: str = "NONE"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "TimeFilter":
        return cls(
            time_start=g["time_start"],
            time_end=g["time_end"],
            time_list=g.get("time_list", "NONE"),
        )

    def validate(self) -> None:
        for ts, lab in [(self.time_start, "time_start"),
                        (self.time_end,   "time_end")]:
            try:
                datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                raise ValueError(f"{lab} format error: {ts}") from e

        if self.time_list != "NONE" and not Path(self.time_list).is_file():
            raise FileNotFoundError(f"time_list not found: {self.time_list}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 1b. Geometry
# ---------------------------------------------------------------------- #
@dataclass
class Geometry:
    external_geo_tsv: str = "NONE"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Geometry":
        return cls(external_geo_tsv=g.get("external_geo_tsv", "NONE"))

    def validate(self) -> None:
        if self.external_geo_tsv != "NONE" and not Path(self.external_geo_tsv).is_file():
            raise FileNotFoundError(f"external_geo_tsv not found: {self.external_geo_tsv}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 2. Compute
# ---------------------------------------------------------------------- #
@dataclass
class Compute:
    win_len:    int
    shift_len:  int
    delta:      float
    normalize:  str
    bands:      str
    sac_len:    int = 86400
    whiten:     str = "AFTER"
    skip_step:  str = "-1"
    phase_only: bool = False

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Compute":
        bands = g.get("bands", "")
        win_len = int(g["win_len"])
        return cls(
            win_len   = win_len,
            shift_len = _resolve_shift_len(g.get("shift_len", "AUTO"), win_len),
            delta     = float(g["delta"]),
            normalize = _resolve_normalize(g.get("normalize", "AUTO"), bands),
            bands     = bands,
            sac_len   = int(float(g.get("sac_len", 86400))),
            whiten    = g.get("whiten", "AFTER").upper(),
            skip_step = g.get("skip_step", "-1"),
            phase_only = _as_bool(g.get("phase_only", False)),
        )

    def validate(self) -> None:
        if self.normalize not in {"OFF", "RUN-ABS", "ONE-BIT", "RUN-ABS-MF"}:
            raise ValueError("invalid normalize value")

        if self.whiten not in {"OFF", "BEFORE", "AFTER", "BOTH"}:
            raise ValueError("invalid whiten value")

        if self.sac_len <= 0:
            raise ValueError("sac_len must be > 0")
        if self.win_len <= 0 or self.shift_len <= 0:
            raise ValueError("win_len and shift_len must be > 0")

        if not re.fullmatch(r"-1|(\d+(,\d+)*)", self.skip_step):
            raise ValueError("skip_step must be -1 or comma-separated ints")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 3. Xcorr   (-C/-D/-Z)
# ---------------------------------------------------------------------- #
@dataclass
class Xcorr:
    max_lag:          int
    distance_range:   str = "-1/50000"
    azimuth_range:    str = "-1/360"
    group_pair_mode:  str = "all"         # intra / inter / all

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Xcorr":
        return cls(
            max_lag          = int(g["max_lag"]),
            distance_range   = g.get("distance_range", "-1/50000"),
            azimuth_range    = g.get("azimuth_range", "-1/360"),
            group_pair_mode  = g.get("group_pair_mode", "all").strip().lower(),
        )

    def validate(self) -> None:
        if self.max_lag <= 0:
            raise ValueError("max_lag must be > 0")
        if self.group_pair_mode not in {
            "auto",
            "all",
            "both",
            "intra",
            "within",
            "same",
            "inter",
            "between",
            "cross",
        }:
            raise ValueError("group_pair_mode must be intra, inter, or all")
        self._check_range(self.distance_range, "distance_range")
        self._check_range(self.azimuth_range,  "azimuth_range")

    @staticmethod
    def _check_range(r: str, name: str):
        try:
            lo, hi = map(float, r.split("/"))
        except Exception:
            raise ValueError(f"{name} must be 'low/high', got '{r}'")
        if lo > hi:
            raise ValueError(f"{name}: lower {lo} > upper {hi}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 3b. XCache
# ---------------------------------------------------------------------- #
@dataclass
class XCache:
    windows_per_xcache: int | None = None
    async_after_sac2spec: bool = True
    async_poll_sec: float = 5.0
    cleanup_timestamp_spack: bool = True

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "XCache":
        value = str(g.get("windows_per_xcache", g.get("windows_per_shard", "AUTO"))).strip()
        async_after_sac2spec = _as_bool(g.get("async_after_sac2spec", True))
        async_poll_sec = float(g.get("async_poll_sec", 5.0))
        cleanup_timestamp_spack = _as_bool(g.get("cleanup_timestamp_spack", True))
        if value.upper() in {"", "AUTO", "NONE"}:
            return cls(
                windows_per_xcache=None,
                async_after_sac2spec=async_after_sac2spec,
                async_poll_sec=async_poll_sec,
                cleanup_timestamp_spack=cleanup_timestamp_spack,
            )
        return cls(
            windows_per_xcache=int(value),
            async_after_sac2spec=async_after_sac2spec,
            async_poll_sec=async_poll_sec,
            cleanup_timestamp_spack=cleanup_timestamp_spack,
        )

    def validate(self) -> None:
        if self.windows_per_xcache is not None and self.windows_per_xcache < 1:
            raise ValueError("windows_per_xcache must be AUTO or a positive integer")
        if self.async_poll_sec <= 0:
            raise ValueError("xcache async_poll_sec must be > 0")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 3c. SourcePack
# ---------------------------------------------------------------------- #
@dataclass
class SourcePack:
    enabled: bool = True
    sort_within_source: bool = True
    async_after_xc: bool = True
    async_poll_sec: float = 5.0

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "SourcePack":
        return cls(
            enabled=True,
            sort_within_source=True,
            async_after_xc=_as_bool(g.get("async_after_xc", True)),
            async_poll_sec=float(g.get("async_poll_sec", 5.0)),
        )

    def validate(self) -> None:
        if self.async_poll_sec <= 0:
            raise ValueError("async_poll_sec must be > 0")
        return None

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 3d. Unpack
# ---------------------------------------------------------------------- #
@dataclass
class Unpack:
    enabled: bool = True
    target: str = "ALL"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Unpack":
        return cls(
            enabled=_as_bool(g.get("enabled", True)),
            target=str(g.get("target", "ALL")).strip().upper(),
        )

    @property
    def output_dir(self) -> str:
        return "result_ncf"

    def validate(self) -> None:
        if self.target not in {"FINAL", "STACK", "ROTATE", "ALL"}:
            raise ValueError("target must be FINAL, STACK, ROTATE, or ALL")

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "target": self.target,
        }


# ---------------------------------------------------------------------- #
# 4. Stack   (-S/-B/-F)
# ---------------------------------------------------------------------- #
@dataclass
class Stack:
    stack_flag:       str = "100"
    pre_stack_size:   int = 10
    tfpws_band:       str = "FULL"
    tfpws_taper_hz:   str = "AUTO"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Stack":
        return cls(
            stack_flag       = g.get("stack_flag", "100"),
            pre_stack_size   = int(g.get("pre_stack_size", 10)),
            tfpws_band       = _normalize_tfpws_band(g.get("tfpws_band", "FULL")),
            tfpws_taper_hz   = str(g.get("tfpws_taper_hz", "AUTO")).strip(),
        )

    def validate(self) -> None:
        if not re.fullmatch(r"[01]{3}", self.stack_flag):
            raise ValueError("stack_flag must be three binary digits")
        if self.pre_stack_size < 1:
            raise ValueError("pre_stack_size must be ≥1")
        if self.tfpws_band != "FULL":
            Xcorr._check_range(self.tfpws_band, "tfpws_band")
        if self.tfpws_taper_hz.upper() not in {"AUTO", "NONE", "OFF"}:
            try:
                taper = float(self.tfpws_taper_hz)
            except ValueError as exc:
                raise ValueError("tfpws_taper_hz must be AUTO or a non-negative Hz value") from exc
            if taper < 0:
                raise ValueError("tfpws_taper_hz must be >= 0")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 5. Executables
# ---------------------------------------------------------------------- #
@dataclass
class Executables:
    sac2spec: str
    xc:       str
    pws:      str | None = None
    tfpws:    str | None = None
    executable_root: str = "AUTO"

    @classmethod
    def from_cfg(cls, g):
        ini_dir = Path(g.get("_ini_dir", ".")).expanduser()
        root = resolve_executable_root(
            g.get("executable_root") or g.get("root"),
            ini_dir=ini_dir,
        )

        return cls(
            sac2spec=resolve_executable(
                value=g.get("sac2spec"),
                default_names=DEFAULT_EXECUTABLES["sac2spec"],
                root=root,
                ini_dir=ini_dir,
            ),
            xc=resolve_executable(
                value=g.get("xc"),
                default_names=DEFAULT_EXECUTABLES["xc"],
                root=root,
                ini_dir=ini_dir,
            ),
            pws=resolve_executable(
                value=g.get("pws"),
                default_names=DEFAULT_EXECUTABLES["pws"],
                root=root,
                ini_dir=ini_dir,
            ),
            tfpws=resolve_executable(
                value=g.get("tfpws"),
                default_names=DEFAULT_EXECUTABLES["tfpws"],
                root=root,
                ini_dir=ini_dir,
            ),
            executable_root=root,
        )

    def validate(self):
        for p in [self.sac2spec, self.xc, self.pws, self.tfpws]:
            if p is None:
                continue
            if p != "NONE" and not Path(p).is_file():
                log.warning(
                    "Executable path not found during config validation; this is "
                    "allowed for skipped steps or Python-native replacements: %s",
                    p,
                )

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 6. Device
# ---------------------------------------------------------------------- #
@dataclass
class Device:
    gpu_list: List[int]
    gpu_memory_mib: List[float] = field(default_factory=list)
    cpu_workers: int = 20

    @classmethod
    def from_cfg(cls, g):
        return cls(
            gpu_list=cls._parse_gpu_list(g.get("gpu_list", "")),
            gpu_memory_mib=cls._parse_gpu_memory_mib(g.get("gpu_memory_mib", "AUTO")),
            cpu_workers=int(g.get("cpu_workers", 20)),
        )

    @staticmethod
    def _parse_gpu_list(value: str) -> List[int]:
        raw = str(value or "").strip()
        if not raw:
            return []

        ids: List[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                raise ValueError("gpu_list contains an empty item")
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"gpu_list must be non-negative: {gpu_id}")
            ids.append(gpu_id)
        return ids

    @staticmethod
    def _parse_gpu_memory_mib(value: str) -> List[float]:
        raw = str(value or "AUTO").strip()
        if raw.upper() in {"", "AUTO", "NONE", "OFF"}:
            return []

        limits: List[float] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                raise ValueError("gpu_memory_mib contains an empty item")
            if token.upper() == "AUTO":
                limits.append(0.0)
                continue
            limit = float(token)
            if limit < 0:
                raise ValueError(f"gpu_memory_mib must be non-negative: {limit}")
            limits.append(limit)
        return limits

    def validate(self):
        if not self.gpu_list:
            self.gpu_list = [0]
        if self.gpu_memory_mib and len(self.gpu_memory_mib) != len(self.gpu_list):
            raise ValueError(
                "gpu_memory_mib length must match gpu_list worker count "
                f"({len(self.gpu_memory_mib)} != {len(self.gpu_list)})"
            )
        if self.cpu_workers < 1:
            raise ValueError("cpu_workers must be >= 1")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 7. Storage
# ---------------------------------------------------------------------- #
@dataclass
class Storage:
    output_dir: Path

    @classmethod
    def from_cfg(cls, g):
        root = g["workspace_dir"]
        return cls(
            output_dir = Path(root).expanduser().resolve(),
        )

    @property
    def workspace_dir(self) -> Path:
        return self.output_dir

    def validate(self):
        Path(self.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return {
            "workspace_dir": str(self.workspace_dir),
        }


# ---------------------------------------------------------------------- #
# 8. Advance
# ---------------------------------------------------------------------- #
@dataclass
class Advance:
    debug: bool = False
    _log_file_path: str = field(default="NONE", init=False, repr=False)

    # ---------- factory ----------
    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Advance":
        return cls(
            debug=_as_bool(g.get("debug", g.get("native_debug", False))),
        )

    @property
    def dry_run(self) -> bool:
        return False

    @property
    def log_file_path(self) -> str:
        return self._log_file_path

    # ---------- validation ----------
    def validate(self, *, output_dir: str | Path) -> None:
        out_dir = Path(output_dir).expanduser().resolve()

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = out_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file_path = str(log_dir / f"fastxc-{ts}.log")
        Path(self._log_file_path).touch(exist_ok=True)

    # ---------- helper ----------
    def to_dict(self): return {"debug": self.debug}


Debug = Advance
