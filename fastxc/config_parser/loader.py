# config_parser/loader.py
# ----------------------------------------------------------------------------- #
"""Load FastXC INI files into dataclass-backed configuration objects."""

from __future__ import annotations

import configparser
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

from .schema import (
    Advance,
    ArrayInfo,
    Device,
    Executables,
    Geometry,
    Preprocess,
    SourcePack,
    Stack,
    Storage,
    TimeFilter,
    Unpack,
    XCache,
    Xcorr,
)

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


BASE_SECTION_MAP: Dict[str, type] = {
    "executables": Executables,
    "preprocess": Preprocess,
    "xcorr": Xcorr,
    "stack": Stack,
    "device": Device,
    "storage": Storage,
}

SEISARRAY_SECTION_RE = re.compile(r"^seisarray([1-9])(?:[._-](.+))?$", re.IGNORECASE)


class Config(Mapping[str, Any]):
    """Main configuration object.

    New configs should use one or more ``[seisarrayN]`` or
    ``[seisarrayN.source]`` sections plus a global ``[time_filter]`` section.
    Legacy ``[array1]`` / ``[array2]`` configs remain readable.
    """

    def __init__(self, ini_path: str | Path, *, env_expand: bool = True) -> None:
        path = Path(ini_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"INI file not found: {path}")

        self.ini_path = path
        self._cp = cp = configparser.ConfigParser(interpolation=None)
        cp.read(path, encoding="utf-8-sig")

        self._sections: Dict[str, Any] = {}
        self._legacy_arrays = not any(SEISARRAY_SECTION_RE.match(sec) for sec in cp.sections())

        self.seisarrays = self._load_seisarrays(cp, env_expand=env_expand)
        if not self.seisarrays:
            raise ConfigError("At least one [seisarrayN] section is required")

        self.time_filter = self._load_time_filter(cp, env_expand=env_expand)
        self._sections["time_filter"] = self.time_filter

        missing_secs: list[str] = []
        for sec, cls in BASE_SECTION_MAP.items():
            if sec not in cp:
                missing_secs.append(sec)
                continue
            advanced_sec = f"advance.{sec}" if sec in {"preprocess", "xcorr"} else None
            kv = self._merged_section_items(cp, sec, advanced_sec, env_expand=env_expand)
            if sec == "executables":
                kv["_ini_dir"] = str(path.parent.resolve())
            try:
                obj = cls.from_cfg(kv)  # type: ignore[arg-type]
            except Exception as exc:
                raise ConfigError(f"[{sec}] parsing error: {exc}") from None
            self._sections[sec] = obj

        if missing_secs:
            raise ConfigError(f"Missing required section(s): {', '.join(missing_secs)}")

        geometry_kv = (
            dict(cp["geometry"].items())
            if "geometry" in cp
            else {"external_geo_tsv": cp["xcorr"].get("external_geo_tsv", "NONE")}
        )
        if env_expand:
            geometry_kv = {k: self._expandenv(v) for k, v in geometry_kv.items()}
        try:
            self._sections["geometry"] = Geometry.from_cfg(geometry_kv)
        except Exception as exc:
            raise ConfigError(f"[geometry] parsing error: {exc}") from None

        xcache_kv = self._merged_section_items(cp, "xcache", "advance.xcache", env_expand=env_expand)
        try:
            self._sections["xcache"] = XCache.from_cfg(xcache_kv)
        except Exception as exc:
            raise ConfigError(f"[xcache] parsing error: {exc}") from None

        sourcepack_kv = self._merged_section_items(
            cp,
            "sourcepack",
            "advance.sourcepack",
            env_expand=env_expand,
        )
        try:
            self._sections["sourcepack"] = SourcePack.from_cfg(sourcepack_kv)
        except Exception as exc:
            raise ConfigError(f"[sourcepack] parsing error: {exc}") from None

        unpack_kv = self._merged_section_items(cp, "unpack", "advance.unpack", env_expand=env_expand)
        try:
            self._sections["unpack"] = Unpack.from_cfg(unpack_kv)
        except Exception as exc:
            raise ConfigError(f"[unpack] parsing error: {exc}") from None

        advance_sec = "advance" if "advance" in cp else "debug" if "debug" in cp else None
        advance_kv = dict(cp[advance_sec].items()) if advance_sec else {}
        if env_expand:
            advance_kv = {k: self._expandenv(v) for k, v in advance_kv.items()}
        try:
            self._sections["advance"] = Advance.from_cfg(advance_kv)
        except Exception as exc:
            raise ConfigError(f"[advance] parsing error: {exc}") from None

        if self._legacy_arrays and self._legacy_has_array2(cp) and "group_pair_mode" not in cp["xcorr"]:
            # Preserve the historical two-array behavior: array1 x array2 only.
            self._sections["xcorr"].group_pair_mode = "inter"

        for name, obj in self._sections.items():
            setattr(self, name, obj)
        self.debug = self.advance

        # Backward-compatible convenience aliases. New code should use
        # cfg.seisarrays and cfg.time_filter.
        self.array1 = self.seisarrays[0]
        if len(self.seisarrays) > 1:
            self.array2 = self.seisarrays[1]

        log.debug(
            "Config build complete. %d seisarray source(s), sections: %s",
            len(self.seisarrays),
            list(self._sections),
        )

    def _load_seisarrays(
        self,
        cp: configparser.ConfigParser,
        *,
        env_expand: bool,
    ) -> list[ArrayInfo]:
        arrays: list[ArrayInfo] = []
        for sec in cp.sections():
            match = SEISARRAY_SECTION_RE.match(sec)
            if not match:
                continue
            group_id, source_name = match.groups()
            kv = dict(cp[sec].items())
            if env_expand:
                kv = {k: self._expandenv(v) for k, v in kv.items()}
            try:
                arrays.append(
                    ArrayInfo.from_cfg(
                        kv,
                        group_id=group_id,
                        section_name=sec,
                        source_name=source_name or "",
                    )
                )
            except Exception as exc:
                raise ConfigError(f"[{sec}] parsing error: {exc}") from None

        if arrays:
            return sorted(arrays, key=lambda item: (int(item.group_id), item.section_name))

        legacy: list[ArrayInfo] = []
        if "array1" in cp:
            kv = dict(cp["array1"].items())
            if env_expand:
                kv = {k: self._expandenv(v) for k, v in kv.items()}
            legacy.append(
                ArrayInfo.from_cfg(
                    kv,
                    group_id="1",
                    section_name="array1",
                    source_name="legacy",
                )
            )
        if self._legacy_has_array2(cp):
            kv = dict(cp["array2"].items())
            if env_expand:
                kv = {k: self._expandenv(v) for k, v in kv.items()}
            legacy.append(
                ArrayInfo.from_cfg(
                    kv,
                    group_id="2",
                    section_name="array2",
                    source_name="legacy",
                )
            )
        return legacy

    def _load_time_filter(
        self,
        cp: configparser.ConfigParser,
        *,
        env_expand: bool,
    ) -> TimeFilter:
        if "time_filter" in cp:
            kv = dict(cp["time_filter"].items())
            if env_expand:
                kv = {k: self._expandenv(v) for k, v in kv.items()}
            try:
                return TimeFilter.from_cfg(kv)
            except Exception as exc:
                raise ConfigError(f"[time_filter] parsing error: {exc}") from None

        if self.seisarrays and self._legacy_arrays:
            try:
                return TimeFilter.from_array_info(self.seisarrays[0])
            except Exception as exc:
                raise ConfigError(f"legacy time filter parsing error: {exc}") from None

        raise ConfigError("Missing required section: time_filter")

    def _merged_section_items(
        self,
        cp: configparser.ConfigParser,
        section: str,
        advanced_section: str | None = None,
        *,
        env_expand: bool,
    ) -> dict[str, str]:
        kv: dict[str, str] = {}
        if advanced_section and advanced_section in cp:
            kv.update(dict(cp[advanced_section].items()))
        if section in cp:
            kv.update(dict(cp[section].items()))
        if env_expand:
            kv = {key: self._expandenv(value) for key, value in kv.items()}
        return kv

    @staticmethod
    def _legacy_has_array2(cp: configparser.ConfigParser) -> bool:
        return "array2" in cp and cp["array2"].get("sac_dir", "NONE").strip().upper() != "NONE"

    @property
    def is_double_array(self) -> bool:
        return len({array.group_id for array in self.seisarrays}) > 1

    @property
    def arrays(self) -> list[ArrayInfo]:
        return list(self.seisarrays)

    @property
    def primary_component_list(self) -> list[str]:
        return list(self.seisarrays[0].component_list)

    def validate_all(self) -> None:
        try:
            for array in self.seisarrays:
                array.validate()
            self.time_filter.validate()
        except Exception as exc:
            raise ConfigError(f"[seisarray/time_filter] validation failed: {exc}") from None

        for name, obj in self._sections.items():
            if name == "time_filter" or not hasattr(obj, "validate"):
                continue
            try:
                if name == "advance":
                    obj.validate(output_dir=self.storage.output_dir)
                else:
                    obj.validate()
            except Exception as exc:
                raise ConfigError(f"[{name}] validation failed: {exc}") from None

        component_lists = {tuple(array.component_list) for array in self.seisarrays}
        if len(component_lists) > 1:
            log.warning(
                "Multiple component_list definitions detected across seisarrays: %s",
                sorted(component_lists),
            )

        log.info("All configuration checks passed.")

    def override(self, section: str, **updates: Any) -> None:
        if section == "debug":
            section = "advance"
        if section not in self._sections:
            raise KeyError(f"Config has no section '{section}'")
        obj = self._sections[section]
        for key, value in updates.items():
            if not hasattr(obj, key):
                raise AttributeError(f"[{section}] has no field '{key}'")
            setattr(obj, key, value)
        log.debug("Override <%s>: %s", section, updates)

    def to_ini(self, path: str | Path, *, include_defaults: bool = True) -> None:
        del include_defaults
        cp_out = configparser.ConfigParser()
        advanced_out: dict[str, dict[str, str]] = {}
        for array in self.seisarrays:
            data = array.to_dict()
            for key in ("group_id", "section_name", "source_name", "time_start", "time_end", "time_list"):
                data.pop(key, None)
            cp_out[array.section_name] = {key: _stringify(value) for key, value in data.items()}

        for name, obj in self._sections.items():
            data = obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__
            data = dict(data)
            if name == "preprocess":
                _move_keys(data, advanced_out.setdefault("advance.preprocess", {}), {"skip_step"})
            elif name == "xcorr":
                _move_keys(data, advanced_out.setdefault("advance.xcorr", {}), {"write_mode", "write_segment"})

            if name == "xcache":
                advanced_out["advance.xcache"] = {key: _stringify(value) for key, value in data.items()}
                continue
            if name == "sourcepack":
                advanced_out["advance.sourcepack"] = {key: _stringify(value) for key, value in data.items()}
                continue

            cp_out[name] = {key: _stringify(value) for key, value in data.items()}

        for section, data in advanced_out.items():
            if data:
                cp_out[section] = data

        with Path(path).open("w", encoding="utf-8") as handle:
            cp_out.write(handle)
        log.info("Config written to %s", path)

    def __getitem__(self, key: str) -> Any:
        if key == "debug":
            key = "advance"
        return self._sections[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sections)

    def __len__(self) -> int:
        return len(self._sections)

    @staticmethod
    def _expandenv(value: str) -> str:
        if "${" in value:
            value = os.path.expandvars(value)
        if value.startswith("~"):
            value = os.path.expanduser(value)
        return value

    def __repr__(self) -> str:  # pragma: no cover
        secs = ", ".join(self._sections)
        return f"<Config [{secs}]>"


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _move_keys(source: dict[str, Any], target: dict[str, str], keys: set[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = _stringify(source.pop(key))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a FastXC INI file.")
    parser.add_argument("ini", help="path to configuration INI")
    parser.add_argument("-q", "--quiet", action="store_true", help="mute INFO logs")
    args = parser.parse_args()

    level = logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        cfg = Config(args.ini)
        cfg.validate_all()
    except ConfigError as exc:
        log.error(exc)
        sys.exit(1)

    log.info(
        "INI OK. Seisarrays: %s. Sections: %s",
        [array.section_name for array in cfg.seisarrays],
        list(cfg),
    )
