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
    Compute,
    Device,
    Executables,
    Geometry,
    SourcePack,
    Stack,
    Storage,
    TimeFilter,
    Unpack,
    Xcorr,
)

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


BASE_SECTION_MAP: Dict[str, type] = {
    "executables": Executables,
}

SEISARRAY_SECTION_RE = re.compile(r"^seisarray([1-9])(?:[._-](.+))?$", re.IGNORECASE)
RETIRED_SECTION_NAMES = {
    "preprocess",
    "xcorr",
    "stack",
    "storage",
    "unpack",
    "advance",
    "advance.preprocess",
    "advance.xcorr",
    "advance.xcache",
    "advance.sourcepack",
    "advance.stack",
}
RETIRED_ARRAY_SECTION_RE = re.compile(r"^array[1-9](?:[._-].*)?$", re.IGNORECASE)
RETIRED_ADVANCE_COMPUTE_KEYS = {
    "sourcepack_enabled",
    "sourcepack_sort_within_source",
    "xc_input",
    "xc_input_mode",
    "windows_per_xcache",
    "windows_per_shard",
    "xcache_async_after_sac2spec",
    "xcache_async_poll_sec",
    "xcache_cleanup_timestamp_spack",
    "sourcepack_async_poll_sec",
}
RETIRED_ADVANCE_STORAGE_KEYS = {
    "result_dir",
}


class Config(Mapping[str, Any]):
    """Main configuration object.

    Configs use one or more ``[seisarrayN]`` or ``[seisarrayN.source]``
    sections plus global ``[time_filter]``, ``[compute]``, ``[device]``,
    ``[advance.compute]`` and ``[advance.storage]`` sections.
    """

    def __init__(self, ini_path: str | Path, *, env_expand: bool = True) -> None:
        path = Path(ini_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"INI file not found: {path}")

        self.ini_path = path
        self._cp = cp = configparser.ConfigParser(interpolation=None)
        cp.read(path, encoding="utf-8-sig")
        self._reject_retired_sections(cp)

        self._sections: Dict[str, Any] = {}

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
            kv = self._section_items(cp, sec, env_expand=env_expand)
            if sec == "executables":
                kv["_ini_dir"] = str(path.parent.resolve())
            try:
                obj = cls.from_cfg(kv)  # type: ignore[arg-type]
            except Exception as exc:
                raise ConfigError(f"[{sec}] parsing error: {exc}") from None
            self._sections[sec] = obj

        if missing_secs:
            raise ConfigError(f"Missing required section(s): {', '.join(missing_secs)}")

        if "compute" not in cp:
            raise ConfigError("Missing required section: compute")
        compute_kv = self._section_items(cp, "compute", env_expand=env_expand)
        advance_compute_kv = self._section_items(cp, "advance.compute", env_expand=env_expand)
        advance_storage_kv = self._section_items(cp, "advance.storage", env_expand=env_expand)
        self._reject_retired_keys("advance.compute", advance_compute_kv, RETIRED_ADVANCE_COMPUTE_KEYS)
        self._reject_retired_keys("advance.storage", advance_storage_kv, RETIRED_ADVANCE_STORAGE_KEYS)

        geometry_kv = self._section_items(cp, "geometry", env_expand=env_expand)
        try:
            self._sections["geometry"] = Geometry.from_cfg(geometry_kv)
        except Exception as exc:
            raise ConfigError(f"[geometry] parsing error: {exc}") from None

        compute_parse_kv = dict(compute_kv)
        _copy_keys(advance_compute_kv, compute_parse_kv, {"skip_step", "phase_only"})
        try:
            self._sections["compute"] = Compute.from_cfg(compute_parse_kv)
        except Exception as exc:
            raise ConfigError(f"[compute] parsing error: {exc}") from None

        if "device" not in cp:
            raise ConfigError("Missing required section: device")
        try:
            self._sections["device"] = Device.from_cfg(self._section_items(cp, "device", env_expand=env_expand))
        except Exception as exc:
            raise ConfigError(f"[device] parsing error: {exc}") from None

        if "max_lag" not in compute_kv:
            raise ConfigError("[compute] missing required field: max_lag")
        xcorr_kv = dict(advance_compute_kv)
        xcorr_kv["max_lag"] = compute_kv["max_lag"]
        try:
            self._sections["xcorr"] = Xcorr.from_cfg(xcorr_kv)
        except Exception as exc:
            raise ConfigError(f"[compute/advance.compute] xcorr parsing error: {exc}") from None

        if "stack_flag" not in compute_kv:
            raise ConfigError("[compute] missing required field: stack_flag")
        stack_kv = dict(advance_compute_kv)
        stack_kv["stack_flag"] = compute_kv["stack_flag"]
        try:
            self._sections["stack"] = Stack.from_cfg(stack_kv)
        except Exception as exc:
            raise ConfigError(f"[compute/advance.compute] stack parsing error: {exc}") from None

        if "workspace_dir" not in compute_kv:
            raise ConfigError("[compute] missing required field: workspace_dir")
        storage_kv = {"workspace_dir": compute_kv["workspace_dir"]}
        try:
            self._sections["storage"] = Storage.from_cfg(storage_kv)
        except Exception as exc:
            raise ConfigError(f"[compute.workspace_dir] parsing error: {exc}") from None

        sourcepack_kv = _project_keys(
            advance_compute_kv,
            {
                "sourcepack_async_after_xc": "async_after_xc",
                "async_poll_sec": "async_poll_sec",
            },
        )
        try:
            self._sections["sourcepack"] = SourcePack.from_cfg(sourcepack_kv)
        except Exception as exc:
            raise ConfigError(f"[advance.compute] sourcepack parsing error: {exc}") from None

        unpack_kv = _project_keys(
            advance_storage_kv,
            {
                "unpack_enabled": "enabled",
                "unpack_target": "target",
            },
        )
        try:
            self._sections["unpack"] = Unpack.from_cfg(unpack_kv)
        except Exception as exc:
            raise ConfigError(f"[advance.storage] parsing error: {exc}") from None

        advance_kv = self._section_items(cp, "debug", env_expand=env_expand)
        try:
            self._sections["advance"] = Advance.from_cfg(advance_kv)
        except Exception as exc:
            raise ConfigError(f"[debug] parsing error: {exc}") from None

        for name, obj in self._sections.items():
            setattr(self, name, obj)
        self.debug = self.advance

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

        return []

    def _reject_retired_sections(self, cp: configparser.ConfigParser) -> None:
        retired = [
            section
            for section in cp.sections()
            if section.lower() in RETIRED_SECTION_NAMES or RETIRED_ARRAY_SECTION_RE.match(section)
        ]
        if retired:
            raise ConfigError(
                "Retired INI section(s) are no longer supported: "
                + ", ".join(sorted(retired))
                + ". Use [seisarrayN], [compute], [advance.compute], [advance.storage], and [debug]."
            )

    def _reject_retired_keys(self, section: str, kv: Mapping[str, str], retired_keys: set[str]) -> None:
        found = sorted(key for key in kv if key in retired_keys)
        if found:
            raise ConfigError(
                f"Retired INI field(s) in [{section}] are no longer supported: "
                + ", ".join(found)
            )

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

        raise ConfigError("Missing required section: time_filter")

    def _section_items(
        self,
        cp: configparser.ConfigParser,
        section: str,
        *,
        env_expand: bool,
    ) -> dict[str, str]:
        kv = dict(cp[section].items()) if section in cp else {}
        if env_expand:
            kv = {key: self._expandenv(value) for key, value in kv.items()}
        return kv

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
            for key in ("group_id", "section_name", "source_name"):
                data.pop(key, None)
            cp_out[array.section_name] = {key: _stringify(value) for key, value in data.items()}

        for name, obj in self._sections.items():
            data = obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__
            data = dict(data)
            if name == "compute":
                _move_keys(data, advanced_out.setdefault("advance.compute", {}), {"skip_step", "phase_only"})
                cp_out["compute"] = {key: _stringify(value) for key, value in data.items()}
                continue
            if name == "xcorr":
                _move_keys(data, cp_out.setdefault("compute", {}), {"max_lag"})
                advanced_out.setdefault("advance.compute", {}).update(
                    {key: _stringify(value) for key, value in data.items()}
                )
                continue
            if name == "stack":
                _move_keys(data, cp_out.setdefault("compute", {}), {"stack_flag"})
                advanced_out.setdefault("advance.compute", {}).update(
                    {key: _stringify(value) for key, value in data.items()}
                )
                continue
            if name == "storage":
                cp_out.setdefault("compute", {})["workspace_dir"] = _stringify(obj.workspace_dir)
                continue

            if name == "sourcepack":
                advanced_out.setdefault("advance.compute", {}).update(
                    _prefixed_items(
                        data,
                        {
                            "async_after_xc": "sourcepack_async_after_xc",
                            "async_poll_sec": "async_poll_sec",
                        },
                    )
                )
                continue
            if name == "unpack":
                advanced_out.setdefault("advance.storage", {}).update(
                    _prefixed_items(
                        data,
                        {
                            "enabled": "unpack_enabled",
                            "target": "unpack_target",
                        },
                    )
                )
                continue
            if name == "advance":
                cp_out["debug"] = {key: _stringify(value) for key, value in data.items()}
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
    if value is None:
        return "AUTO"
    if isinstance(value, (list, tuple)):
        if not value:
            return "AUTO"
        return ",".join(str(item) for item in value)
    return str(value)


def _move_keys(source: dict[str, Any], target: dict[str, str], keys: set[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = _stringify(source.pop(key))


def _copy_keys(source: Mapping[str, str], target: dict[str, str], keys: set[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def _project_keys(source: Mapping[str, str], mapping: Mapping[str, str]) -> dict[str, str]:
    return {
        target_key: source[source_key]
        for source_key, target_key in mapping.items()
        if source_key in source
    }


def _prefixed_items(source: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, str]:
    return {
        target_key: _stringify(source[source_key])
        for source_key, target_key in mapping.items()
        if source_key in source
    }


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
