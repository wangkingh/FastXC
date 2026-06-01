from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable

from pandas import Timestamp

from fastxc.config_parser.schema import ArrayInfo, TimeFilter

from .arrays import SeisArray

logger = logging.getLogger(__name__)


def _read_station_list(sta_list_path: str) -> list[str]:
    station_list: list[str] = []
    if sta_list_path != "NONE":
        with open(sta_list_path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                station_list.append(line)
    return station_list


def _read_time_list(time_list_path: str) -> list[Timestamp]:
    times: list[Timestamp] = []
    if time_list_path != "NONE":
        with open(time_list_path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                times.append(Timestamp(line))
    return times


def _gen_seis_file_group(
    sac_dir: str,
    pattern: str,
    sta_list_path: str,
    component_list: list[str],
    time_filter: TimeFilter,
    cpu_workers: int,
) -> Dict:
    """Generate a grouped SAC file table for one physical data source."""
    if sac_dir.upper() == "NONE":
        return {}

    criteria = {}
    sta_list = _read_station_list(sta_list_path)
    if sta_list:
        criteria["station"] = {"type": "list", "data_type": "str", "value": sta_list}
    else:
        logger.warning("Sta_list is empty, will not use it as a criteria")

    time_list = _read_time_list(time_filter.time_list)
    if time_list:
        criteria["time"] = {"type": "list", "data_type": "datetime", "value": time_list}
        logger.info("Time_list %s is used as a criteria", time_list)
    else:
        time_range = [Timestamp(time_filter.time_start), Timestamp(time_filter.time_end)]
        criteria["time"] = {
            "type": "range",
            "data_type": "datetime",
            "value": time_range,
        }
        logger.warning("Time_list is empty, will use time_range as a criteria")
        logger.info("Time_range %s is used as a criteria", time_range)

    criteria["component"] = {
        "type": "list",
        "data_type": "str",
        "value": component_list,
    }

    seis_array = SeisArray(sac_dir, pattern)
    seis_array.match(threads=cpu_workers)
    seis_array.filter(criteria, threads=cpu_workers)

    # Always include network/location in the key. The matcher supplies stable
    # defaults, so this avoids accidental collisions when multiple source
    # sections share one group id.
    group_labels = ["station", "time", "network", "location"]
    sort_labels = ["component"]
    seis_array.group(labels=group_labels, sort_labels=sort_labels, filtered=True)
    return seis_array.files_group or {}


def organize_seisarrays(
    seisarrays: Iterable[ArrayInfo],
    time_filter: TimeFilter,
    cpu_workers: int,
) -> tuple[dict[str, Dict], dict[str, list[ArrayInfo]]]:
    """Match/filter all configured seisarray sources and merge by group id."""
    files_by_group: dict[str, Dict] = OrderedDict()
    arrays_by_group: dict[str, list[ArrayInfo]] = OrderedDict()

    for array in seisarrays:
        files_group = _gen_seis_file_group(
            str(Path(array.sac_dir).expanduser()),
            array.pattern,
            array.sta_list,
            array.component_list,
            time_filter,
            cpu_workers,
        )
        group_id = str(array.group_id)
        arrays_by_group.setdefault(group_id, []).append(array)
        target = files_by_group.setdefault(group_id, OrderedDict())
        _merge_files_group(target, files_group)

    return files_by_group, arrays_by_group


def _merge_files_group(target: Dict, incoming: Dict) -> None:
    for key, info in incoming.items():
        if key not in target:
            target[key] = {
                field: list(value) if isinstance(value, list) else value
                for field, value in info.items()
            }
            continue

        merged = target[key]
        for field, value in info.items():
            if isinstance(value, list):
                merged.setdefault(field, [])
                merged[field].extend(value)
            elif field not in merged:
                merged[field] = value
