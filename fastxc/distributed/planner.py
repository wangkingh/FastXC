from __future__ import annotations

import configparser
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fastxc.inventory import require_inventory, sac_index_path

from .resources import NodeResource, check_node, load_nodes


SAC_INDEX_HEADER = ["timestamp", "nsl_id", "network", "station", "location", "component", "sac_path"]
RUN_PLAN_HEADER = [
    "task_id",
    "node",
    "host",
    "workspace",
    "config",
    "sac_index",
    "timestamp_start",
    "timestamp_end",
    "timestamp_count",
    "row_count",
    "status",
]


@dataclass(frozen=True)
class DistributedPlan:
    root: Path
    run_plan: Path
    sourcepack_inputs: Path
    task_count: int
    timestamp_count: int
    row_count: int


def write_distributed_plan(
    config: Any,
    *,
    resources: str | Path | None = None,
    parts: int | None = None,
    plan_dir: str | Path | None = None,
    force: bool = False,
    check: bool = True,
) -> DistributedPlan:
    require_inventory(config)
    workspace = Path(config.storage.output_dir).expanduser().resolve()
    root = Path(plan_dir).expanduser().resolve() if plan_dir else workspace / "distributed"
    tasks_root = root / "tasks"
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)

    rows = _read_sac_index(sac_index_path(workspace))
    if not rows:
        raise ValueError(f"SAC index is empty: {sac_index_path(workspace)}")

    nodes = load_nodes(resources, config, parts=parts, workspace_root=tasks_root)
    if check:
        sample_sacs = [row["sac_path"] for row in rows[: min(8, len(rows))]]
        errors: list[str] = []
        for node in nodes:
            errors.extend(check_node(node, sample_sac_paths=sample_sacs))
        if errors:
            raise RuntimeError("Distributed resource check failed:\n  " + "\n  ".join(errors))

    chunks = _split_timestamp_chunks(rows, len(nodes))
    task_rows: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        node = nodes[index % len(nodes)]
        task_id = f"task_{len(task_rows):04d}"
        task_root = tasks_root / task_id
        task_workspace = _task_workspace(node, task_root)
        task_workspace.mkdir(parents=True, exist_ok=True)
        task_sac_index = task_workspace / "manifest" / "sac_index.tsv"
        _write_sac_index(task_sac_index, chunk)
        _copy_shared_inventory(workspace, task_workspace)
        task_config = task_root / "config.ini"
        _write_task_config(config.ini_path, task_config, node, task_workspace)

        timestamps = sorted({row["timestamp"] for row in chunk})
        task_rows.append(
            {
                "task_id": task_id,
                "node": node.name,
                "host": node.host,
                "workspace": task_workspace.as_posix(),
                "config": task_config.as_posix(),
                "sac_index": task_sac_index.as_posix(),
                "timestamp_start": timestamps[0],
                "timestamp_end": timestamps[-1],
                "timestamp_count": str(len(timestamps)),
                "row_count": str(len(chunk)),
                "status": "PENDING",
            }
        )

    run_plan = root / "run_plan.tsv"
    _write_tsv(run_plan, RUN_PLAN_HEADER, task_rows)
    sourcepack_inputs = root / "sourcepack_inputs.txt"
    sourcepack_inputs.write_text("", encoding="utf-8")
    _write_plan_metadata(root, config, nodes, task_rows, resources)
    return DistributedPlan(
        root=root,
        run_plan=run_plan,
        sourcepack_inputs=sourcepack_inputs,
        task_count=len(task_rows),
        timestamp_count=len({row["timestamp"] for row in rows}),
        row_count=len(rows),
    )


def load_distributed_plan(run_plan: str | Path) -> list[dict[str, str]]:
    path = Path(run_plan).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        missing = set(RUN_PLAN_HEADER).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Run plan missing fields {sorted(missing)}: {path}")
        return [dict(row) for row in reader]


def collect_plan_sourcepacks(
    run_plan: str | Path,
    *,
    output_list: str | Path | None = None,
    main_workspace: str | Path | None = None,
) -> Path:
    plan_path = Path(run_plan).expanduser().resolve()
    rows = load_distributed_plan(plan_path)
    indexes: list[Path] = []
    for row in rows:
        workspace = Path(row["workspace"]).expanduser().resolve()
        indexes.extend(sorted((workspace / "sourcepack").glob("*/sourcepack_index.tsv")))

    target = Path(output_list).expanduser().resolve() if output_list else plan_path.parent / "sourcepack_inputs.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{path.resolve().as_posix()}\n" for path in indexes), encoding="utf-8")

    if main_workspace is not None:
        mirror = Path(main_workspace).expanduser().resolve() / "sourcepack_inputs.txt"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _read_sac_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        missing = set(SAC_INDEX_HEADER).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"SAC index missing fields {sorted(missing)}: {path}")
        return [dict(row) for row in reader if row.get("timestamp")]


def _split_timestamp_chunks(rows: list[dict[str, str]], parts: int) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["timestamp"], []).append(row)
    timestamps = sorted(grouped)
    parts = max(1, int(parts))
    chunks: list[list[dict[str, str]]] = []
    for index in range(parts):
        start = index * len(timestamps) // parts
        end = (index + 1) * len(timestamps) // parts
        chunk_timestamps = timestamps[start:end]
        chunk: list[dict[str, str]] = []
        for timestamp in chunk_timestamps:
            chunk.extend(grouped[timestamp])
        chunks.append(chunk)
    return chunks


def _write_sac_index(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(path, SAC_INDEX_HEADER, rows)


def _write_tsv(path: Path, header: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_shared_inventory(src_workspace: Path, task_workspace: Path) -> None:
    for name in ("path_plan",):
        src = src_workspace / name
        dst = task_workspace / name
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
    for name in ("filter.txt",):
        src = src_workspace / name
        if src.is_file():
            shutil.copy2(src, task_workspace / name)
    meta = {
        "schema": "fastxc-task-inventory-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_workspace": src_workspace.as_posix(),
        "sac_index": (task_workspace / "manifest" / "sac_index.tsv").as_posix(),
        "allowed_paths": (task_workspace / "path_plan" / "allowed_paths.tsv").as_posix(),
    }
    (task_workspace / "inventory.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_task_config(
    base_ini: str | Path,
    output_ini: Path,
    node: NodeResource,
    task_workspace: Path,
) -> None:
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(Path(base_ini).expanduser(), encoding="utf-8-sig")
    _ensure_section(cp, "compute")
    cp["compute"]["workspace_dir"] = task_workspace.as_posix()
    _ensure_section(cp, "executables")
    cp["executables"]["executable_root"] = "NONE"
    for key, value in node.executables.items():
        cp["executables"][key] = value
    _ensure_section(cp, "device")
    if node.gpus is not None:
        cp["device"]["gpu_list"] = node.gpus
    if node.gpu_memory_mib is not None:
        cp["device"]["gpu_memory_mib"] = node.gpu_memory_mib
    if node.cpu_workers is not None:
        cp["device"]["cpu_workers"] = str(node.cpu_workers)
    cp["compute"]["stack_flag"] = "000"
    _ensure_section(cp, "advance.storage")
    cp["advance.storage"]["unpack_enabled"] = "False"
    output_ini.parent.mkdir(parents=True, exist_ok=True)
    with output_ini.open("w", encoding="utf-8") as handle:
        cp.write(handle)


def _ensure_section(cp: configparser.ConfigParser, section: str) -> None:
    if not cp.has_section(section):
        cp.add_section(section)


def _task_workspace(node: NodeResource, task_root: Path) -> Path:
    if node.workspace is not None:
        return node.workspace.expanduser().resolve()
    return (task_root / "workspace").resolve()


def _write_plan_metadata(
    root: Path,
    config: Any,
    nodes: list[NodeResource],
    task_rows: list[dict[str, str]],
    resources: str | Path | None,
) -> None:
    metadata = {
        "schema": "fastxc-distributed-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config.ini_path).expanduser().resolve()),
        "workspace": str(Path(config.storage.output_dir).expanduser().resolve()),
        "resources": str(Path(resources).expanduser().resolve()) if resources else None,
        "nodes": [
            {
                "name": node.name,
                "host": node.host,
                "workspace": str(node.workspace) if node.workspace else None,
                "gpus": node.gpus,
                "gpu_memory_mib": node.gpu_memory_mib,
                "cpu_workers": node.cpu_workers,
            }
            for node in nodes
        ],
        "tasks": task_rows,
    }
    (root / "plan.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
