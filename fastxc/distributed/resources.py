from __future__ import annotations

import configparser
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class NodeResource:
    name: str
    host: str = "localhost"
    workspace: Path | None = None
    gpus: str | None = None
    gpu_memory_mib: str | None = None
    cpu_workers: int | None = None
    executables: dict[str, str] = field(default_factory=dict)
    max_jobs: int = 1

    @property
    def is_local(self) -> bool:
        return self.host.strip().lower() in LOCAL_HOSTS


def default_local_nodes(
    config: Any,
    *,
    parts: int | None = None,
    workspace_root: str | Path | None = None,
) -> list[NodeResource]:
    count = max(1, int(parts or 1))
    gpus = ",".join(str(gpu_id) for gpu_id in config.device.gpu_list) or "0"
    gpu_memory = (
        ",".join(f"{limit:g}" for limit in config.device.gpu_memory_mib)
        if config.device.gpu_memory_mib
        else "AUTO"
    )
    executables = _config_executables(config)
    root = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path(config.storage.output_dir).expanduser().resolve() / "distributed" / "tasks"
    )
    return [
        NodeResource(
            name=f"local{i:04d}",
            host="localhost",
            workspace=root / f"task_{i:04d}" / "workspace",
            gpus=gpus,
            gpu_memory_mib=gpu_memory,
            cpu_workers=config.device.cpu_workers,
            executables=executables,
        )
        for i in range(count)
    ]


def load_nodes(
    resource_path: str | Path | None,
    config: Any,
    *,
    parts: int | None = None,
    workspace_root: str | Path | None = None,
) -> list[NodeResource]:
    if resource_path is None:
        return default_local_nodes(config, parts=parts, workspace_root=workspace_root)

    path = Path(resource_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resource config not found: {path}")

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding="utf-8-sig")
    nodes: list[NodeResource] = []
    default_execs = _config_executables(config)
    for section in cp.sections():
        if not section.lower().startswith("node."):
            continue
        name = section.split(".", 1)[1].strip()
        values = cp[section]
        workspace = values.get("workspace")
        executables = dict(default_execs)
        for key in ("sac2spec", "xc", "pws", "tfpws"):
            if values.get(key):
                executables[key] = values.get(key, "").strip()
        bin_dir = values.get("bin_dir")
        if bin_dir:
            bin_root = Path(bin_dir).expanduser()
            for key, value in list(executables.items()):
                if value.upper() in {"", "AUTO", "NONE"}:
                    continue
                if not values.get(key):
                    executables[key] = str(bin_root / Path(value).name)
                    continue
                value_path = Path(value).expanduser()
                if not value_path.is_absolute() and "/" not in value and "\\" not in value:
                    executables[key] = str(bin_root / value)

        nodes.append(
            NodeResource(
                name=name,
                host=values.get("host", "localhost").strip(),
                workspace=Path(workspace).expanduser() if workspace else None,
                gpus=values.get("gpus", values.get("gpu_list")),
                gpu_memory_mib=values.get("gpu_memory_mib"),
                cpu_workers=int(values["cpu_workers"]) if values.get("cpu_workers") else None,
                executables=executables,
                max_jobs=int(values.get("max_jobs", 1)),
            )
        )

    if not nodes:
        raise ValueError(f"No [node.NAME] sections found in resource config: {path}")

    if parts is not None and parts > len(nodes):
        expanded: list[NodeResource] = []
        for index in range(parts):
            base = nodes[index % len(nodes)]
            workspace = base.workspace
            if workspace is not None:
                workspace = workspace.parent / f"{base.name}_part{index:04d}"
            expanded.append(
                NodeResource(
                    name=f"{base.name}_part{index:04d}",
                    host=base.host,
                    workspace=workspace,
                    gpus=base.gpus,
                    gpu_memory_mib=base.gpu_memory_mib,
                    cpu_workers=base.cpu_workers,
                    executables=base.executables,
                    max_jobs=base.max_jobs,
                )
            )
        return expanded

    return nodes[: max(1, parts)] if parts is not None else nodes


def check_node(node: NodeResource, *, sample_sac_paths: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    if node.workspace is None:
        errors.append(f"{node.name}: workspace is not configured")
        return errors

    if node.is_local:
        node.workspace.mkdir(parents=True, exist_ok=True)
        if not node.workspace.exists():
            errors.append(f"{node.name}: workspace is not writable: {node.workspace}")
        for label, exe in node.executables.items():
            if exe.upper() == "NONE":
                continue
            if not Path(exe).expanduser().is_file():
                errors.append(f"{node.name}: executable not found for {label}: {exe}")
        for sac_path in sample_sac_paths or []:
            if not Path(sac_path).expanduser().is_file():
                errors.append(f"{node.name}: sample SAC path is not readable: {sac_path}")
        return errors

    quoted_workspace = _shell_quote(node.workspace.as_posix())
    cmd = f"mkdir -p {quoted_workspace} && test -w {quoted_workspace}"
    if subprocess.run(["ssh", node.host, cmd]).returncode != 0:
        errors.append(f"{node.name}: remote workspace is not writable: {node.workspace}")
    for label, exe in node.executables.items():
        if exe.upper() == "NONE":
            continue
        if subprocess.run(["ssh", node.host, f"test -f {_shell_quote(exe)}"]).returncode != 0:
            errors.append(f"{node.name}: remote executable not found for {label}: {exe}")
    for sac_path in sample_sac_paths or []:
        if subprocess.run(["ssh", node.host, f"test -r {_shell_quote(sac_path)}"]).returncode != 0:
            errors.append(f"{node.name}: remote sample SAC path is not readable: {sac_path}")
    return errors


def _config_executables(config: Any) -> dict[str, str]:
    return {
        "sac2spec": str(config.executables.sac2spec),
        "xc": str(config.executables.xc),
        "pws": str(config.executables.pws or "NONE"),
        "tfpws": str(config.executables.tfpws or "NONE"),
    }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
