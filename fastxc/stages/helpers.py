from __future__ import annotations

from pathlib import Path


def enabled_stack_methods(cfg) -> list[str]:
    methods: list[str] = []
    if cfg.stack.stack_flag[0] == "1":
        methods.append("linearstack")
    if cfg.stack.stack_flag[1] == "1":
        methods.append("pws")
    if cfg.stack.stack_flag[2] == "1":
        methods.append("tfpws")
    return methods


def unpack_product_name(target_name: str) -> str:
    rotate = target_name.startswith("rtz_")
    method = target_name.removeprefix("rtz_").removesuffix("_sourcepack")
    if method == "linearstack":
        method = "linear"
    coord = "RTZ" if rotate else "Z"
    return f"ncf_{method}_{coord}"


def unpack_output_root(cfg) -> Path:
    return (cfg.storage.output_dir / "result_ncf").resolve()


def unpack_targets(cfg) -> list[tuple[str, Path]]:
    out = cfg.storage.output_dir
    target = cfg.unpack.target
    methods = enabled_stack_methods(cfg)
    stack = [
        (unpack_product_name(f"{method}_sourcepack"), out / "stack" / f"{method}_sourcepack" / "STACK")
        for method in methods
    ]
    rotate = [
        (unpack_product_name(f"rtz_{method}_sourcepack"), out / "stack" / f"rtz_{method}_sourcepack" / "STACK")
        for method in methods
    ]

    if target == "STACK":
        return stack
    if target == "ROTATE":
        return rotate
    if target == "ALL":
        return stack + rotate

    existing_rotate = [(name, path) for name, path in rotate if path.exists()]
    if existing_rotate:
        return existing_rotate
    existing_stack = [(name, path) for name, path in stack if path.exists()]
    if existing_stack:
        return existing_stack
    return []
