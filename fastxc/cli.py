from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

from .config_parser import Config, ConfigError
from .controller import FastXCController
from .system import configure_logging, executable_report, write_template_config


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)
    configure_logging(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)

    if args.command == "init":
        return _cmd_init(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "run-plan":
        return _cmd_run_plan(args)
    if args.command == "collect-plan":
        return _cmd_collect_plan(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "sac2dat":
        return _cmd_sac2dat(args)
    if args.command == "extract":
        return _cmd_extract(args)
    if args.command == "sourcepack":
        return _cmd_sourcepack(args)
    if args.command == "unpack":
        return _cmd_unpack(args)
    if args.command == "decode-spack":
        return _cmd_decode_spack(args)
    if args.command == "inspect-xcache":
        return _cmd_inspect_xcache(args)

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastxc", description="FastXC pipeline runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose Python logging")
    sub = parser.add_subparsers(dest="command")

    init_parser = sub.add_parser("init", help="write a starter config.ini")
    init_parser.add_argument("-o", "--output", default="config.ini", help="output config path")
    init_parser.add_argument("-f", "--force", action="store_true", help="overwrite output if it exists")

    doctor_parser = sub.add_parser("doctor", help="check executable discovery and optional config parsing")
    doctor_parser.add_argument("config", nargs="?", help="optional config.ini to parse")

    prepare_parser = sub.add_parser("prepare", help="prepare the reusable XC inventory")
    prepare_parser.add_argument("config", help="path to config.ini")

    plan_parser = sub.add_parser("plan", help="write a static timestamp task plan")
    plan_parser.add_argument("config", help="path to config.ini")
    plan_parser.add_argument("-R", "--resources", help="optional local/remote resource ini")
    plan_parser.add_argument("-N", "--parts", type=int, help="number of timestamp task slices")
    plan_parser.add_argument("-O", "--output", help="plan output directory; default workspace/distributed")
    plan_parser.add_argument("-f", "--force", action="store_true", help="replace an existing plan directory")
    plan_parser.add_argument("--no-check", action="store_true", help="skip executable/workspace/sample SAC checks")

    run_plan_parser = sub.add_parser("run-plan", help="run task configs from a static plan")
    run_plan_parser.add_argument("run_plan", help="path to distributed/run_plan.tsv")
    run_plan_parser.add_argument("--repo", help="FastXC repository path for task commands")
    run_plan_parser.add_argument("--python", dest="python_exe", help="Python executable for task commands")
    run_plan_parser.add_argument("-j", "--jobs", type=int, default=1, help="number of tasks to run concurrently")
    run_plan_parser.add_argument("--no-collect", action="store_true", help="do not collect sourcepack index list after tasks")
    run_plan_parser.add_argument("--main-workspace", help="workspace where sourcepack_inputs.txt should be mirrored")

    collect_plan_parser = sub.add_parser("collect-plan", help="collect task SourcePack indexes into one input list")
    collect_plan_parser.add_argument("run_plan", help="path to distributed/run_plan.tsv")
    collect_plan_parser.add_argument("-O", "--output", help="output sourcepack_inputs.txt path")
    collect_plan_parser.add_argument("--main-workspace", help="workspace where sourcepack_inputs.txt should be mirrored")

    run_parser = sub.add_parser("run", help="run computation from a prepared inventory")
    run_parser.add_argument("config", help="path to config.ini")
    run_parser.add_argument(
        "--only",
        help="comma-separated compute stages to run, e.g. LinearStack,PwsStack,TfPwsStack,Rotate",
    )
    run_parser.add_argument("--skip", help="comma-separated compute stages to skip")

    sac2dat_parser = sub.add_parser("sac2dat", help="convert stacked SAC files to DAT text files")
    sac2dat_parser.add_argument("-I", "--input", required=True, help="input directory containing .sac files")
    sac2dat_parser.add_argument("-O", "--output", required=True, help="output DAT directory")

    extract_parser = sub.add_parser("extract", help="extract BigSAC files to SAC files")
    extract_parser.add_argument("-I", "--input", required=True, help="input directory containing .bigsac files")
    extract_parser.add_argument("-O", "--output", required=True, help="output directory for extracted SAC files")

    sourcepack_parser = sub.add_parser("sourcepack", help="group XC pack output by timestamp and virtual source")
    sourcepack_parser.add_argument("-I", "--input", required=True, help="XC output root or xcpack directory")
    sourcepack_parser.add_argument("-O", "--output", required=True, help="output sourcepack directory")
    sourcepack_parser.add_argument(
        "--sort",
        action="store_true",
        default=True,
        help="sort each timestamp index by source/receiver/component",
    )
    sourcepack_parser.add_argument(
        "--keep-order",
        action="store_false",
        dest="sort",
        help="preserve XC encounter order in the generated index",
    )

    unpack_parser = sub.add_parser("unpack", help="unpack SourcePack indexes to SAC files")
    unpack_parser.add_argument("-I", "--input", required=True, help="sourcepack directory or sourcepack_index.tsv")
    unpack_parser.add_argument("-O", "--output", required=True, help="output directory for unpacked SAC files")
    unpack_parser.add_argument("-T", "--threads", type=int, default=1, help="parallel output file workers")

    decode_spack_parser = sub.add_parser("decode-spack", help="decode SAC2SPEC spack records to SEGSPEC files")
    decode_spack_parser.add_argument("-I", "--input", required=True, help="workspace root, spack root, timestamp dir, or TSV")
    decode_spack_parser.add_argument("-O", "--output", required=True, help="output directory for decoded SEGSPEC files")
    decode_spack_parser.add_argument("--timestamp", help="only decode one timestamp")
    decode_spack_parser.add_argument("--nsl-id", type=int, help="only decode one nsl_id")
    decode_spack_parser.add_argument("--network", help="only decode one network")
    decode_spack_parser.add_argument("--station", help="only decode one station")
    decode_spack_parser.add_argument("--location", help="only decode one location")
    decode_spack_parser.add_argument("--component", help="only decode one component")
    decode_spack_parser.add_argument("--limit", type=int, help="maximum number of records to decode")
    decode_spack_parser.add_argument("--dry-run", action="store_true", help="count matching records without writing files")
    decode_spack_parser.add_argument("-f", "--force", action="store_true", help="overwrite existing decoded files")

    inspect_xcache_parser = sub.add_parser("inspect-xcache", help="inspect xcache binary shard headers")
    inspect_xcache_parser.add_argument("-I", "--input", required=True, help=".xcspec file, xcache dir, or xcspec_index.tsv")
    inspect_xcache_parser.add_argument("--sources", type=int, default=10, help="number of SourceEntry rows to print per shard")
    inspect_xcache_parser.add_argument("--hash-payload", action="store_true", help="compute SHA256 for payload bytes")
    inspect_xcache_parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        target = Path(args.output).expanduser().resolve()
        if target.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing config: {target}")
        write_template_config(target)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Wrote {target}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    print("FastXC doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("Executables:")
    for name, path in executable_report():
        print(f"  {name}: {path}")

    if args.config:
        try:
            cfg = Config(args.config)
            print(f"Config: OK ({Path(args.config).expanduser().resolve()})")
            print(f"  inventory_dir: {cfg.storage.workspace_dir}")
            print(f"  gpu_list: {','.join(str(g) for g in cfg.device.gpu_list)}")
            gpu_memory = ",".join(f"{m:g}" for m in cfg.device.gpu_memory_mib) or "AUTO"
            print(f"  gpu_memory_mib: {gpu_memory}")
            print(f"  cpu_workers: {cfg.device.cpu_workers}")
            print(f"  sac2spec: {cfg.executables.sac2spec}")
            print(f"  xc: {cfg.executables.xc}")
            print(f"  pws: {cfg.executables.pws}")
            print(f"  tfpws: {cfg.executables.tfpws}")
            windows = cfg.xcache.windows_per_xcache
            print(f"  windows_per_xcache: {windows if windows is not None else 'AUTO'}")
            print(f"  xcache_async_after_sac2spec: {cfg.xcache.async_after_sac2spec}")
            print(f"  xcache_cleanup_timestamp_spack: {cfg.xcache.cleanup_timestamp_spack}")
            print(f"  xcorr_write_mode: {cfg.xcorr.write_mode}")
            print(f"  sourcepack_enabled: {cfg.sourcepack.enabled}")
            print(f"  sourcepack_async_after_xc: {cfg.sourcepack.async_after_xc}")
            print(f"  unpack_enabled: {cfg.unpack.enabled}")
            print(f"  unpack_target: {cfg.unpack.target}")
        except ConfigError as exc:
            print(f"Config: FAILED ({exc})")
            return 1
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    try:
        FastXCController(args.config).prepare()
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .distributed import write_distributed_plan

    try:
        cfg = Config(args.config)
        cfg.validate_all()
        plan = write_distributed_plan(
            cfg,
            resources=args.resources,
            parts=args.parts,
            plan_dir=args.output,
            force=args.force,
            check=not args.no_check,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(
        f"Wrote distributed plan: {plan.run_plan} "
        f"({plan.task_count} task(s), {plan.timestamp_count} timestamp(s), {plan.row_count} SAC row(s))."
    )
    print(f"SourcePack input list placeholder: {plan.sourcepack_inputs}")
    return 0


def _cmd_run_plan(args: argparse.Namespace) -> int:
    from .distributed import run_distributed_plan

    try:
        output = run_distributed_plan(
            args.run_plan,
            repo_dir=args.repo,
            python_exe=args.python_exe,
            collect=not args.no_collect,
            main_workspace=args.main_workspace,
            jobs=args.jobs,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    if output is not None:
        print(f"Collected SourcePack input list: {output}")
    return 0


def _cmd_collect_plan(args: argparse.Namespace) -> int:
    from .distributed import collect_plan_sourcepacks

    try:
        output = collect_plan_sourcepacks(
            args.run_plan,
            output_list=args.output,
            main_workspace=args.main_workspace,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Collected SourcePack input list: {output}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        FastXCController(args.config).compute(_run_step_modes(args.only, args.skip))
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


def _run_step_modes(only: str | None, skip: str | None) -> dict[str, str] | None:
    from .stages import COMPUTE_STEPS

    if not only and not skip:
        return None

    modes: dict[str, str] = {}
    if only:
        selected = _split_stage_list(only)
        modes.update({name: "SKIP" for name in COMPUTE_STEPS})
        for name in selected:
            modes[name] = "ALL"
    if skip:
        for name in _split_stage_list(skip):
            modes[name] = "SKIP"
    return modes


def _split_stage_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _cmd_sac2dat(args: argparse.Namespace) -> int:
    from .tools.sac2dat import convert_sac_dir

    try:
        results = convert_sac_dir(args.input, args.output)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Converted {len(results)} SAC file(s) to DAT.")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from .tools.extract import extract_bigsac_dir

    try:
        count = extract_bigsac_dir(args.input, args.output)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Extracted {count} SAC segment(s).")
    return 0


def _cmd_sourcepack(args: argparse.Namespace) -> int:
    from .operators.sourcepack import build_sourcepack

    try:
        result = build_sourcepack(
            args.input,
            args.output,
            sort_within_source=args.sort,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(
        f"Built {len(result.index_paths)} timestamp source index(es), "
        f"{result.source_count} source(s), {result.record_count} record(s)."
    )
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    from .operators.sourcepack import unpack_sourcepack

    try:
        result = unpack_sourcepack(args.input, args.output, max_workers=args.threads)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(
        f"Unpacked {result.record_count} record(s) into {result.file_count} SAC file(s), "
        f"{result.bytes_written} byte(s)."
    )
    return 0


def _cmd_decode_spack(args: argparse.Namespace) -> int:
    from .tools.decode_spack import decode_spack

    try:
        result = decode_spack(
            args.input,
            args.output,
            timestamp=args.timestamp,
            nsl_id=args.nsl_id,
            network=args.network,
            station=args.station,
            location=args.location,
            component=args.component,
            limit=args.limit,
            dry_run=args.dry_run,
            overwrite=args.force,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    verb = "Would decode" if args.dry_run else "Decoded"
    print(f"{verb} {result.record_count} SEGSPEC record(s), {result.bytes_written} byte(s).")
    print(f"Output: {result.output_dir}")
    return 0


def _cmd_inspect_xcache(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .tools.inspect_xcache import format_inspections, inspect_xcache

    try:
        inspections = inspect_xcache(
            args.input,
            source_limit=max(args.sources, 0),
            hash_payload=args.hash_payload,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    if args.json:
        print(json.dumps([asdict(item) for item in inspections], indent=2, ensure_ascii=False))
    else:
        print(format_inspections(inspections))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
