from __future__ import annotations

import argparse
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
    if args.command == "sac2spec":
        return _cmd_sac2spec_stage(args)
    if args.command == "xc":
        return _cmd_xc_stage(args)
    if args.command == "stack":
        return _cmd_stack_stage(args)
    if args.command == "rotate":
        return _cmd_rotate_stage(args)
    if args.command == "sac2dat":
        return _cmd_sac2dat(args)
    if args.command == "sourcepack":
        return _cmd_sourcepack(args)
    if args.command == "unpack":
        return _cmd_unpack(args)
    if args.command == "plot-rtz-grid":
        return _cmd_plot_rtz_grid(args)
    if args.command == "extract-ncf":
        return _cmd_extract_ncf(args)
    if args.command == "extract-stepack":
        return _cmd_extract_stepack(args)
    if args.command == "plot-stepack-mat":
        return _cmd_plot_stepack_mat(args)

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

    sac2spec_stage_parser = sub.add_parser("sac2spec", help="run only the SAC2SPEC stage from config")
    sac2spec_stage_parser.add_argument("config", help="path to config.ini")

    xc_stage_parser = sub.add_parser("xc", help="run XC from prepared stepack, then build SourcePack")
    xc_stage_parser.add_argument("config", help="path to config.ini")
    xc_stage_parser.add_argument("--no-sourcepack", action="store_true", help="skip SourcePack indexing after XC")

    stack_stage_parser = sub.add_parser("stack", help="run stack stage(s) from SourcePack")
    stack_stage_parser.add_argument("config", help="path to config.ini")
    stack_stage_parser.add_argument(
        "--method",
        default="all",
        help="comma-separated stack method(s): all, linear, pws, tfpws; enabled methods still honor stack_flag",
    )

    rotate_stage_parser = sub.add_parser("rotate", help="run rotation stage from stack SourcePack")
    rotate_stage_parser.add_argument("config", help="path to config.ini")

    sac2dat_parser = sub.add_parser("sac2dat", help="convert stacked SAC files to DAT text files")
    sac2dat_parser.add_argument("-I", "--input", required=True, help="input directory containing .sac files")
    sac2dat_parser.add_argument("-O", "--output", required=True, help="output DAT directory")

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

    extract_ncf_parser = sub.add_parser("extract-ncf", help="extract one NCF SAC record from SourcePack or XC pack indexes")
    extract_ncf_input = extract_ncf_parser.add_mutually_exclusive_group(required=True)
    extract_ncf_input.add_argument(
        "-I",
        "--input",
        help="SourcePack index/directory, xcpack directory, or XC output root",
    )
    extract_ncf_input.add_argument("--workspace", help="FastXC workspace containing sourcepack/ or ncf/xcpack/")
    extract_ncf_parser.add_argument("--timestamp", help="timestamp to extract; required for workspace/xcpack inputs")
    extract_ncf_parser.add_argument("--source", required=True, help="source station code")
    extract_ncf_parser.add_argument("--receiver", required=True, help="receiver station code")
    extract_ncf_parser.add_argument("--component-pair", required=True, help="component pair, e.g. BHE-BHZ or R-Z")
    extract_ncf_parser.add_argument("--src-network", help="optional source network filter")
    extract_ncf_parser.add_argument("--rec-network", help="optional receiver network filter")
    extract_ncf_parser.add_argument("--src-location", help="optional source location filter")
    extract_ncf_parser.add_argument("--rec-location", help="optional receiver location filter")
    extract_ncf_parser.add_argument("--allow-reverse", action="store_true", help="also match receiver/source reversed")
    extract_ncf_parser.add_argument("--dry-run", action="store_true", help="show the matched record without writing")
    extract_ncf_parser.add_argument("-O", "--output", required=True, help="output SAC path")

    plot_rtz_parser = sub.add_parser(
        "plot-rtz-grid",
        help="plot a single-component or 3x3 gather from unpacked result_ncf SAC files",
    )
    plot_rtz_parser.add_argument("-I", "--input", required=True, help="unpacked result_ncf SAC directory")
    plot_rtz_parser.add_argument("--source", required=True, help="virtual source station or key")
    plot_rtz_parser.add_argument("-O", "--output", help="output PNG path")
    plot_rtz_parser.add_argument("--title", default="AUTO", help="figure title")
    plot_rtz_parser.add_argument("--receiver", action="append", default=[], help="receiver station/key to include")
    plot_rtz_parser.add_argument("--lag-window", type=float, default=20.0, help="half window in seconds around zero lag")
    plot_rtz_parser.add_argument("--scale", type=float, default=0.0, help="amplitude scale in km; 0 means auto")
    plot_rtz_parser.add_argument("--max-receivers", type=int, default=0, help="maximum receivers to draw")
    plot_rtz_parser.add_argument("--sample-stride", type=int, default=1, help="draw every Nth receiver")
    plot_rtz_parser.add_argument("--min-distance", type=float, help="minimum receiver distance in km")
    plot_rtz_parser.add_argument("--max-distance", type=float, help="maximum receiver distance in km")
    plot_rtz_parser.add_argument("--linewidth", type=float, default=0.55, help="trace line width")
    plot_rtz_parser.add_argument("--dpi", type=int, default=180, help="output figure DPI")

    extract_stepack_parser = sub.add_parser("extract-stepack", help="extract one station's spectra from stepack")
    extract_stepack_parser.add_argument("--workspace", help="FastXC workspace containing stepack/")
    extract_stepack_parser.add_argument("--stepack", help="stepack directory or a single stepack TSV")
    extract_stepack_parser.add_argument("--timestamp", required=True, help="timestamp to extract")
    extract_stepack_parser.add_argument("--station", required=True, help="station code or full key fragment")
    extract_stepack_parser.add_argument("--network", help="optional network filter")
    extract_stepack_parser.add_argument("--location", help="optional location filter")
    extract_stepack_parser.add_argument("--components", default="ALL", help="comma-separated components; default ALL")
    extract_stepack_parser.add_argument(
        "--component-match",
        choices=("exact", "tail", "auto"),
        default="auto",
        help="component matching mode for --components",
    )
    extract_stepack_parser.add_argument("-O", "--output", required=True, help="output .mat path")
    extract_stepack_parser.add_argument("--no-compress", action="store_true", help="disable .mat compression")
    extract_stepack_parser.add_argument("--plot", action="store_true", help="also write a quick-look PNG")
    extract_stepack_parser.add_argument("--plot-output", help="output PNG path; defaults beside the .mat file")
    extract_stepack_parser.add_argument(
        "--quantity",
        choices=("amplitude", "power", "phase", "real", "imag"),
        default="amplitude",
        help="spectrum quantity to plot when --plot is enabled",
    )
    extract_stepack_parser.add_argument("--db", action="store_true", help="plot amplitude/power in dB")
    extract_stepack_parser.add_argument("--min-frequency", type=float, default=0.0, help="minimum frequency in Hz")
    extract_stepack_parser.add_argument("--max-frequency", type=float, help="maximum frequency in Hz")
    extract_stepack_parser.add_argument("--smooth-step", type=float, default=0.35, help="Gaussian smoothing sigma along step axis")
    extract_stepack_parser.add_argument(
        "--smooth-frequency",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma along frequency-bin axis",
    )
    extract_stepack_parser.add_argument("--no-smooth", action="store_true", help="disable plot smoothing")
    extract_stepack_parser.add_argument("--plot-title", default="AUTO", help="figure title for --plot")
    extract_stepack_parser.add_argument("--dpi", type=int, default=180, help="output figure DPI for --plot")

    plot_stepack_parser = sub.add_parser("plot-stepack-mat", help="plot spectra exported by extract-stepack")
    plot_stepack_parser.add_argument("-I", "--input", required=True, help="input .mat file")
    plot_stepack_parser.add_argument("-O", "--output", required=True, help="output PNG path")
    plot_stepack_parser.add_argument(
        "--quantity",
        choices=("amplitude", "power", "phase", "real", "imag"),
        default="amplitude",
        help="spectrum quantity to plot",
    )
    plot_stepack_parser.add_argument("--db", action="store_true", help="plot amplitude/power in dB")
    plot_stepack_parser.add_argument("--min-frequency", type=float, default=0.0, help="minimum frequency in Hz")
    plot_stepack_parser.add_argument("--max-frequency", type=float, help="maximum frequency in Hz")
    plot_stepack_parser.add_argument("--smooth-step", type=float, default=0.35, help="Gaussian smoothing sigma along step axis")
    plot_stepack_parser.add_argument(
        "--smooth-frequency",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma along frequency-bin axis",
    )
    plot_stepack_parser.add_argument("--no-smooth", action="store_true", help="disable plot smoothing")
    plot_stepack_parser.add_argument("--title", default="AUTO", help="figure title")
    plot_stepack_parser.add_argument("--dpi", type=int, default=180, help="output figure DPI")

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
            print("  XC data source: stepack")
            print(f"  autocorr_mode: {cfg.xcorr.autocorr_mode}")
            print(f"  async_poll_sec: {cfg.sourcepack.async_poll_sec:g}")
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


def _cmd_sac2spec_stage(args: argparse.Namespace) -> int:
    return _run_config_stages(args.config, ["Sac2Spec"])


def _cmd_xc_stage(args: argparse.Namespace) -> int:
    stages = ["CrossCorrelation"]
    if not args.no_sourcepack:
        stages.append("SourcePack")
    return _run_config_stages(args.config, stages)


def _cmd_stack_stage(args: argparse.Namespace) -> int:
    try:
        stages = _stack_stage_names(args.method)
    except ValueError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return _run_config_stages(args.config, stages)


def _cmd_rotate_stage(args: argparse.Namespace) -> int:
    return _run_config_stages(args.config, ["Rotate"])


def _run_config_stages(config: str, stages: list[str]) -> int:
    try:
        FastXCController(config).compute(_only_stage_modes(stages))
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


def _only_stage_modes(stages: list[str]) -> dict[str, str]:
    from .stages import COMPUTE_STEPS

    selected = set(stages)
    return {name: ("ALL" if name in selected else "SKIP") for name in COMPUTE_STEPS}


def _stack_stage_names(methods: str) -> list[str]:
    mapping = {
        "linear": "LinearStack",
        "linearstack": "LinearStack",
        "pws": "PwsStack",
        "tfpws": "TfPwsStack",
    }
    requested = [item.strip().lower() for item in methods.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return ["LinearStack", "PwsStack", "TfPwsStack"]
    if "all" in requested:
        raise ValueError("'all' cannot be combined with explicit stack methods")

    stages: list[str] = []
    invalid: list[str] = []
    for method in requested:
        stage = mapping.get(method)
        if stage is None:
            invalid.append(method)
        elif stage not in stages:
            stages.append(stage)
    if invalid:
        raise ValueError(f"unsupported stack method(s): {', '.join(invalid)}")
    return stages


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


def _cmd_plot_rtz_grid(args: argparse.Namespace) -> int:
    from .tools.plot_rtz_grid import run

    try:
        result = run(args)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Wrote {result.output_path}")
    print(
        f"Source {result.source_key}: plotted {result.receiver_count} receiver(s), "
        f"{result.trace_count} trace(s)."
    )
    return 0


def _cmd_extract_ncf(args: argparse.Namespace) -> int:
    from .tools.extract_ncf import run

    try:
        result = run(args)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    if args.dry_run:
        print("Matched NCF record:")
        print(f"  index: {result.index_path}")
        print(f"  pack: {result.record_path}")
        print(f"  offset: {result.record_offset}")
        print(f"  bytes: {result.record_bytes}")
        print(
            "  pair: "
            f"{result.row.get('src_station')}-{result.row.get('rec_station')} "
            f"{result.row.get('src_component')}-{result.row.get('rec_component')}"
        )
        if result.reversed_match:
            print("  match: reversed")
    else:
        print(
            f"Wrote {result.output_path} "
            f"({result.record_bytes} byte(s) from {result.record_path} offset {result.record_offset})."
        )
        if result.reversed_match:
            print("Matched reversed source/receiver order.")
    return 0



def _cmd_extract_stepack(args: argparse.Namespace) -> int:
    from .tools.extract_stepack import run

    try:
        result = run(args)
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(
        f"Wrote {result.output_path} "
        f"({result.component_count} component(s), {result.nstep} step(s), {result.nspec} frequency bin(s))."
    )
    if result.plot_output_path is not None:
        print(f"Wrote {result.plot_output_path}")
    return 0


def _cmd_plot_stepack_mat(args: argparse.Namespace) -> int:
    from .tools.plot_stepack_mat import plot_stepack_mat

    try:
        output = plot_stepack_mat(
            args.input,
            output=args.output,
            max_frequency=args.max_frequency,
            min_frequency=args.min_frequency,
            quantity=args.quantity,
            db=args.db,
            smooth_step=0.0 if args.no_smooth else args.smooth_step,
            smooth_frequency=0.0 if args.no_smooth else args.smooth_frequency,
            dpi=args.dpi,
            title=args.title,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
