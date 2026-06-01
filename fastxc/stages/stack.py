from __future__ import annotations

import logging

from fastxc.io.sourcepack import discover_workspace_sourcepack_inputs
from fastxc.operators.stacking import linear_stack_sourcepack_indexes
from fastxc.adapters import gen_pws_sourcepack_cmd, gen_tfpws_sourcepack_cmd, weighted_stack_deployer

from .base import StageContext, skip_stage, stage_done, wants_command, wants_deploy, wants_prepare

logger = logging.getLogger(__name__)


class LinearStackStage:
    name = "LinearStack"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        cfg = ctx.cfg
        if cfg.stack.stack_flag[0] != "1":
            logger.info("Linear stack disabled by stack_flag=%s.", cfg.stack.stack_flag)
            return

        out = cfg.storage.output_dir
        if wants_prepare(mode) or wants_command(mode):
            try:
                indexes = discover_workspace_sourcepack_inputs(out)
            except FileNotFoundError:
                logger.warning("No SourcePack index found under %s; skip linear stack manifest.", out / "sourcepack")
                return
            manifest = out / "stack" / "linearstack_sourcepack" / "manifests" / "linearstack_inputs.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(str(path) for path in indexes), encoding="utf-8")
            logger.info("Linear stack input manifest saved to %s (%d input(s)).", manifest, len(indexes))

        if wants_deploy(mode):
            try:
                indexes = discover_workspace_sourcepack_inputs(out)
            except FileNotFoundError as exc:
                logger.warning("%s; skip linear stack.", exc)
                return
            results = linear_stack_sourcepack_indexes(
                indexes,
                out / "stack",
                dry_run=cfg.debug.dry_run,
            )
            stage_done("Linear stack sourcepack finished (%d records).", len(results))


class WeightedStackStage:
    def __init__(self, name: str, method: str, flag_index: int):
        self.name = name
        self.method = method
        self.flag_index = flag_index

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        cfg = ctx.cfg
        if cfg.stack.stack_flag[self.flag_index] != "1":
            logger.info("%s disabled by stack_flag=%s.", self.name, cfg.stack.stack_flag)
            return

        exe = cfg.executables.pws if self.method == "pws" else cfg.executables.tfpws
        out = cfg.storage.output_dir
        commands: list[str] = []
        if wants_command(mode) or wants_deploy(mode):
            try:
                if self.method == "pws":
                    commands = gen_pws_sourcepack_cmd(
                        stack_exe=exe,
                        output_dir=out,
                        sub_stack_size=cfg.stack.sub_stack_size,
                        gpu_ids=cfg.device.gpu_list,
                        gpu_memory_mib=cfg.device.gpu_memory_mib,
                        cpu_workers=cfg.device.cpu_workers,
                    )
                elif self.method == "tfpws":
                    commands = gen_tfpws_sourcepack_cmd(
                        stack_exe=exe,
                        output_dir=out,
                        sub_stack_size=cfg.stack.sub_stack_size,
                        gpu_ids=cfg.device.gpu_list,
                        gpu_memory_mib=cfg.device.gpu_memory_mib,
                        tfpws_band=cfg.stack.tfpws_band,
                        tfpws_taper_hz=cfg.stack.tfpws_taper_hz,
                    )
                else:
                    logger.info("%s skipped: unsupported method %s.", self.name, self.method)
                    return
            except FileNotFoundError as exc:
                logger.warning("%s; skip %s.", exc, self.name)
                return
        if wants_deploy(mode):
            weighted_stack_deployer(commands, out, self.method, cfg.debug.log_file_path, cfg.debug.dry_run)
            stage_done("%s sourcepack stack finished.", self.method.upper())
