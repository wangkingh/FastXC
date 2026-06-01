from __future__ import annotations

import logging
from pathlib import Path

from fastxc.operators.sourcepack import AsyncSourcePackMaterializer
from fastxc.adapters import gen_xc_cmd, xc_deployer

from .base import StageContext, skip_stage, stage_done, wants_command, wants_deploy

logger = logging.getLogger(__name__)


class CrossCorrelationStage:
    name = "CrossCorrelation"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        if skip_stage(self.name, mode):
            return

        cfg = ctx.cfg
        out = cfg.storage.output_dir
        commands: list[str] = []
        if wants_command(mode) or wants_deploy(mode):
            commands = gen_xc_cmd(
                xcspec_index_file=out / "xcache" / "xcspec_index.tsv",
                allowed_paths_file=out / "path_plan" / "allowed_paths.tsv",
                output_dir=out,
                xc_exe=cfg.executables.xc,
                ncf_dir=out / "ncf",
                cclength=cfg.xcorr.max_lag,
                gpu_ids=cfg.device.gpu_list,
                gpu_memory_mib=cfg.device.gpu_memory_mib,
                cpu_workers=cfg.device.cpu_workers,
                debug_mode=cfg.debug.debug,
            )
        if wants_deploy(mode):
            materializer = _start_sourcepack_materializer(ctx, modes.get("SourcePack"))
            side_progress = _side_progress_files(materializer)
            try:
                xc_deployer(
                    commands,
                    out,
                    cfg.debug.log_file_path,
                    cfg.debug.dry_run,
                    side_progress_files=side_progress,
                )
            finally:
                if materializer is not None:
                    ctx.async_sourcepack_result = materializer.finish()
            stage_done("Cross-correlation finished.")


def _start_sourcepack_materializer(ctx: StageContext, sourcepack_mode: str | None) -> AsyncSourcePackMaterializer | None:
    cfg = ctx.cfg
    if sourcepack_mode is None or not wants_deploy(sourcepack_mode):
        return None
    if not cfg.sourcepack.async_after_xc or cfg.debug.dry_run:
        return None

    out = cfg.storage.output_dir
    materializer = AsyncSourcePackMaterializer(
        out / "ncf",
        out / "sourcepack",
        sort_within_source=cfg.sourcepack.sort_within_source,
        progress_file=out / "progress" / "sourcepack_progress.tsv",
        poll_interval_sec=cfg.sourcepack.async_poll_sec,
    )
    materializer.start()
    logger.info("Async SourcePack materializer started: %s", out / "sourcepack")
    return materializer


def _side_progress_files(materializer: AsyncSourcePackMaterializer | None) -> dict[str, str | Path] | None:
    if materializer is None or materializer.progress_file is None:
        return None
    return {"sourcepack": materializer.progress_file}
