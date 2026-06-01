from __future__ import annotations

import logging
from pathlib import Path

from fastxc.operators.cleanup import AsyncSpackSweeper
from fastxc.operators.xcache import AsyncXCacheMaterializer
from fastxc.adapters import gen_sac2spec_cmd, sac2spec_deployer

from .base import StageContext, skip_stage, stage_done, wants_command, wants_deploy

logger = logging.getLogger(__name__)


class Sac2SpecStage:
    name = "Sac2Spec"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        if skip_stage(self.name, mode):
            return

        cfg = ctx.cfg
        commands: list[str] = []
        if wants_command(mode) or wants_deploy(mode):
            pp = cfg.preprocess
            commands = gen_sac2spec_cmd(
                component_num=len(cfg.primary_component_list),
                sac2spec_exe=cfg.executables.sac2spec,
                output_dir=cfg.storage.output_dir,
                sac_len=pp.sac_len,
                win_len=pp.win_len,
                shift_len=pp.shift_len,
                normalize=pp.normalize,
                cpu_workers=cfg.device.cpu_workers,
                whiten=pp.whiten,
                skip_step=pp.skip_step,
                xcorr_lag_sec=cfg.xcorr.max_lag,
                gpu_ids=cfg.device.gpu_list,
                gpu_memory_mib=cfg.device.gpu_memory_mib,
                output_phase_only=pp.output_phase_only,
                debug_mode=cfg.debug.debug,
            )

        if wants_deploy(mode):
            sweeper = _start_spack_sweeper(ctx, modes.get("XCache"))
            materializer = _start_xcache_materializer(ctx, modes.get("XCache"))
            side_progress = _side_progress_files(materializer=materializer, sweeper=sweeper)
            finish_error: BaseException | None = None
            try:
                sac2spec_deployer(
                    commands,
                    cfg.storage.output_dir,
                    cfg.debug.log_file_path,
                    cfg.debug.dry_run,
                    side_progress_files=side_progress,
                )
            finally:
                if materializer is not None:
                    try:
                        result = materializer.finish()
                        if result.timestamp_count > 0:
                            ctx.async_xcache_result = result
                        else:
                            logger.info("Async XCache found no timestamp spack; XCache stage will use the synchronous path.")
                    except BaseException as exc:
                        finish_error = exc
                if sweeper is not None:
                    try:
                        ctx.async_spack_sweep_result = sweeper.finish()
                        logger.info(
                            "Async spack sweeper finished: %d timestamp(s), %.3f GiB deleted.",
                            ctx.async_spack_sweep_result.deleted_count,
                            ctx.async_spack_sweep_result.bytes_deleted / 1024**3,
                        )
                    except BaseException as exc:
                        if finish_error is None:
                            finish_error = exc
                if finish_error is not None:
                    raise finish_error

            stage_done("SAC2SPEC finished.")


def _start_xcache_materializer(ctx: StageContext, xcache_mode: str | None) -> AsyncXCacheMaterializer | None:
    cfg = ctx.cfg
    if xcache_mode is None or not wants_deploy(xcache_mode):
        return None
    if not cfg.xcache.async_after_sac2spec or cfg.debug.dry_run:
        return None

    out = cfg.storage.output_dir
    cleanup_marker_dir = None
    if cfg.xcache.cleanup_timestamp_spack:
        cleanup_marker_dir = out / "xcache" / "_cleanup" / "spack_ready"
    materializer = AsyncXCacheMaterializer(
        out,
        out / "xcache",
        windows_per_xcache=cfg.xcache.windows_per_xcache,
        progress_file=out / "progress" / "xcache_progress.tsv",
        poll_interval_sec=cfg.xcache.async_poll_sec,
        cleanup_marker_dir=cleanup_marker_dir,
    )
    materializer.start()
    logger.info("Async XCache materializer started: %s", out / "xcache")
    return materializer


def _start_spack_sweeper(ctx: StageContext, xcache_mode: str | None) -> AsyncSpackSweeper | None:
    cfg = ctx.cfg
    if xcache_mode is None or not wants_deploy(xcache_mode):
        return None
    if not cfg.xcache.async_after_sac2spec or not cfg.xcache.cleanup_timestamp_spack:
        return None
    if cfg.debug.dry_run:
        return None

    out = cfg.storage.output_dir
    marker_dir = out / "xcache" / "_cleanup" / "spack_ready"
    sweeper = AsyncSpackSweeper(
        out,
        marker_dir,
        progress_file=out / "progress" / "spack_sweeper_progress.tsv",
        poll_interval_sec=cfg.xcache.async_poll_sec,
    )
    sweeper.start()
    logger.info("Async spack sweeper started: %s", marker_dir)
    return sweeper


def _side_progress_files(
    *,
    materializer: AsyncXCacheMaterializer | None,
    sweeper: AsyncSpackSweeper | None,
) -> dict[str, str | Path] | None:
    side: dict[str, str | Path] = {}
    if materializer is not None and materializer.progress_file is not None:
        side["xcache"] = materializer.progress_file
    if sweeper is not None and sweeper.progress_file is not None:
        side["spack-cleanup"] = sweeper.progress_file
    return side or None
