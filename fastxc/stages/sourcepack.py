from __future__ import annotations

import logging

from fastxc.operators.sourcepack import build_sourcepack

from .base import StageContext, skip_stage, stage_done, wants_command, wants_deploy, wants_prepare

logger = logging.getLogger(__name__)


class SourcePackStage:
    name = "SourcePack"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        cfg = ctx.cfg
        if not cfg.sourcepack.enabled:
            logger.info("SourcePack disabled by [sourcepack].enabled=False.")
            return
        if not (wants_prepare(mode) or wants_command(mode) or wants_deploy(mode)):
            return

        out = cfg.storage.output_dir
        if ctx.async_sourcepack_result is not None:
            result = ctx.async_sourcepack_result
            stage_done(
                "Source index finished asynchronously: %s (%d source(s), %d record(s)).",
                result.output_dir,
                result.source_count,
                result.record_count,
            )
            return
        if cfg.debug.dry_run:
            logger.info("SourcePack dry-run: would read %s and write %s.", out / "ncf", out / "sourcepack")
            return

        result = build_sourcepack(
            out / "ncf",
            out / "sourcepack",
            sort_within_source=cfg.sourcepack.sort_within_source,
            progress_file=out / "progress" / "sourcepack_progress.tsv",
        )
        stage_done(
            "Source index finished: %s (%d source(s), %d record(s)).",
            result.output_dir,
            result.source_count,
            result.record_count,
        )
