from __future__ import annotations

import logging

from fastxc.operators.sourcepack import unpack_sourcepack

from .base import StageContext, skip_stage, stage_done, wants_deploy
from .helpers import unpack_output_root, unpack_targets

logger = logging.getLogger(__name__)


class UnpackStage:
    name = "Unpack"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        cfg = ctx.cfg
        if not cfg.unpack.enabled:
            logger.info("Unpack disabled by [unpack].enabled=False.")
            return
        if not wants_deploy(mode):
            return

        output_root = unpack_output_root(cfg)
        targets = unpack_targets(cfg)
        if not targets:
            logger.warning("No final SourcePack output found for unpack target=%s.", cfg.unpack.target)
            return

        threads = cfg.unpack.threads or cfg.device.cpu_workers
        total_files = 0
        total_records = 0
        total_bytes = 0
        for name, sourcepack_dir in targets:
            if not sourcepack_dir.exists():
                logger.info("Unpack target missing, skip %s: %s", name, sourcepack_dir)
                continue
            result = unpack_sourcepack(
                sourcepack_dir,
                output_root / name,
                max_workers=threads,
                dry_run=cfg.debug.dry_run,
            )
            total_files += result.file_count
            total_records += result.record_count
            total_bytes += result.bytes_written
            logger.info(
                "Unpacked %s: %d record(s), %d file(s), %d byte(s).",
                name,
                result.record_count,
                result.file_count,
                result.bytes_written,
            )

        stage_done(
            "Unpack finished: %d record(s), %d file(s), %d byte(s), output=%s.",
            total_records,
            total_files,
            total_bytes,
            output_root,
        )
