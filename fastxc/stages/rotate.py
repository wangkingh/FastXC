from __future__ import annotations

import logging

from fastxc.operators.rotation import rotate_sourcepack_stack

from .base import StageContext, skip_stage, stage_done, wants_deploy
from .helpers import enabled_stack_methods

logger = logging.getLogger(__name__)


class RotateStage:
    name = "Rotate"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        cfg = ctx.cfg
        if len(cfg.primary_component_list) != 3:
            logger.warning("Rotation needs 3 components; skipping.")
            return

        methods = enabled_stack_methods(cfg)
        if not methods:
            logger.info("Rotate skipped because no stack method is enabled.")
            return
        if not wants_deploy(mode):
            return

        total = 0
        for method in methods:
            try:
                results = rotate_sourcepack_stack(
                    cfg.storage.output_dir,
                    method,
                    dry_run=cfg.debug.dry_run,
                )
            except FileNotFoundError as exc:
                logger.warning("%s; skip %s rotate.", exc, method)
                continue
            total += len(results)
            logger.info("Rotate %s sourcepack finished (%d pair group(s)).", method, len(results))
        stage_done("Rotate sourcepack finished (%d pair group(s) across %d method(s)).", total, len(methods))
