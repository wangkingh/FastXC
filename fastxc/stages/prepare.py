from __future__ import annotations

import logging

from fastxc.inventory import build_inventory
from fastxc.operators.filter_design import design_filter

from .base import StageContext, skip_stage, wants_prepare

logger = logging.getLogger(__name__)


class GenerateFilterStage:
    name = "GenerateFilter"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode) or not wants_prepare(mode):
            return
        cfg = ctx.cfg
        path = cfg.storage.output_dir / "filter.txt"
        design_filter(cfg.compute.delta, cfg.compute.bands, path)
        logger.info("Filter file generated: %s", path)


class PrepareInventoryStage:
    name = "PrepareInventory"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode) or not wants_prepare(mode):
            return

        cfg = ctx.cfg
        result = build_inventory(cfg)
        plan = cfg.path_plan
        logger.info(
            "Path plan done: %d source section(s), %d group(s), "
            "%d NSL nodes, %d allowed paths, %d pair checks.",
            len(cfg.seisarrays),
            len(cfg.files_groups),
            result.nsl_count,
            result.allowed_path_count,
            plan.pair_checks_total,
        )
        logger.info(
            "Inventory written: %s (%d SAC row(s), %d timestamp(s)).",
            result.sac_index,
            result.sac_row_count,
            result.timestamp_count,
        )
