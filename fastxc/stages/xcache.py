from __future__ import annotations

from fastxc.operators.xcache import build_xcache

from .base import StageContext, skip_stage, stage_done, wants_command, wants_deploy, wants_prepare


class XCacheStage:
    name = "XCache"

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        del modes
        if skip_stage(self.name, mode):
            return
        if not (wants_prepare(mode) or wants_command(mode) or wants_deploy(mode)):
            return

        cfg = ctx.cfg
        if ctx.async_xcache_result is not None:
            result = ctx.async_xcache_result
            stage_done(
                "XCache finished asynchronously: %s (%d timestamp(s), %d shard(s)).",
                result.index_path,
                result.timestamp_count,
                result.shard_count,
            )
            return

        index_path = build_xcache(
            cfg.storage.output_dir,
            windows_per_xcache=cfg.xcache.windows_per_xcache,
        )
        stage_done("XCache finished: %s.", index_path)
