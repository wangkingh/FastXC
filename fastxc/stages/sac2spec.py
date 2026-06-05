from __future__ import annotations

import logging

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
            pp = cfg.compute
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
                phase_only=pp.phase_only,
                debug_mode=cfg.debug.debug,
            )

        if wants_deploy(mode):
            del modes
            sac2spec_deployer(
                commands,
                cfg.storage.output_dir,
                cfg.debug.log_file_path,
                cfg.debug.dry_run,
            )

            stage_done("SAC2SPEC finished.")
