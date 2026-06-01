from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from fastxc.config_parser import Config

logger = logging.getLogger(__name__)


class StepMode:
    SKIP = "SKIP"
    PREPARE_ONLY = "PREPARE"
    CMD_ONLY = "CMD_ONLY"
    DEPLOY_ONLY = "DEPLOY"
    ALL = "ALL"


VALID_MODES = {
    StepMode.SKIP,
    StepMode.PREPARE_ONLY,
    StepMode.CMD_ONLY,
    StepMode.DEPLOY_ONLY,
    StepMode.ALL,
}

MODE_ALIASES = {
    "CMD": StepMode.CMD_ONLY,
    "COMMAND": StepMode.CMD_ONLY,
    "COMMAND_ONLY": StepMode.CMD_ONLY,
    "PREPARE_ONLY": StepMode.PREPARE_ONLY,
    "DEPLOY_ONLY": StepMode.DEPLOY_ONLY,
    "RUN": StepMode.ALL,
}

PREPARE_STEPS = ("GenerateFilter", "PrepareInventory")
COMPUTE_STEPS = (
    "Sac2Spec",
    "XCache",
    "CrossCorrelation",
    "SourcePack",
    "LinearStack",
    "PwsStack",
    "TfPwsStack",
    "Rotate",
    "Unpack",
)


@dataclass
class StageContext:
    cfg: Config
    async_sourcepack_result: object | None = None
    async_xcache_result: object | None = None
    async_spack_sweep_result: object | None = None


class Stage(Protocol):
    name: str

    def run(self, ctx: StageContext, mode: str, modes: dict[str, str]) -> None:
        ...


def normalize_mode(value: str) -> str:
    return MODE_ALIASES.get(str(value).strip().upper(), str(value).strip().upper())


def wants_prepare(mode: str) -> bool:
    return mode in {StepMode.PREPARE_ONLY, StepMode.CMD_ONLY, StepMode.ALL}


def wants_command(mode: str) -> bool:
    return mode in {StepMode.CMD_ONLY, StepMode.ALL}


def wants_deploy(mode: str) -> bool:
    return mode in {StepMode.DEPLOY_ONLY, StepMode.ALL}


def skip_stage(name: str, mode: str) -> bool:
    if mode != StepMode.SKIP:
        return False
    logger.info("Skipping %s.", name)
    return True


def stage_done(message: str, *args: object) -> None:
    logger.info(message + "\n" + "_" * 80 + "\n", *args)


def run_stages(stages: list[Stage], ctx: StageContext, modes: dict[str, str]) -> None:
    for stage in stages:
        stage.run(ctx, modes[stage.name], modes)
