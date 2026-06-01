from __future__ import annotations

from .base import (
    COMPUTE_STEPS,
    MODE_ALIASES,
    PREPARE_STEPS,
    VALID_MODES,
    StageContext,
    StepMode,
    normalize_mode,
    run_stages,
)
from .prepare import GenerateFilterStage, PrepareInventoryStage
from .rotate import RotateStage
from .sac2spec import Sac2SpecStage
from .sourcepack import SourcePackStage
from .stack import LinearStackStage, WeightedStackStage
from .unpack import UnpackStage
from .xc import CrossCorrelationStage
from .xcache import XCacheStage


def prepare_stages():
    return [
        GenerateFilterStage(),
        PrepareInventoryStage(),
    ]


def compute_stages():
    return [
        Sac2SpecStage(),
        XCacheStage(),
        CrossCorrelationStage(),
        SourcePackStage(),
        LinearStackStage(),
        WeightedStackStage("PwsStack", "pws", 1),
        WeightedStackStage("TfPwsStack", "tfpws", 2),
        RotateStage(),
        UnpackStage(),
    ]


__all__ = [
    "COMPUTE_STEPS",
    "MODE_ALIASES",
    "PREPARE_STEPS",
    "VALID_MODES",
    "StageContext",
    "StepMode",
    "compute_stages",
    "normalize_mode",
    "prepare_stages",
    "run_stages",
]
