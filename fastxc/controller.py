from __future__ import annotations

import logging
from difflib import get_close_matches

from .config_parser import Config, ConfigError
from .inventory import require_inventory
from .stages import (
    COMPUTE_STEPS,
    PREPARE_STEPS,
    VALID_MODES,
    StageContext,
    StepMode,
    compute_stages,
    normalize_mode,
    prepare_stages,
    run_stages,
)
from .system import configure_logging

logger = logging.getLogger(__name__)


class FastXCController:
    """Thin workflow controller for FastXC.

    Stage implementations live under :mod:`fastxc.stages`; this class only
    loads configuration, resolves step modes, and runs the stage sequence.
    """

    def __init__(self, ini_path: str):
        self.cfg = Config(ini_path)
        self.cfg.validate_all()
        configure_logging(log_file_path=self.cfg.debug.log_file_path)
        self.context = StageContext(self.cfg)

    def prepare(self, steps_config: dict | None = None) -> None:
        modes = self._modes(steps_config)
        run_stages(prepare_stages(), self.context, modes)

        meta_path = self.cfg.storage.output_dir / "inventory.meta.json"
        logger.info(
            "XC inventory prepared: %s",
            meta_path if meta_path.is_file() else self.cfg.storage.output_dir,
        )

    def compute(self, steps_config: dict | None = None) -> None:
        require_inventory(self.cfg)
        modes = self._modes(steps_config)
        run_stages(compute_stages(), self.context, modes)

    def run_all(self, steps_config: dict | None = None) -> None:
        self.prepare(steps_config)
        self.compute(steps_config)

    def _modes(self, steps_config: dict | None) -> dict[str, str]:
        names = PREPARE_STEPS + COMPUTE_STEPS
        modes = {name: "ALL" for name in names}
        if not steps_config:
            return modes

        for name, raw_mode in steps_config.items():
            if name not in modes:
                guess = get_close_matches(name, names, n=1, cutoff=0.6)
                hint = f" Did you mean '{guess[0]}'?" if guess else ""
                raise ConfigError(f"Unrecognized step '{name}'.{hint}")

            mode = normalize_mode(raw_mode)
            if mode not in VALID_MODES:
                valid = ", ".join(sorted(VALID_MODES))
                raise ConfigError(f"Invalid mode '{raw_mode}' for step '{name}'. Valid modes: {valid}")
            modes[name] = mode
        return modes
