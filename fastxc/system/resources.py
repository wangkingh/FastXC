from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def write_template_config(output_path: str | Path) -> Path:
    """Copy the packaged starter config to a user-editable path."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    template = resources.files("fastxc.resources").joinpath("template.ini")
    with resources.as_file(template) as template_path:
        shutil.copy2(template_path, target)
    return target
