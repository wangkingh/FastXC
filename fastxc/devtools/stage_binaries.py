from __future__ import annotations

import argparse
import platform
import shutil
import stat
from pathlib import Path


DEFAULT_BINARIES = ("sac2spec", "xc_fast", "ncf_pws", "ncf_tfpws")


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    if system.startswith("linux"):
        return f"linux-{machine}"
    return f"{system}-{machine}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def binary_name(name: str) -> str:
    return name


def stage_binaries(
    *,
    source_dir: str | Path | None = None,
    package_dir: str | Path | None = None,
    names: tuple[str, ...] = DEFAULT_BINARIES,
    allow_missing: bool = False,
) -> list[tuple[Path, Path]]:
    root = repo_root()
    source = Path(source_dir).expanduser().resolve() if source_dir else root / "bin"
    package = (
        Path(package_dir).expanduser().resolve()
        if package_dir
        else root / "fastxc" / "bin" / platform_tag()
    )
    package.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for name in names:
        src = source / binary_name(name)
        if not src.is_file():
            missing.append(src)
            continue
        dst = package / src.name
        shutil.copy2(src, dst)
        mode = dst.stat().st_mode
        dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        copied.append((src, dst))

    if missing and not allow_missing:
        preview = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing native binary/binaries. Run `make native-full` or the "
            f"appropriate native target first.\n  {preview}"
        )
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage native binaries into the FastXC Python package.")
    parser.add_argument("--source-dir", help="directory containing built native binaries")
    parser.add_argument("--package-dir", help="destination package binary directory")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="binary name to stage; repeat for multiple names",
    )
    parser.add_argument("--allow-missing", action="store_true", help="skip missing binaries")
    args = parser.parse_args(argv)

    copied = stage_binaries(
        source_dir=args.source_dir,
        package_dir=args.package_dir,
        names=tuple(args.only) if args.only else DEFAULT_BINARIES,
        allow_missing=args.allow_missing,
    )
    for src, dst in copied:
        print(f"{src} -> {dst}")
    print(f"staged {len(copied)} binary/binaries for {platform_tag()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
