from __future__ import annotations

import argparse

from fastxc.operators.sourcepack import unpack_sourcepack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unpack SourcePack indexes to SAC files.")
    parser.add_argument("-I", "--input", required=True, help="sourcepack directory or sourcepack_index.tsv")
    parser.add_argument("-O", "--output", required=True, help="output directory for unpacked files")
    parser.add_argument("-T", "--threads", type=int, default=1, help="parallel output file workers")
    parser.add_argument("--dry-run", action="store_true", help="count outputs without writing files")
    args = parser.parse_args(argv)

    result = unpack_sourcepack(
        args.input,
        args.output,
        max_workers=args.threads,
        dry_run=args.dry_run,
    )
    verb = "Would unpack" if args.dry_run else "Unpacked"
    print(
        f"{verb} {result.record_count} record(s) into {result.file_count} file(s), "
        f"{result.bytes_written} byte(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
