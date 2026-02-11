"""
fix_raw_filenames.py

Clean up raw parquet filenames where the round number has a leading zero.

Current bad pattern (single-digit rounds):
    data/raw/<year>/<year>_0R_<session_name>.parquet
Example:
    2022_01_Practice_1.parquet

Desired pattern:
    data/raw/<year>/<year>_R_<session_name>.parquet
Example:
    2022_1_Practice_1.parquet

Usage (from project root: race-sim-v2/):

    # See what would be renamed, without changing anything
    python fix_raw_filenames.py --dry-run

    # Actually rename files
    python fix_raw_filenames.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix raw parquet filenames with padded round numbers (01..09)."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Root directory containing per-year raw data folders (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be renamed without actually renaming files.",
    )
    return parser.parse_args()


# Pattern:
#   2022_01_Practice_1.parquet
#   ^^^^ ^^ ^^^^^^^^^^^
#   year round session
FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<round_padded>0[1-9])_(?P<rest>.+)\.parquet$"
)


def fix_filenames(raw_dir: Path, dry_run: bool = True) -> None:
    if not raw_dir.exists():
        print(f"[ERROR] Raw dir does not exist: {raw_dir}")
        return

    print(f"Scanning raw dir: {raw_dir} (dry_run={dry_run})")

    num_checked = 0
    num_matched = 0
    num_renamed = 0
    collisions = 0

    # Expect structure: raw_dir/<year>/*.parquet
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        for path in sorted(year_dir.glob("*.parquet")):
            num_checked += 1
            fname = path.name
            m = FILENAME_RE.match(fname)
            if not m:
                continue

            num_matched += 1

            year = m.group("year")
            round_padded = m.group("round_padded")  # e.g. "01"
            rest = m.group("rest")                  # e.g. "Practice_1"

            round_int = int(round_padded)           # "01" -> 1
            new_fname = f"{year}_{round_int}_{rest}.parquet"
            new_path = path.with_name(new_fname)

            if new_path.exists():
                print(
                    f"[WARN] Target already exists, skipping to avoid overwrite:\n"
                    f"       from: {path}\n"
                    f"       to:   {new_path}"
                )
                collisions += 1
                continue

            print(f"[RENAME] {path}  ->  {new_path}")
            if not dry_run:
                path.rename(new_path)
                num_renamed += 1

    print("\nSummary:")
    print(f"  Files checked : {num_checked}")
    print(f"  Files matched : {num_matched}")
    print(f"  Files renamed : {num_renamed if not dry_run else 0} (dry_run={dry_run})")
    print(f"  Collisions    : {collisions}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    fix_filenames(raw_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
