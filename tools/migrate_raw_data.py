"""
Migration script to convert old raw-data naming conventions into the new v2 layout.

Old filenames look like:
    2023_1_Practice_1.parquet
or:
    {year}_{round}_{session_name}.parquet

New structure:
    race-sim-v2/data/raw/<year>/<year>_<round>_<safe_session_name>.parquet
"""

import re
import shutil
from pathlib import Path
import pandas as pd


# =============== USER CONFIGURATION ===============

# Path to folder containing OLD parquet files
OLD_DATA_DIR = Path(r"C:/Coding/Python/NAP Project 2/race-sim-v1/Raw Data")   # <-- CHANGE THIS

# Path to v2 project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # race-sim-v2/
NEW_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Should files be MOVED or COPIED?
MOVE_FILES = False   # True = move, False = copy

# ==================================================


def slugify(name: str) -> str:
    """Convert session name into safe filename component."""
    name = name.replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "", name)


def is_parquet_valid(path: Path) -> bool:
    """Check whether a parquet file can be read correctly."""
    try:
        pd.read_parquet(path)
        return True
    except Exception:
        return False


def parse_old_filename(name: str):
    """
    Extract (year, round, session_name) from filenames like:
       2023_1_Practice_1.parquet
       2024_17_Qualifying.parquet
       2025_03_FP2.parquet
       2023_5_Sprint_Shootout.parquet

    Returns tuple (year, round, session_name) or None if it cannot parse.
    """

    base = name.replace(".parquet", "")
    parts = base.split("_")

    if len(parts) < 3:
        return None  # Not enough info

    try:
        year = int(parts[0])
        rnd = int(parts[1])
    except ValueError:
        return None

    # session name may itself contain underscores
    session_name = "_".join(parts[2:])
    return year, rnd, session_name


def migrate_raw_data():
    if not OLD_DATA_DIR.exists():
        raise FileNotFoundError(f"Old data directory does not exist: {OLD_DATA_DIR}")

    NEW_RAW_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(OLD_DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in old folder.\n")

    failures = []

    for old_path in parquet_files:
        fname = old_path.name
        parsed = parse_old_filename(fname)

        if parsed is None:
            print(f"[SKIP] Could not parse filename: {fname}")
            failures.append(fname)
            continue

        year, rnd, session_name = parsed
        safe_name = slugify(session_name)

        # New target directory and filename
        year_dir = NEW_RAW_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        new_path = year_dir / f"{year}_{rnd:02d}_{safe_name}.parquet"

        # If destination exists and is valid, skip
        if new_path.exists() and is_parquet_valid(new_path):
            print(f"[SKIP] Already exists & valid: {new_path.name}")
            continue

        print(f"[COPY] {fname} -> {new_path.relative_to(PROJECT_ROOT)}")

        try:
            if MOVE_FILES:
                shutil.move(str(old_path), str(new_path))
            else:
                shutil.copy2(str(old_path), str(new_path))
        except Exception as e:
            print(f"    ERROR copying {fname}: {e}")
            failures.append(fname)

    print("\n========== SUMMARY ==========")
    print(f"Total files: {len(parquet_files)}")
    print(f"Failures:    {len(failures)}")

    if failures:
        print("\nUnparsed or failed files:")
        for f in failures:
            print("  -", f)


if __name__ == "__main__":
    migrate_raw_data()
