"""
Download raw data for each F1 session listed in data/sessions_index.json
and save one parquet file per session inside:

    data/raw/<year>/<year>_<round>_<session>.parquet

This is the v2 version adapted to the new modular project layout.
"""

import json
import re
import time
from pathlib import Path

import fastf1
import pandas as pd


# =============== CONFIGURATION ===============

PROJECT_ROOT = Path("C:/Coding/Python/NAP Project 2/race-sim-v2")           # race-sim-v2/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
SESSIONS_INDEX_PATH = DATA_DIR / "sessions_index.json"

MAX_RETRIES = 3
BASE_RETRY_DELAY = 60        # seconds
RATE_LIMIT_COOLDOWN = 120    # seconds
DELAY_BETWEEN_SESSIONS = 10  # seconds

# =============================================


def slugify(name: str) -> str:
    name = name.replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "", name)


def is_valid_parquet(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _ = pd.read_parquet(path)
        return True
    except Exception as e:
        print(f"    Corrupt parquet {path.name}: {e}")
        return False


def looks_like_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    keywords = [
        "Failed to load any schedule data",
        "Failed to load schedule from FastF1 backend",
        "Failed to load schedule from F1 API backend",
        "Failed to load schedule from Ergast API backend",
        "429",
    ]
    return any(k in msg for k in keywords)


def download_all_sessions(
    index_path: Path,
    cache_dir: Path,
    raw_dir: Path,
    max_retries: int = 3,
    base_retry_delay: int = 60,
    delay_between_sessions: int = 10,
    rate_limit_cooldown: int = 120,
) -> None:

    if not index_path.exists():
        raise FileNotFoundError(f"Session index JSON not found: {index_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    sessions = json.loads(index_path.read_text())
    print(f"Loaded {len(sessions)} sessions from {index_path}\n")

    failed = []

    for i, sess in enumerate(sessions, start=1):
        year = sess["year"]
        gp = sess["grand_prix"]
        rnd = sess["round"]
        sess_name = sess["session_name"]

        safe_name = slugify(sess_name)
        year_dir = raw_dir / str(year)
        year_dir.mkdir(exist_ok=True, parents=True)

        outfile = year_dir / f"{year}_{rnd:02d}_{safe_name}.parquet"

        # Resume logic
        if is_valid_parquet(outfile):
            print(f"[{i}/{len(sessions)}] Skipping existing: {outfile.name}")
            continue
        elif outfile.exists():
            print(f"[{i}/{len(sessions)}] Removing corrupt: {outfile.name}")
            outfile.unlink()

        print(f"[{i}/{len(sessions)}] Downloading {year} R{rnd} - {gp} - {sess_name}")

        success = False

        for attempt in range(1, max_retries + 1):
            try:
                session = fastf1.get_session(year, gp, sess_name)
                session.load()

                laps = session.laps.copy()
                laps["Year"] = year
                laps["RoundNumber"] = rnd
                laps["EventName"] = gp
                laps["SessionName"] = sess_name

                laps.to_parquet(outfile, index=False)
                print(f"    Saved to {outfile}")
                success = True
                break

            except Exception as e:
                print(
                    f"    ERROR attempt {attempt}/{max_retries} for "
                    f"{year} R{rnd} {gp} {sess_name}: {e}"
                )

                if looks_like_rate_limit_error(e):
                    print(f"    Rate limit detected; cooling down {rate_limit_cooldown}s")
                    time.sleep(rate_limit_cooldown)
                    fastf1.Cache.enable_cache(str(cache_dir))
                else:
                    if attempt < max_retries:
                        wait = base_retry_delay * attempt
                        print(f"    Retrying in {wait} seconds...")
                        time.sleep(wait)

                if attempt == max_retries and not success:
                    print("    FAILED permanently.")
                    failed.append(sess)

        if success and delay_between_sessions > 0:
            time.sleep(delay_between_sessions)

    # Summary
    print("\n================ SUMMARY ================")
    print(f"Total sessions attempted : {len(sessions)}")
    print(f"Total failed             : {len(failed)}")

    if failed:
        print("\nFailed sessions:")
        for s in failed:
            print(f"  - {s['year']} R{s['round']} {s['grand_prix']} {s['session_name']}")


if __name__ == "__main__":
    download_all_sessions(
        index_path=SESSIONS_INDEX_PATH,
        cache_dir=CACHE_DIR,
        raw_dir=RAW_DIR,
        max_retries=MAX_RETRIES,
        base_retry_delay=BASE_RETRY_DELAY,
        delay_between_sessions=DELAY_BETWEEN_SESSIONS,
        rate_limit_cooldown=RATE_LIMIT_COOLDOWN,
    )
