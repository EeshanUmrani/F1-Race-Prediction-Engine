"""
Build a JSON index of all F1 sessions (race weekends only) for a given year range.
This is the v2 version adapted to the new project layout.

Output:
    data/sessions_index.json
Structure:
[
  {
    "year": 2023,
    "grand_prix": "Bahrain Grand Prix",
    "round": 1,
    "session_number": 1,
    "session_name": "Practice 1"
  },
  ...
]
"""

import json
import datetime as dt
from pathlib import Path

import fastf1
import pandas as pd


# ================== CONFIGURATION ==================

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # race-sim-v2/
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_PATH = DATA_DIR / "sessions_index.json"

START_YEAR = 2022
END_YEAR = 2025

# ====================================================


def build_session_list(
    start_year: int,
    end_year: int,
    cache_dir: Path,
    output_path: Path,
) -> None:

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    now_utc = dt.datetime.now(dt.timezone.utc)
    all_sessions: list[dict] = []

    for year in range(start_year, end_year + 1):
        print(f"Processing season {year}...")

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if schedule is None or len(schedule) == 0:
            print(f"  Warning: empty schedule for {year}, skipping.")
            continue

        for _, event in schedule.iterrows():
            if "F1ApiSupport" in schedule.columns and not bool(event.get("F1ApiSupport", True)):
                continue

            event_name = str(event["EventName"])
            round_number = int(event["RoundNumber"])

            for sess_num in range(1, 5 + 1):  # Session1–Session5
                sess_col = f"Session{sess_num}"
                date_col = f"Session{sess_num}DateUtc"

                if sess_col not in schedule.columns or date_col not in schedule.columns:
                    continue

                sess_name = event.get(sess_col)
                sess_date = event.get(date_col)

                if pd.isna(sess_name) or pd.isna(sess_date):
                    continue

                if sess_date.tzinfo is None:
                    sess_dt_utc = sess_date.replace(tzinfo=dt.timezone.utc)
                else:
                    sess_dt_utc = sess_date.astimezone(dt.timezone.utc)

                # Skip sessions in the future
                if sess_dt_utc > now_utc:
                    continue

                all_sessions.append(
                    {
                        "year": year,
                        "grand_prix": event_name,
                        "round": round_number,
                        "session_number": sess_num,
                        "session_name": str(sess_name),
                    }
                )

    all_sessions.sort(key=lambda x: (x["year"], x["round"], x["session_number"]))

    output_path.write_text(json.dumps(all_sessions, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {len(all_sessions)} sessions to {output_path.resolve()}")


if __name__ == "__main__":
    build_session_list(
        start_year=START_YEAR,
        end_year=END_YEAR,
        cache_dir=CACHE_DIR,
        output_path=OUTPUT_PATH,
    )
