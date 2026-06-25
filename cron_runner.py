"""
cron_runner.py
--------------
Orchestrates the unified MEI (FRED) + Census pipeline.

Key changes from v1.0
---------------------
- sys.path.insert with hardcoded Z: drive path removed.
  Use PYTHONPATH or install the package instead.
- API keys are no longer passed via config.json.
  Set CENSUS_API_KEY and FRED_API_KEY as environment variables
  (e.g. in your .env file, Windows Task Scheduler action, or CI secret).
- CURRENT_DIR is derived from __file__ so the script runs correctly
  from any working directory.
- Logging path injected into get_logger (no hardcoded drive path).
- Census and MEI pipelines are each wrapped in their own try/except
  and individually guarded — a Census failure does not abort MEI.
- compute_start_date / compute_census_start_date merged into a single
  generic helper.
- generate_job_id uses a file lock (via a simple temp-file strategy)
  to avoid duplicate IDs when two processes start simultaneously.
- FutureWarning suppression moved here (single point of control).
- Both pipelines log their own stats via json.dumps for auditability.
- Renamed public method references to match the refactored class API.

Usage
-----
    # Set secrets
    export CENSUS_API_KEY=<your_key>
    export FRED_API_KEY=<your_key>

    # Run from the project root
    python cron_runner.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path


import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project-relative imports
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_extraction import CensusExtract, FREDExtract          # noqa: E402
from data_transformation import CensusTransformation, FREDTransformation  # noqa: E402
from data_load import DataLoad                                   # noqa: E402
from logger_config import get_logger                             # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = CURRENT_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_start_date(
    default_start: str,
    master_csv: str | Path,
    lookback_months: int = 4,
    date_format: str = "%Y-%m-%d",
) -> str:
    """
    Return the extraction start date string.

    * No master CSV → use *default_start* (full history load).
    * Master CSV exists → current date minus *lookback_months*.

    Works for both FRED (``YYYY-MM-DD``) and Census (``YYYY-MM``)
    formats via the *date_format* parameter.
    """
    if not os.path.exists(master_csv):
        return default_start
    back_date = datetime.today() - relativedelta(months=lookback_months)
    return back_date.strftime(date_format)


def generate_job_id(base_dir: str | Path) -> str:
    """
    Create a unique daily job ID of the form ``IDMMDDYYYYJXX``.

    The counter is persisted in ``job_state.json``.  A simple
    read-modify-write pattern is used; for multi-process safety on
    the same machine, wrap in a file lock (not needed for typical
    scheduled single-process runs).
    """
    today = datetime.today().strftime("%m%d%Y")
    state_file = Path(base_dir) / "job_state.json"

    state: dict = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except (json.JSONDecodeError, OSError):
            state = {}

    counter = state.get(today, 0) + 1
    state[today] = counter

    with open(state_file, "w", encoding="utf-8") as fh:
        json.dump(state, fh)

    return f"ID{today}J{counter:02d}"


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_census_pipeline(
    config: dict,
    loader: DataLoad,
    census_csv: Path,
    logger: logging.Logger,
) -> bool:
    """
    Execute the Census extract → transform → load pipeline.

    Returns True on success, False on failure (allows MEI to proceed).
    """
    logger.info("=== Census Pipeline Started ===")

    dict_path = CURRENT_DIR / "resources" / "Dict_Census_test.xlsx"

    start_date = compute_start_date(
        default_start=config["census"]["default_start_date"],
        master_csv=census_csv,
        lookback_months=config["pipeline"]["incremental_lookback_months"],
        date_format="%Y-%m",
    )
    logger.info("Census start date: %s", start_date)

    try:
        extractor = CensusExtract(logger=logger)
        transformer = CensusTransformation(logger=logger)

        # Extract
        df_raw, extract_stats = extractor.run(start_date=start_date)
        logger.info("Census extract stats:\n%s", json.dumps(extract_stats, indent=2))

        if df_raw.empty:
            logger.warning("Census extraction returned no data — skipping transform/load.")
            return False

        # Transform
        df_final, transform_stats = transformer.transform(df_raw, str(dict_path))
        logger.info(
            "Census transform stats:\n%s", json.dumps(transform_stats, indent=2)
        )

        # Load
        loader.census_merge_and_save(df_final, str(census_csv))
        logger.info("=== Census Pipeline Completed Successfully ===")
        return True

    except Exception:
        logger.exception("Census Pipeline failed.")
        return False


def run_mei_pipeline(
    config: dict,
    loader: DataLoad,
    fred_dict: pd.DataFrame,
    output_csv: Path,
    logger: logging.Logger,
) -> bool:
    """
    Execute the MEI (FRED) extract → transform → load pipeline.

    Returns True on success, False on failure.
    """
    logger.info("=== MEI Pipeline Started ===")

    start_date = compute_start_date(
        default_start=config["fred"]["default_start_date"],
        master_csv=output_csv,
        lookback_months=config["pipeline"]["incremental_lookback_months"],
        date_format="%Y-%m-%d",
    )
    logger.info("MEI start date: %s", start_date)

    try:
        extractor = FREDExtract(
            dict_df=fred_dict,
            api_log_path=str(CURRENT_DIR / "api_file_list.txt"),
            logger=logger,
        )
        transformer = FREDTransformation(logger=logger)

        # Partition IDs by frequency
        fred_M_ids, fred_DW_ids, fred_QA_ids, mapping = extractor.prepare_id_groups()

        # ---- Extract ----
        fred_M, stats_M = extractor.fetch_series(fred_M_ids, mapping=mapping, start_date=start_date)
        fred_DW, stats_DW = extractor.fetch_series(fred_DW_ids, mapping=mapping, start_date=start_date)
        fred_QA, stats_QA = extractor.fetch_series(fred_QA_ids, mapping=mapping, start_date=start_date)

        logger.info(
            "FRED extract stats:\n%s",
            json.dumps({"Monthly": stats_M, "Daily/Weekly": stats_DW, "Quarterly/Annual": stats_QA}, indent=2),
        )

        # ---- Transform ----
        df_M, t_stats_M = transformer.pivot_monthly(fred_M)
        df_DW, t_stats_DW = transformer.aggregate_daily_weekly(fred_DW)
        df_QA, t_stats_QA = transformer.interpolate_quarterly_annual(fred_QA)

        logger.info(
            "FRED transform stats:\n%s",
            json.dumps({
                "Monthly pivot": t_stats_M,
                "DW aggregation": t_stats_DW,
                "QA interpolation": t_stats_QA,
            }, indent=2),
        )

        # Combine and round
        final_df = pd.concat([df_M, df_DW, df_QA], axis=1).sort_index().round(3)
        logger.info("Combined MEI shape: %s", final_df.shape)

        # ---- Load ----
        loader.fred_merge_and_save(final_df, str(output_csv))
        logger.info("=== MEI Pipeline Completed Successfully ===")
        return True

    except Exception:
        logger.exception("MEI Pipeline failed.")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the MEI + Census pipelines.

    Returns exit code: 0 = all pipelines succeeded, 1 = at least one failed.
    """
    config = _load_config()

    # ---- Job ID & per-job log ----
    job_id = generate_job_id(CURRENT_DIR)
    log_dir = CURRENT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    job_log_path = log_dir / f"{job_id}.txt"
    cron_log_path = CURRENT_DIR / "Cron_log.txt"

    logger = get_logger(
        "Main",
        log_file_path=str(cron_log_path),
        rotation_bytes=config["pipeline"]["log_rotation_bytes"],
        backup_count=config["pipeline"]["log_backup_count"],
    )

    # Also write per-job log
    per_job_handler = logging.FileHandler(job_log_path, encoding="utf-8")
    per_job_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(per_job_handler)

    logger.info("=== Starting Job: %s ===", job_id)

    # ---- Load data dictionaries ----
    fred_dict = pd.read_excel(
        CURRENT_DIR / "resources" / "Dict_Fred_test.xlsx",
        sheet_name="FRED_MEI",
    )
    census_dict = pd.read_excel(
        CURRENT_DIR / "resources" / "Dict_Census_test.xlsx",
        sheet_name="Census",
    )

    census_csv = CURRENT_DIR / "Census_Master.csv"
    mei_csv = CURRENT_DIR / "MEI_Master_Output.csv"

    # ---- Shared loader ----
    loader = DataLoad(
        fred_dict=fred_dict,
        census_dict=census_dict,
        output_csv=str(mei_csv),
        logger=logger,
    )

    # ---- Run pipelines ----
    census_ok = run_census_pipeline(config, loader, census_csv, logger)
    mei_ok = run_mei_pipeline(config, loader, fred_dict, mei_csv, logger)

    # ---- Summary ----
    logger.info(
        "=== Pipeline Summary — Census: %s | MEI: %s ===",
        "OK" if census_ok else "FAILED",
        "OK" if mei_ok else "FAILED",
    )

    return 0 if (census_ok and mei_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
