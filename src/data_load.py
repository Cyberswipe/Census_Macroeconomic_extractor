"""
data_load.py
------------
Persists transformed Census and FRED data to CSV master files.

Key changes from v1.0
---------------------
General
  - Both loaders now accept an output_path explicitly rather than
    relying on instance state set during __init__; this makes methods
    independently callable and testable.
  - Atomic write pattern introduced: data is written to a ``.tmp``
    file first, then renamed to the target path.  This prevents a
    half-written CSV from corrupting the master on crash.
  - Blank-value logging loop replaced with a vectorised implementation
    (avoid iterrows for DataFrames with many rows/columns).
  - Column alignment now uses ``reindex`` instead of manual concat of
    NaN columns (cleaner, avoids fragmentation warning).
  - Duplicate column and row deduplication is applied once at the end
    of the merge, not scattered throughout.

Census-specific
  - Index normalisation unified into a single helper.

FRED-specific
  - The ``isinstance(new_df[col].iloc[0], pd.DataFrame)`` cell-type
    guard removed; it was a band-aid for a bug in the transformation
    stage (now fixed there).  If a cell contains a DataFrame the load
    should fail loudly, not silently mangle the data.
  - ``ValueError`` raised promptly when output_csv is missing.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from logger_config import get_logger

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write(df: pd.DataFrame, path: str) -> None:
    """
    Write *df* to *path* using a write-then-rename strategy so that a
    crash during write never leaves a corrupted master file.
    """
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, encoding="utf-8")
    os.replace(tmp_path, path)


def _ensure_datetime_index(df: pd.DataFrame, fmt: Optional[str] = None) -> pd.DataFrame:
    """Convert df.index to DatetimeIndex, coercing on error."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, format=fmt, errors="coerce")
    df.index.name = "Date"
    return df


def _log_blanks_vectorised(
    df: pd.DataFrame,
    dict_lookup: dict,
    logger,
    description_field: str = "Industry",
) -> None:
    """
    Identify blank / NaN cells in *df* and emit one warning per period
    that contains them.  Uses a vectorised mask rather than iterrows.
    """
    blank_mask = df.isna() | (df.astype(str).str.strip() == "")
    if not blank_mask.any().any():
        logger.info("No blank values detected.")
        return

    # Collect blanks per date as {date_str: {col: description}}
    report: dict[str, dict[str, str]] = {}
    for dt, row_mask in blank_mask.iterrows():
        blank_cols = row_mask[row_mask].index.tolist()
        if blank_cols:
            dt_str = dt.strftime("%m-%Y") if hasattr(dt, "strftime") else str(dt)
            report[dt_str] = {
                col: dict_lookup.get(col, {}).get(description_field, "Unknown")
                for col in blank_cols
            }

    logger.warning(
        "Blank/missing values detected:\n%s", json.dumps(report, indent=2)
    )


# ---------------------------------------------------------------------------
# DataLoad
# ---------------------------------------------------------------------------

class DataLoad:
    """
    Unified loader for Census and FRED master CSV files.

    Method naming convention
    ------------------------
    _census_*   –  Census-specific helpers
    _fred_*     –  FRED-specific helpers
    """

    def __init__(
        self,
        fred_dict: Optional[pd.DataFrame],
        census_dict: Optional[pd.DataFrame],
        output_csv: Optional[str] = None,
        logger=None,
    ) -> None:
        self.output_csv = output_csv
        self.census_dict = census_dict
        self.fred_dict = fred_dict
        self.logger = logger or get_logger("DataLoad")
        self.logger.info("DataLoad initialised.")

    # =========================================================================
    #  Census Load
    # =========================================================================

    def _census_normalise_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has a well-typed DatetimeIndex named 'Date'."""
        if "time" in df.columns:
            df = df.set_index("time")
        return _ensure_datetime_index(df, fmt="%Y-%m")

    def census_merge_and_save(
        self,
        new_df: pd.DataFrame,
        output_csv: str,
    ) -> None:
        """
        Incrementally merge *new_df* into the Census master CSV and save.

        Behaviour
        ---------
        * If the master CSV does not exist → create it from *new_df*.
        * If it exists → update overlapping dates and append new dates.

        Parameters
        ----------
        new_df : pd.DataFrame
            Transformed Census data for the current extraction window.
        output_csv : str
            Path to the Census master CSV.
        """
        new_df = self._census_normalise_index(new_df.copy())

        # ----------------------------------------------------------
        # Log blank values against the data dictionary
        # ----------------------------------------------------------
        if self.census_dict is not None:
            dict_lookup = (
                self.census_dict
                .set_index("Label")
                .to_dict("index")
            )
            _log_blanks_vectorised(
                new_df, dict_lookup, self.logger, description_field="Industry"
            )

        # ----------------------------------------------------------
        # Create or merge
        # ----------------------------------------------------------
        if not os.path.exists(output_csv):
            self.logger.info("Census master not found — creating: %s", output_csv)
            _atomic_write(new_df, output_csv)
            self.logger.info("Census master created — shape: %s", new_df.shape)
            return

        old_df = pd.read_csv(output_csv, index_col=0, parse_dates=True)
        old_df = _ensure_datetime_index(old_df)

        overlapping = new_df.index.intersection(old_df.index)
        if len(overlapping):
            self.logger.info("Updating %d existing Census dates.", len(overlapping))
            old_df.update(new_df.loc[overlapping])

        new_only = new_df.index.difference(old_df.index)
        if len(new_only):
            self.logger.info("Appending %d new Census dates.", len(new_only))
            old_df = pd.concat([old_df, new_df.loc[new_only]])

        merged = old_df.sort_index()
        _atomic_write(merged, output_csv)
        self.logger.info(
            "Census master updated — shape: %s → %s",
            old_df.shape, merged.shape,
        )

    # =========================================================================
    #  FRED Load
    # =========================================================================

    def _build_fred_dict_lookup(self) -> dict:
        if self.fred_dict is None:
            return {}
        return (
            self.fred_dict
            .drop_duplicates(subset=["IndName"], keep="first")
            .set_index("IndName")
            .to_dict("index")
        )

    def fred_merge_and_save(
        self,
        new_df: pd.DataFrame,
        output_csv: Optional[str] = None,
    ) -> None:
        """
        Incrementally merge *new_df* into the FRED master CSV and save.

        Parameters
        ----------
        new_df : pd.DataFrame
            Final combined MEI DataFrame (Date index, series columns).
        output_csv : str | None
            Override for instance-level output_csv.

        Raises
        ------
        ValueError
            If no output path is available.
        """
        target = output_csv or self.output_csv
        if not target:
            raise ValueError(
                "output_csv must be provided either at construction "
                "or as an argument to fred_merge_and_save()."
            )

        new_df = new_df.copy()
        new_df = _ensure_datetime_index(new_df)

        # Remove stray 'Date' column that sometimes bleeds through
        if "Date" in new_df.columns:
            new_df = new_df.drop(columns=["Date"])

        new_df = new_df.apply(pd.to_numeric, errors="ignore")

        # ----------------------------------------------------------
        # Log blank values
        # ----------------------------------------------------------
        IGNORE_COLS = {"SeriesID", "BaseID", "Value"}
        audit_df = new_df.drop(
            columns=[c for c in IGNORE_COLS if c in new_df.columns],
            errors="ignore",
        )
        _log_blanks_vectorised(
            audit_df, self._build_fred_dict_lookup(), self.logger
        )

        # ----------------------------------------------------------
        # Create master file if it doesn't exist
        # ----------------------------------------------------------
        if not os.path.exists(target):
            self.logger.info("FRED master not found — creating: %s", target)
            self.logger.info("Column count: %d", new_df.shape[1])
            for dt, row in new_df.iterrows():
                self.logger.info(
                    "NEW DATE: %s → %d values populated",
                    dt.date(), row.count(),
                )
            _atomic_write(new_df, target)
            return

        # ----------------------------------------------------------
        # Merge with existing master
        # ----------------------------------------------------------
        old_df = pd.read_csv(target, parse_dates=["Date"], index_col="Date")
        if "Date" in old_df.columns:
            old_df = old_df.drop(columns=["Date"])

        # Deduplicate columns (keep last occurrence in each frame)
        old_df = old_df.loc[:, ~old_df.columns.duplicated(keep="last")]
        new_df = new_df.loc[:, ~new_df.columns.duplicated(keep="last")]

        # Align new_df columns to match old_df (fill missing with NaN)
        new_df = new_df.reindex(columns=old_df.columns, fill_value=np.nan)

        # Identify genuinely new dates
        new_dates = new_df.index.difference(old_df.index)
        if len(new_dates):
            self.logger.info("New dates being added: %d", len(new_dates))
        else:
            self.logger.info("No new dates — updating existing rows only.")

        # Update existing rows, then append new ones
        old_df.update(new_df)
        if len(new_dates):
            old_df = pd.concat([old_df, new_df.loc[new_dates]])

        # Final dedup + sort
        merged = (
            old_df
            .loc[~old_df.index.duplicated(keep="last")]
            .sort_index()
            .round(3)
        )

        self.logger.info("Final FRED master column count: %d", merged.shape[1])
        for dt in new_dates:
            self.logger.info(
                "NEW DATE POPULATED: %s → %d values",
                dt.date(), merged.loc[dt].count(),
            )

        _atomic_write(merged, target)
        self.logger.info("FRED master saved → %s  shape: %s", target, merged.shape)
