"""
data_transformation.py
-----------------------
Transforms raw extracted data into monthly time-series DataFrames.

from __future__ import annotations

import time
import warnings

import pandas as pd

from logger_config import get_logger

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Census Transform
# ---------------------------------------------------------------------------

class CensusTransformation:
    """
    Cleans, pivots, and maps Census extracted data to final column names.
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("Census_Transform")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ind_name(df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the ``IndName`` composite key from component columns.

        The ``seasonally_adj`` column is mapped from ``'yes'/'no'``
        to ``'adjusted'/'notAdjusted'`` before concatenation.
        """
        df = df.copy()
        df["seasonally_adj"] = df["seasonally_adj"].map(
            {"yes": "adjusted", "no": "notAdjusted"}
        )
        df["IndName"] = (
            df["report"].str.strip()
            + "-" + df["category_code"].str.strip()
            + "-" + df["data_type_code"].str.strip()
            + "-" + df["geo_level_code"].str.strip()
            + "-" + df["seasonally_adj"].fillna("unknown")
        )
        return df

    @staticmethod
    def _pivot(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot from long (time × IndName × cell_value) to wide."""
        return df.pivot_table(
            index="time",
            columns="IndName",
            values="cell_value",
            aggfunc="first",   # each (time, IndName) should be unique
        )

    def _filter_and_rename(
        self, df_pivot: pd.DataFrame, dict_path: str
    ) -> pd.DataFrame:
        """
        Keep only series listed in the Census dictionary and rename
        columns to their human-readable labels.
        """
        dictionary = pd.read_excel(dict_path, sheet_name="Census")
        selected = dictionary["IndName"].tolist()

        available = df_pivot.columns.intersection(selected).tolist()
        missing = sorted(set(selected) - set(available))

        if missing:
            self.logger.warning(
                "Skipping %d missing Census series: %s",
                len(missing), missing,
            )

        df_out = df_pivot[available].copy()
        rename_map = dictionary.set_index("IndName")["Label"].to_dict()
        df_out = df_out.rename(columns=rename_map)
        return df_out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transform(
        self, df: pd.DataFrame, dict_path: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Full Census transformation: build IndName → pivot → filter/rename.

        Parameters
        ----------
        df : pd.DataFrame
            Raw extraction output from CensusExtract.run().
        dict_path : str
            Path to the Census data dictionary Excel file.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            (final wide DataFrame, stats dict)
        """
        t0 = time.perf_counter()

        df = self._build_ind_name(df)
        pre_pivot_shape = df.shape

        df_pivot = self._pivot(df)
        post_pivot_shape = df_pivot.shape

        df_final = self._filter_and_rename(df_pivot, dict_path)
        final_shape = df_final.shape

        elapsed = time.perf_counter() - t0

        stats = {
            "pre_pivot_shape": pre_pivot_shape,
            "post_pivot_shape": post_pivot_shape,
            "available_series": final_shape[1],
            "missing_series": post_pivot_shape[1] - final_shape[1],
            "final_shape": final_shape,
            "elapsed_seconds": round(elapsed, 3),
        }

        self.logger.info(
            "Census transform done in %.3fs — %s final shape", elapsed, final_shape
        )
        return df_final, stats


# ---------------------------------------------------------------------------
# FRED Transform
# ---------------------------------------------------------------------------

class FREDTransformation:
    """
    Converts FRED observations to monthly frequency:

    * Monthly series    → simple pivot (already monthly)
    * Daily/Weekly      → pivot then resample to month-end with
                          last / median / mean aggregations
    * Quarterly/Annual  → pivot then resample to month-start with
                          linear and pad interpolation
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger or get_logger("FRED_Transform")
        self.logger.info("FREDTransformation initialised.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Pivot a long FRED DataFrame to wide (Date × SeriesID) format.

        Returns an empty DataFrame with matching stats if input is empty.
        """
        if df.empty:
            return df.copy(), {"input_rows": 0, "pivot_shape": (0, 0)}

        pivot = df.pivot_table(
            index="Date",
            columns="SeriesID",
            values="Value",
            aggfunc="first",
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.apply(pd.to_numeric, errors="coerce")
        stats = {"input_rows": len(df), "pivot_shape": pivot.shape}
        return pivot, stats

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pivot_monthly(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Pivot monthly-frequency FRED data (no resampling needed).

        Returns
        -------
        tuple[pd.DataFrame, dict]
        """
        pivot, stats = self._pivot(df)
        stats["method"] = "pivot_only"
        self.logger.info("Monthly pivot shape: %s", stats["pivot_shape"])
        return pivot, stats

    def aggregate_daily_weekly(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Resample daily/weekly FRED observations to calendar-month end.

        Three aggregation columns are produced per series:
        ``<series>_last``, ``<series>_median``, ``<series>_mean``.

        Returns
        -------
        tuple[pd.DataFrame, dict]
        """
        pivot, stats = self._pivot(df)
        if pivot.empty:
            stats.update({"monthly_shape": (0, 0), "aggregations": []})
            return pivot, stats

        methods = ["last", "median", "mean"]
        # Resample to month-end, compute all three aggregations
        monthly = pivot.resample("ME").agg(methods)
        # Flatten MultiIndex columns → "SERIES_last", "SERIES_median" …
        monthly.columns = [f"{col}_{func}" for col, func in monthly.columns]
        # Align index to month-start (consistent with quarterly data)
        monthly.index = monthly.index + pd.offsets.MonthBegin(-1)

        stats.update({"monthly_shape": monthly.shape, "aggregations": methods})
        self.logger.info(
            "Daily/weekly aggregation complete — shape: %s", monthly.shape
        )
        return monthly, stats

    def interpolate_quarterly_annual(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Interpolate quarterly/annual FRED data to monthly frequency.

        Two interpolation methods are produced per series:
        ``<series>_LinearInterpolate`` and ``<series>_PadInterpolate``.

        Returns
        -------
        tuple[pd.DataFrame, dict]
        """
        pivot, stats = self._pivot(df)
        if pivot.empty:
            stats.update({"monthly_shape": (0, 0), "interpolation": []})
            return pivot, stats

        # Linear interpolation
        linear = pivot.resample("MS").interpolate("linear")
        linear = linear.rename(
            columns={c: f"{c}_LinearInterpolate" for c in pivot.columns}
        )

        # Forward-fill (pad) interpolation
        padded = pivot.resample("MS").interpolate("pad")
        padded = padded.rename(
            columns={c: f"{c}_PadInterpolate" for c in pivot.columns}
        )

        combined = linear.join(padded, how="outer")
        stats.update({
            "monthly_shape": combined.shape,
            "interpolation": ["linear", "pad"],
        })
        self.logger.info(
            "Quarterly/annual interpolation complete — shape: %s", combined.shape
        )
        return combined, stats
