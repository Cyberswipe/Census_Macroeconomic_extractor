"""
data_extraction.py
------------------
Handles all inbound API calls for the MEI + Census pipeline.

from __future__ import annotations

import json
import os
import time
import urllib.parse
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from logger_config import get_logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.json") -> dict:
    """Load pipeline configuration from JSON."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


CONFIG = _load_config()


# ---------------------------------------------------------------------------
# Census Extraction
# ---------------------------------------------------------------------------

class CensusExtract:
    """
    Extracts Census Bureau monthly EITS datasets via their public API.

    The Census API key is read from the environment variable
    ``CENSUS_API_KEY``.  An explicit ``api_key`` constructor argument
    is available for testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        logger=None,
    ) -> None:
        cfg = CONFIG["census"]

        self.api_key: str = api_key or os.environ.get("CENSUS_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "Census API key not found. "
                "Set the CENSUS_API_KEY environment variable."
            )

        self.default_start_date: str = cfg["default_start_date"]
        self.base_page: str = cfg["base_page"]
        self.quarterly_ds: set[str] = set(cfg["quarterly_datasets"])
        self.timeout: int = cfg["request_timeout_seconds"]
        self.logger = logger or get_logger("Census_Extract")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scrape_base_api_links(self) -> list[str]:
        """
        Scrape the Census EITS documentation page and return a list of
        base API URLs, each pre-appended with the common field selectors.

        Returns
        -------
        list[str]
            URLs in the form ``<base>?get=cell_value,time_slot_id&for=US&``.

        Raises
        ------
        requests.HTTPError
            If the documentation page returns a non-2xx status.
        ValueError
            If no data table is found on the page.
        """
        resp = requests.get(self.base_page, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError(
                f"No HTML table found on Census EITS page: {self.base_page}"
            )

        urls: list[str] = []
        for row in table.find_all("tr")[2:]:          # skip header rows
            cols = row.find_all("td")
            if not cols:
                continue
            api_base = cols[-1].text.strip()
            if api_base:
                urls.append(api_base + "?get=cell_value,time_slot_id&for=US&")

        return urls

    @staticmethod
    def _build_params(
        api_key: str,
        start_date: Optional[str] = None,
        default_start: str = "from2015-01",
        error_data: str = "",
        seasonally_adj: str = "",
        category_code: str = "",
        data_type_code: str = "",
        geo_level_code: str = "",
    ) -> dict:
        """Construct Census API query parameters."""
        if start_date:
            raw = str(start_date).replace(" ", "")
            # Normalise to "fromYYYY-MM" format
            if raw.lower().startswith("from"):
                time_param = raw
            else:
                time_param = "from" + raw[:7]
        else:
            time_param = default_start

        return {
            "error_data": error_data,
            "seasonally_adj": seasonally_adj,
            "category_code": category_code,
            "data_type_code": data_type_code,
            "geo_level_code": geo_level_code,
            "time": time_param,
            "key": api_key,
        }

    @staticmethod
    def _json_to_df(payload: list) -> pd.DataFrame:
        """Convert Census JSON list-of-lists to a DataFrame."""
        if not payload or len(payload) < 2:
            return pd.DataFrame()
        return pd.DataFrame(payload[1:], columns=payload[0])

    def _dataset_code_from_url(self, url: str) -> str:
        """Extract the dataset short-code from a Census API URL."""
        try:
            code = url.split("/")[6].split("?")[0]
        except IndexError:
            code = ""
        return "M3ADV" if code == "advm3" else code.upper()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, start_date: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
        """
        Extract all monthly EITS datasets for the given start date.

        Parameters
        ----------
        start_date : str | None
            ``"YYYY-MM"`` or ``"fromYYYY-MM"`` string.
            Defaults to the ``default_start_date`` in config.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            (combined raw data, extraction stats dict)
        """
        start_time = datetime.now()
        params = self._build_params(
            api_key=self.api_key,
            start_date=start_date,
            default_start=self.default_start_date,
        )

        # Build and filter URLs
        all_urls = [
            base + urllib.parse.urlencode(params)
            for base in self._scrape_base_api_links()
        ]
        monthly_urls = [
            u for u in all_urls
            if u.split("/")[6].split("?")[0] not in self.quarterly_ds
        ]

        frames: list[pd.DataFrame] = []
        extract_count = 0
        skipped_count = 0
        report_shapes: dict[str, dict] = {}

        session = requests.Session()

        for url in monthly_urls:
            dataset_code = self._dataset_code_from_url(url)
            try:
                resp = session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
            except requests.exceptions.Timeout:
                skipped_count += 1
                self.logger.warning("Timeout for dataset %s", dataset_code)
                continue
            except requests.exceptions.HTTPError as exc:
                skipped_count += 1
                self.logger.warning(
                    "HTTP error for dataset %s: %s", dataset_code, exc
                )
                continue
            except (ValueError, KeyError) as exc:
                skipped_count += 1
                self.logger.warning(
                    "Malformed JSON for dataset %s: %s", dataset_code, exc
                )
                continue

            df = self._json_to_df(payload)
            if df.empty:
                skipped_count += 1
                self.logger.warning("Empty response for dataset %s", dataset_code)
                continue

            df["report"] = dataset_code
            extract_count += 1
            report_shapes[dataset_code] = {"rows": df.shape[0], "cols": df.shape[1]}
            self.logger.info(
                "Extracted %d rows from %s", df.shape[0], dataset_code
            )
            frames.append(df)

        data_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            "total_attempted": len(monthly_urls),
            "extracted": extract_count,
            "skipped": skipped_count,
            "rows": data_all.shape[0],
            "columns": data_all.shape[1],
            "elapsed_seconds": round(elapsed, 2),
            "per_report": report_shapes,
        }

        self.logger.info(
            "Census extraction complete in %.2fs — %d/%d succeeded",
            elapsed, extract_count, len(monthly_urls),
        )
        return data_all, stats


# ---------------------------------------------------------------------------
# FRED Extraction
# ---------------------------------------------------------------------------

class FREDExtract:
    """
    Extracts macroeconomic indicator data from the FRED API
    (Federal Reserve Bank of St. Louis).

    The FRED API key is read from the environment variable
    ``FRED_API_KEY``.  An explicit ``api_key`` constructor argument
    is available for testing.
    """

    def __init__(
        self,
        dict_df: pd.DataFrame,
        api_key: Optional[str] = None,
        api_log_path: Optional[str] = None,
        logger=None,
    ) -> None:
        cfg = CONFIG["fred"]

        self.api_key: str = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "FRED API key not found. "
                "Set the FRED_API_KEY environment variable."
            )

        self.api_url: str = cfg["api_url"]
        self.api_log_path: Optional[str] = api_log_path
        self.default_start_date: str = cfg["default_start_date"]
        self.max_retries: int = cfg["max_retries"]
        self.retry_backoff: int = cfg["retry_backoff_seconds"]

        self.logger = logger or get_logger("FRED_Extract")

        # Validate and deduplicate the data dictionary
        if dict_df["IndName"].duplicated().any():
            dup_n = dict_df["IndName"].duplicated().sum()
            self.logger.warning(
                "Dictionary has %d duplicate IndName entries — keeping first.",
                dup_n,
            )
            dict_df = dict_df.drop_duplicates(subset=["IndName"], keep="first")

        self.dict_df = dict_df.copy()

        # Lookup: IndName → {Industry, Original Frequency}
        self.series_meta: dict[str, dict] = (
            self.dict_df
            .set_index("IndName")[["Industry", "Original Frequency"]]
            .to_dict("index")
        )

        self.logger.info("FREDExtract initialised with %d series.", len(self.dict_df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, series_id: str, start_date: str) -> str:
        return (
            f"{self.api_url}"
            f"?series_id={series_id}"
            f"&api_key={self.api_key}"
            f"&file_type=json"
            f"&observation_start={start_date}"
        )

    def _log_api_url(self, url: str) -> None:
        """Append the called URL to the API log file (if configured)."""
        if not self.api_log_path:
            return
        try:
            log_dir = os.path.dirname(self.api_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.api_log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n{url}")
        except OSError as exc:
            self.logger.warning("Could not write API log: %s", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare_id_groups(self) -> tuple[list, list, list, dict]:
        """
        Partition series IDs by original frequency and build
        a base-ID → [original-ID, …] mapping for DW/QA series.

        Returns
        -------
        tuple
            (fred_M_ids, fred_DW_ids, fred_QA_ids, mapping)
            where *_ids are lists of series IDs and *mapping* is
            ``{base_id: [original_id, …]}``.
        """
        mapping: dict[str, list[str]] = {}
        for ind in self.dict_df["IndName"].dropna():
            base = ind.split("_")[0]
            mapping.setdefault(base, []).append(ind)

        freq = self.dict_df.set_index("IndName")["Original Frequency"]

        fred_M_ids: list[str] = (
            self.dict_df[freq == "M"]["IndName"].tolist()
        )
        fred_DW_ids: list[str] = sorted({
            ind.split("_")[0]
            for ind in self.dict_df[freq.isin(["D", "W"])]["IndName"]
        })
        fred_QA_ids: list[str] = sorted({
            ind.split("_")[0]
            for ind in self.dict_df[freq.isin(["Q", "A"])]["IndName"]
        })

        return fred_M_ids, fred_DW_ids, fred_QA_ids, mapping

    def fetch_series(
        self,
        series_list: list[str],
        mapping: Optional[dict] = None,
        start_date: Optional[str] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Fetch observations for a list of FRED series IDs.

        Parameters
        ----------
        series_list : list[str]
            Base FRED series IDs to fetch.
        mapping : dict | None
            Maps base_id → [original_ids].  When supplied, each
            observation is replicated for every original ID.
        start_date : str | None
            ``"YYYY-MM-DD"`` extraction start.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            (observations DataFrame, stats dict)
        """
        if mapping is None:
            mapping = {}
        start_date = start_date or self.default_start_date
        self.logger.info("FRED fetch starting from %s for %d series.", start_date, len(series_list))

        rows: list[dict] = []
        success_count = retry_success_count = failed_count = 0
        failed_ids: list[str] = []

        session = requests.Session()

        for series_id in series_list:
            url = self._build_url(series_id, start_date)
            self._log_api_url(url)
            original_ids = mapping.get(series_id, [series_id])

            response = None

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = session.get(url, timeout=30)
                except requests.exceptions.RequestException as exc:
                    self.logger.warning(
                        "Network error for %s (attempt %d): %s",
                        series_id, attempt, exc,
                    )
                    time.sleep(self.retry_backoff)
                    continue

                if response.status_code == 200:
                    if attempt == 1:
                        success_count += 1
                    else:
                        retry_success_count += 1
                    break

                if response.status_code in (429, 403):
                    self.logger.warning(
                        "[RATE LIMIT] %s (attempt %d/%d) — waiting %ds",
                        series_id, attempt, self.max_retries, self.retry_backoff,
                    )
                    time.sleep(self.retry_backoff)
                    continue

                # Non-retriable HTTP error
                self.logger.error(
                    "Non-retriable HTTP %d for %s", response.status_code, series_id
                )
                failed_count += 1
                failed_ids.append(series_id)
                response = None
                break

            else:
                # Exhausted retries
                self.logger.error("FAILED %s after %d retries.", series_id, self.max_retries)
                failed_count += 1
                failed_ids.append(series_id)
                continue

            if response is None or response.status_code != 200:
                continue

            try:
                data = response.json()
            except ValueError:
                self.logger.error("Non-JSON response for %s", series_id)
                failed_count += 1
                failed_ids.append(series_id)
                continue

            observations = data.get("observations", [])
            for obs in observations:
                for orig in original_ids:
                    rows.append({
                        "SeriesID": orig,
                        "BaseID": series_id,
                        "Date": obs["date"],
                        "Value": obs["value"],
                    })

            for orig in original_ids:
                meta = self.series_meta.get(orig, {})
                self.logger.info(
                    "Fetched %s | base=%s | industry=%s | freq=%s",
                    orig, series_id,
                    meta.get("Industry", "N/A"),
                    meta.get("Original Frequency", "N/A"),
                )

        # Assemble result
        if rows:
            df = pd.DataFrame(rows)
            df.drop_duplicates(inplace=True)
            df["Value"] = pd.to_numeric(df["Value"].replace(".", None), errors="coerce")
            df["Date"] = pd.to_datetime(df["Date"])
        else:
            df = pd.DataFrame(columns=["SeriesID", "BaseID", "Date", "Value"])

        stats = {
            "success_first_try": success_count,
            "success_after_retry": retry_success_count,
            "failed": failed_count,
            "failed_ids": failed_ids,
            "series_attempted": len(series_list),
            "rows_returned": len(df),
        }

        self.logger.info(
            "FRED fetch done — %d ok, %d retried, %d failed, %d rows",
            success_count, retry_success_count, failed_count, len(df),
        )
        return df, stats
