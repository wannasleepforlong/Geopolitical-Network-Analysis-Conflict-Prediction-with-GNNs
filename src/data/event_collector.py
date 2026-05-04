"""
GDELT Event Collector -- fixed GDELT 2.0 schema
==================================================
Identified correct country-code columns via data exploration.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from .fips_filter import COUNTRY_WHITELIST, normalize_actor_code


class GDELTEventCollector:
    """Download GDELT 2.0 export CSVs and filter to major powers."""

    GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

    # CORRECT GDELT 2.0 event export schema (tab-separated, no header)
    # Verified by inspecting raw 2023-06-15 file from data.gdeltproject.org
    GDELT_SCHEMA = {
        "EventID":        0,
        "EventDate":      1,
        "Actor1Code":     5,
        "Actor1CountryCode": 7,
        "Actor2Code":     15,
        "Actor2CountryCode": 17,
        "EventCode":      26,
        "EventRootCode":  28,
        "QuadClass":      29,
        "GoldsteinScale": 30,
        "NumArticles":    33,
        "AvgTone":        34,
    }

    _COLS = list(GDELT_SCHEMA.values())
    _NAMES = list(GDELT_SCHEMA.keys())

    def __init__(self, cache_dir: str = "./gdelt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────

    def fetch_events(
        self,
        start_date: str,
        end_date: str,
        countries: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / f"events_{start_date}_{end_date}_v3.pkl"
        if use_cache and cache_path.exists():
            print(f"[GDELT] Loading cached events from {cache_path}")
            return pd.read_pickle(cache_path)

        urls = self._get_gdelt_file_list(start_date, end_date, files_per_day=1)
        if not urls:
            return pd.DataFrame(columns=self._NAMES)

        frames: List[pd.DataFrame] = []
        for url in tqdm(urls, desc="Downloading GDELT"):
            df = self._download_single(url)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=self._NAMES)

        events = pd.concat(frames, ignore_index=True)

        # Normalise actor codes: if raw code isn't a country, fallback to geographic country code
        a1_norm = events["Actor1Code"].apply(normalize_actor_code)
        a1_fb   = events["Actor1CountryCode"].apply(normalize_actor_code)
        events["Actor1Code"] = a1_norm.combine_first(a1_fb)

        a2_norm = events["Actor2Code"].apply(normalize_actor_code)
        a2_fb   = events["Actor2CountryCode"].apply(normalize_actor_code)
        events["Actor2Code"] = a2_norm.combine_first(a2_fb)

        # Drop unknown / self-loops
        events = events[
            events["Actor1Code"].notna()
            & events["Actor2Code"].notna()
            & (events["Actor1Code"] != events["Actor2Code"])
        ].copy()

        whitelist = frozenset(c.upper() for c in countries) if countries else COUNTRY_WHITELIST
        events = events[
            events["Actor1Code"].isin(whitelist)
            & events["Actor2Code"].isin(whitelist)
        ].copy()

        # Convert numerics
        for col in ["GoldsteinScale", "AvgTone", "QuadClass"]:
            events[col] = pd.to_numeric(events[col], errors="coerce")
        events["EventDate"] = pd.to_datetime(
            events["EventDate"], format="%Y%m%d", errors="coerce"
        )

        # Drop helper cols
        events = events.drop(columns=["Actor1CountryCode", "Actor2CountryCode"], errors="ignore")

        if use_cache:
            events.to_pickle(cache_path)
            print(f"[GDELT] Cached {len(events)} events to {cache_path}")

        return events.reset_index(drop=True)

    # ── internals ────────────────────────────────────────────────────────────

    def _get_gdelt_file_list(
        self, start_date: str, end_date: str, files_per_day: int = 1
    ) -> List[str]:
        resp = requests.get(self.GDELT_MASTER_URL, timeout=60)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        daily_urls: Dict[str, List[str]] = {}
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            url = parts[2]
            if ".export.CSV.zip" not in url:
                continue
            try:
                fname = url.split("/")[-1]
                d = datetime.strptime(fname[:8], "%Y%m%d")
                if start <= d <= end:
                    daily_urls.setdefault(fname[:8], []).append(url)
            except ValueError:
                continue

        urls: List[str] = []
        for day_key in sorted(daily_urls):
            day_list = sorted(daily_urls[day_key])
            step = max(1, len(day_list) // files_per_day)
            urls.extend(day_list[::step][:files_per_day])

        print(f"[GDELT] Found {len(urls)} URLs (sampled {files_per_day}/day).")
        return urls

    def _download_single(self, url: str) -> Optional[pd.DataFrame]:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        sep="\t",
                        header=None,
                        usecols=self._COLS,
                        names=self._NAMES,
                        dtype={
                            "EventDate": str,
                            "Actor1Code": str,
                            "Actor2Code": str,
                            "Actor1CountryCode": str,
                            "Actor2CountryCode": str,
                        },
                        low_memory=False,
                    )
            return df
        except Exception as exc:
            print(f"[GDELT] Failed {url}: {exc}")
            return None
