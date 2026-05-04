"""
Geopolitical Network Builder
============================
Builds monthly temporal networks from GDELT event DataFrames.
Only processes events where both actors are known FIPS countries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from .fips_filter import COUNTRY_WHITELIST


class GeopoliticalNetworkBuilder:
    """
    Aggregate GDELT events into monthly conflict/cooperation networks.
    """

    def __init__(self) -> None:
        self.networks: Dict[str, Any] = {}
        self.country_indices: Dict[str, int] = {}
        self.reverse_indices: Dict[int, str] = {}

    def build_temporal_networks(
        self,
        events_df: pd.DataFrame,
        time_window: str = "M",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        events_df : pd.DataFrame
            Must contain columns: EventDate, Actor1Code, Actor2Code,
            QuadClass, GoldsteinScale, AvgTone.
        time_window : str
            Pandas period alias: 'M' = monthly, 'W' = weekly, 'Q' = quarterly.

        Returns
        -------
        dict[str, dict]
            Keys are period strings (e.g. '2023-01'); values are network dicts.
        """
        if events_df.empty:
            print("[NetworkBuilder] Empty DataFrame — returning empty dict.")
            return {}

        # Ensure datetime
        events_df = events_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(events_df["EventDate"]):
            events_df["EventDate"] = pd.to_datetime(
                events_df["EventDate"], format="%Y%m%d", errors="coerce"
            )
        events_df = events_df.dropna(subset=["EventDate"])
        events_df["Period"] = events_df["EventDate"].dt.to_period(time_window)

        # Country index mapping (fixed, alphabetical, only whitelist)
        self.country_indices = {
            code: i for i, code in enumerate(sorted(COUNTRY_WHITELIST))
        }
        self.reverse_indices = {i: code for code, i in self.country_indices.items()}
        num_countries = len(self.country_indices)

        for period, group in events_df.groupby("Period"):
            self.networks[str(period)] = self._build_single_network(group, num_countries)

        print(
            f"[NetworkBuilder] Built {len(self.networks)} networks "
            f"for {num_countries} countries."
        )
        return self.networks

    def _build_single_network(
        self,
        events_df: pd.DataFrame,
        num_countries: int,
    ) -> Dict[str, Any]:
        """Aggregate one time-period of events into adjacency tensors."""
        adj_conflict = np.zeros((num_countries, num_countries), dtype=np.float32)
        adj_coop = np.zeros((num_countries, num_countries), dtype=np.float32)
        gold_sum = np.zeros((num_countries, num_countries), dtype=np.float32)
        gold_cnt = np.zeros((num_countries, num_countries), dtype=np.float32)
        tone_sum = np.zeros((num_countries, num_countries), dtype=np.float32)
        tone_cnt = np.zeros((num_countries, num_countries), dtype=np.float32)
        event_cnt = np.zeros((num_countries, num_countries), dtype=np.float32)

        for _, row in events_df.iterrows():
            a1, a2 = row["Actor1Code"], row["Actor2Code"]
            if a1 not in self.country_indices or a2 not in self.country_indices:
                continue
            i, j = self.country_indices[a1], self.country_indices[a2]
            if i == j:
                continue

            qc = row.get("QuadClass")
            if pd.notna(qc):
                if qc in (3, 4):
                    adj_conflict[i, j] += 1
                elif qc in (1, 2):
                    adj_coop[i, j] += 1

            gs = row.get("GoldsteinScale")
            if pd.notna(gs):
                gold_sum[i, j] += gs
                gold_cnt[i, j] += 1

            at = row.get("AvgTone")
            if pd.notna(at):
                tone_sum[i, j] += at
                tone_cnt[i, j] += 1

            event_cnt[i, j] += 1

        return {
            "adjacency_conflict": adj_conflict,
            "adjacency_cooperation": adj_coop,
            "adjacency_goldstein_sum": gold_sum,
            "adjacency_goldstein_count": gold_cnt,
            "adjacency_tone_sum": tone_sum,
            "adjacency_tone_count": tone_cnt,
            "adjacency_event_count": event_cnt,
            "num_events": len(events_df),
        }
