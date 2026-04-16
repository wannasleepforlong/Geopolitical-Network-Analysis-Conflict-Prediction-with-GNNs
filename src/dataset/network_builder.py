import pandas as pd
import numpy as np
import logging 
from typing import Dict, Any


class GeopoliticalNetworkBuilder:
    def __init__(self, logger = None):
        self.networks = {}
        self.country_indices = {}
        self.reverse_indices = {}
        self.logger = logger or logging.getLogger("null")
        self.logger.addHandler(logging.NullHandler())
    
    def build_temporal_networks(self, events_df: pd.DataFrame, time_window: str = 'M') -> Dict[str, Any]:
        if events_df.empty:
            self.logger.error("DataFrame is empty. Cannot build networks.")
            return {}

        # Ensure numeric columns are properly typed (handles both fresh and cached data)
        events_df['GoldsteinScale'] = pd.to_numeric(events_df['GoldsteinScale'], errors='coerce')
        events_df['AvgTone'] = pd.to_numeric(events_df['AvgTone'], errors='coerce')
        events_df['QuadClass'] = pd.to_numeric(events_df['QuadClass'], errors='coerce')
        events_df['EventDate'] = pd.to_datetime(events_df['EventDate'], format='%Y%m%d', errors='coerce')
        events_df = events_df.dropna(subset=['EventDate'])
        events_df['Period'] = events_df['EventDate'].dt.to_period(time_window)
        
        all_actors = set(events_df['Actor1Code'].dropna()) | set(events_df['Actor2Code'].dropna())
        self.country_indices = {code: i for i, code in enumerate(sorted(all_actors))}
        self.reverse_indices = {i: code for code, i in self.country_indices.items()}
        num_countries = len(self.country_indices)
        
        for period, group_events in events_df.groupby('Period'):
            self.networks[str(period)] = self._build_single_network(group_events, num_countries)
        
        return self.networks

    def _build_single_network(self, events_df: pd.DataFrame, num_countries: int) -> Dict[str, Any]:
        adj_conflict = np.zeros((num_countries, num_countries))
        adj_cooperation = np.zeros((num_countries, num_countries))
        adj_goldstein_sum = np.zeros((num_countries, num_countries))
        adj_goldstein_count = np.zeros((num_countries, num_countries))
        adj_tone_sum = np.zeros((num_countries, num_countries))
        adj_tone_count = np.zeros((num_countries, num_countries))
        adj_event_count = np.zeros((num_countries, num_countries))
        
        for _, row in events_df.iterrows():
            a1, a2 = row['Actor1Code'], row['Actor2Code']
            if a1 in self.country_indices and a2 in self.country_indices:
                i, j = self.country_indices[a1], self.country_indices[a2]
                quad = row['QuadClass']
                if quad in [3, 4]:
                    adj_conflict[i, j] += 1
                elif quad in [1, 2]:
                    adj_cooperation[i, j] += 1

                if pd.notna(row['GoldsteinScale']):
                    adj_goldstein_sum[i, j] += row['GoldsteinScale']
                    adj_goldstein_count[i, j] += 1
                if pd.notna(row['AvgTone']):
                    adj_tone_sum[i, j] += row['AvgTone']
                    adj_tone_count[i, j] += 1

                adj_event_count[i, j] += 1
        
        return {
            'adjacency_conflict': adj_conflict,
            'adjacency_cooperation': adj_cooperation,
            'adjacency_goldstein_sum': adj_goldstein_sum,
            'adjacency_goldstein_count': adj_goldstein_count,
            'adjacency_tone_sum': adj_tone_sum,
            'adjacency_tone_count': adj_tone_count,
            'adjacency_event_count': adj_event_count,
            'num_events': len(events_df),
        }

