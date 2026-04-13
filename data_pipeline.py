import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GDELT Event Collector (CORRECTED)
# ─────────────────────────────────────────────────────────────────────────────

class GDELTEventCollector:
    GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    
    # Define the schema explicitly to ensure empty DataFrames still have columns
    GDELT_SCHEMA = {
        'EventID': 0,
        'EventDate': 1,
        'Actor1Code': 4,
        'Actor2Code': 5,
        'EventCode': 8,
        'EventRootCode': 9,
        'GoldsteinScale': 15,
        'NumArticles': 32,
        'AvgTone': 34,
        'QuadClass': 27,
    }
    
    def __init__(self, cache_dir: str = "./gdelt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_gdelt_file_list(self, start_date: str, end_date: str) -> List[str]:
        """Fetch GDELT master file list and filter by '.export.' and date."""
        try:
            resp = requests.get(self.GDELT_MASTER_URL, timeout=30)
            resp.raise_for_status()
            
            lines = resp.text.strip().split('\n')
            urls = []
            
            # Date objects for comparison
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            for line in lines:
                parts = line.split()
                if len(parts) < 3: continue
                
                url = parts[2]
                
                # FIX: GDELT v2 events are stored in '.export.CSV.zip' files
                if '.export.CSV.zip' not in url:
                    continue
                
                try:
                    # URL format: http://.../YYYYMMDDHHMMSS.export.CSV.zip
                    file_name = url.split('/')[-1]
                    date_str = file_name[:8] 
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if start <= file_date <= end:
                        urls.append(url)
                except (ValueError, IndexError):
                    continue
            
            logger.info(f"Found {len(urls)} GDELT export files.")
            return sorted(urls)
        except Exception as e:
            logger.error(f"Failed to fetch master list: {e}")
            return []

    def fetch_events(
        self, 
        start_date: str,
        end_date: str,
        countries: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_file = self.cache_dir / f"events_{start_date}_{end_date}.pkl"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached events from {cache_file}")
            return pickle.load(open(cache_file, 'rb'))
        
        urls = self._get_gdelt_file_list(start_date, end_date)
        
        # FIX: Ensure we return a DataFrame with correct columns even if no URLs found
        if not urls:
            logger.warning("No GDELT files found for date range.")
            return pd.DataFrame(columns=self.GDELT_SCHEMA.keys())
        
        events_list = []
        # Sampling for demonstration (Use all urls in production)
        # sample_urls = urls[::max(1, len(urls)//10)] 
        sample_urls = urls
        
        
        for url in tqdm(sample_urls, desc="Downloading GDELT events"):
            df = self._fetch_gdelt_file(url)
            if df is not None:
                events_list.append(df)
        
        if not events_list:
            return pd.DataFrame(columns=self.GDELT_SCHEMA.keys())
        
        events_df = pd.concat(events_list, ignore_index=True)
        
        if countries:
            mask = (events_df['Actor1Code'].isin(countries)) | (events_df['Actor2Code'].isin(countries))
            events_df = events_df[mask]
        
        pickle.dump(events_df, open(cache_file, 'wb'))
        return events_df

    def _fetch_gdelt_file(self, url: str) -> Optional[pd.DataFrame]:
        import io, zipfile
        try:
            resp = requests.get(url, timeout=60)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_file = z.namelist()[0]
                with z.open(csv_file) as f:
                    df = pd.read_csv(
                        f, sep='\t', header=None,
                        usecols=list(self.GDELT_SCHEMA.values()),
                        names=list(self.GDELT_SCHEMA.keys()),
                        dtype={'EventDate': str, 'Actor1Code': str, 'Actor2Code': str},
                        low_memory=False
                    )
                    # Convert numeric columns to proper types
                    df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
                    df['QuadClass'] = pd.to_numeric(df['QuadClass'], errors='coerce')
                    return df
        except Exception:
            return None

# ─────────────────────────────────────────────────────────────────────────────
# 2. Geopolitical Network Builder (WITH SAFETY CHECKS)
# ─────────────────────────────────────────────────────────────────────────────

class GeopoliticalNetworkBuilder:
    def __init__(self):
        self.networks = {}
        self.country_indices = {}
        self.reverse_indices = {}
    
    def build_temporal_networks(self, events_df: pd.DataFrame, time_window: str = 'M') -> Dict[str, Any]:
        if events_df.empty:
            logger.error("DataFrame is empty. Cannot build networks.")
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


class GNNDataPreprocessor:
    def __init__(self, output_dir: str = "./gdelt_processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save(self, networks: Dict[str, Any], country_indices: Dict[str, int]) -> Dict[str, Any]:
        periods = sorted(networks.keys())
        num_periods = len(periods)
        num_countries = len(country_indices)

        node_features = np.zeros((num_periods, num_countries, 6), dtype=np.float32)
        edge_features = np.zeros((num_periods, num_countries, num_countries, 5), dtype=np.float32)
        edge_labels = np.zeros((num_periods, num_countries, num_countries), dtype=np.float32)
        valid_mask = np.zeros((num_periods, num_countries, num_countries), dtype=bool)

        avg_goldstein = []
        event_presence = []

        for t, period in enumerate(periods):
            net = networks[period]
            conflict = net['adjacency_conflict']
            coop = net['adjacency_cooperation']
            tone_avg = np.divide(
                net['adjacency_tone_sum'],
                net['adjacency_tone_count'],
                out=np.zeros_like(net['adjacency_tone_sum']),
                where=net['adjacency_tone_count'] > 0,
            )
            goldstein_avg = np.divide(
                net['adjacency_goldstein_sum'],
                net['adjacency_goldstein_count'],
                out=np.zeros_like(net['adjacency_goldstein_sum']),
                where=net['adjacency_goldstein_count'] > 0,
            )
            total_outgoing = conflict + coop
            outgoing_counts = total_outgoing.sum(axis=1)
            incoming_counts = total_outgoing.sum(axis=0)

            node_features[t, :, 0] = conflict.sum(axis=1)
            node_features[t, :, 1] = conflict.sum(axis=0)
            node_features[t, :, 2] = coop.sum(axis=1)
            node_features[t, :, 3] = np.divide(
                net['adjacency_tone_sum'].sum(axis=1),
                np.maximum(1, outgoing_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=outgoing_counts > 0,
            )
            node_features[t, :, 4] = np.divide(
                net['adjacency_tone_sum'].sum(axis=0),
                np.maximum(1, incoming_counts),
                out=np.zeros(num_countries, dtype=np.float32),
                where=incoming_counts > 0,
            )
            node_features[t, :, 5] = outgoing_counts + incoming_counts

            edge_features[t, :, :, 0] = conflict
            edge_features[t, :, :, 1] = coop
            edge_features[t, :, :, 2] = tone_avg
            edge_features[t, :, :, 3] = goldstein_avg
            edge_features[t, :, :, 4] = np.divide(
                np.abs(conflict - coop),
                np.maximum(1, conflict + coop),
                out=np.zeros_like(conflict, dtype=np.float32),
                where=(conflict + coop) > 0,
            )

            valid_mask[t] = (total_outgoing > 0)
            np.fill_diagonal(valid_mask[t], False)

            avg_goldstein.append(goldstein_avg)
            event_presence.append(total_outgoing > 0)

        for t in range(num_periods - 1):
            current_gold = avg_goldstein[t]
            next_gold = avg_goldstein[t + 1]
            label_mask = (event_presence[t] | event_presence[t + 1])
            edge_labels[t] = (
                (next_gold < current_gold - 0.5) & label_mask
            ).astype(np.float32)
        np.fill_diagonal(edge_labels[-1], 0.0)

        np.save(self.output_dir / 'node_features.npy', node_features)
        np.save(self.output_dir / 'edge_features.npy', edge_features)
        np.save(self.output_dir / 'edge_labels.npy', edge_labels)
        np.save(self.output_dir / 'valid_mask.npy', valid_mask)

        metadata = {
            'country_indices': country_indices,
            'reverse_indices': {str(i): code for i, code in sorted({v: k for k, v in country_indices.items()}.items())},
            'periods': periods,
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved processed data to {self.output_dir}")
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'valid_mask': valid_mask,
            'metadata': metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    collector = GDELTEventCollector()
    events = collector.fetch_events(
        start_date="2025-01-01", 
        end_date="2026-01-01",
        countries=['USA', 'CHN', 'RUS', 'IND']
    )
    
    if not events.empty:
        builder = GeopoliticalNetworkBuilder()
        networks = builder.build_temporal_networks(events)
        print(f"Successfully built networks for periods: {list(networks.keys())}")

        preprocessor = GNNDataPreprocessor()
        preprocessor.process_and_save(networks, builder.country_indices)
    else:
        print("No events collected. Check your date range or connection.")

if __name__ == "__main__":
    main()