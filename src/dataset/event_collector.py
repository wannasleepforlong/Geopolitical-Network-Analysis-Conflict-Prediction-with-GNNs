from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pickle
import logging

import pandas as pd
import requests
from tqdm import tqdm


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
    
    def __init__(self, logger = None, cache_dir: str = "./data/gdelt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger or logging.getLogger("null")
        self.logger.addHandler(logging.NullHandler())


    def _get_gdelt_file_list(self, start_date: str, end_date: str) -> List[str]:
        """Fetch GDELT master file list and filter by '.export.' and date."""
        try:
            resp = requests.get(self.GDELT_MASTER_URL, timeout=30)
            resp.raise_for_status()
            
            lines = resp.text.strip().split('\n')
            urls = []
            
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
            
            self.logger.info(f"Found {len(urls)} GDELT export files.")
            return sorted(urls)
        except Exception as e:
            self.logger.error(f"Failed to fetch master list: {e}")
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
            self.logger.info(f"Loading cached events from {cache_file}")
            return pickle.load(open(cache_file, 'rb'))
        
        urls = self._get_gdelt_file_list(start_date, end_date)
        
        # FIX: Ensure we return a DataFrame with correct columns even if no URLs found
        if not urls:
            self.logger.warning("No GDELT files found for date range.")
            return pd.DataFrame(columns=list(GDELTEventCollector.GDELT_SCHEMA.keys()))
        
        events_list = []
        # Sampling for demonstration (Use all urls in production)
        # sample_urls = urls[::max(1, len(urls)//10)] 
        sample_urls = urls
        
        
        for url in tqdm(sample_urls, desc="Downloading GDELT events"):
            df = self._fetch_gdelt_file(url)
            if df is not None:
                events_list.append(df)
        
        if not events_list:
            return pd.DataFrame(columns=list(GDELTEventCollector.GDELT_SCHEMA.keys()))
        
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