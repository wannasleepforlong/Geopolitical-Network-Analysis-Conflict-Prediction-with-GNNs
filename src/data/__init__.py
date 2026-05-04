"""
src.data package
================
Clean GDELT data pipeline with FIPS country filtering.

Modules:
    fips_filter        – whitelist and normalisation
    event_collector    – download GDELT export CSVs
    network_builder    – aggregate monthly networks
    data_preprocessor  – build numpy tensors for GNNs
"""
from .event_collector import GDELTEventCollector
from .network_builder import GeopoliticalNetworkBuilder
from .data_preprocessor import GNNDataPreprocessor
from .fips_filter import COUNTRY_WHITELIST, COUNTRY_LIST

__all__ = [
    "GDELTEventCollector",
    "GeopoliticalNetworkBuilder",
    "GNNDataPreprocessor",
    "COUNTRY_WHITELIST",
    "COUNTRY_LIST",
]
