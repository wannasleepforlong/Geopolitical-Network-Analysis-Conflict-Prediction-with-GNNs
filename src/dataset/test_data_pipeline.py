import logging
from datetime import datetime

from .event_collector import GDELTEventCollector
from .network_builder import GeopoliticalNetworkBuilder
from .data_preprocessor import GNNDataPreprocessor


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"result/logs/test_gdelt_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    collector = GDELTEventCollector(logger=logger)
    events = collector.fetch_events(
        start_date="2026-02-18", 
        end_date="2026-02-28",
        countries=['USA', 'CHN', 'RUS', 'IND']
    )
    
    if not events.empty:
        builder = GeopoliticalNetworkBuilder(logger=logger)
        networks = builder.build_temporal_networks(events)
        print(f"Successfully built networks for periods: {list(networks.keys())}")

        preprocessor = GNNDataPreprocessor(logger=logger)
        preprocessor.process_and_save(networks, builder.country_indices)
    else:
        print("No events collected. Check your date range or connection.")