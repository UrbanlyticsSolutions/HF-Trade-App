import os
import logging
import argparse
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from clients.fmp_stable_client import FMPStableClient
from clients.database import MarketDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FetchIntraday')

def fetch_data(days=60):
    """Fetch intraday data for the last N days"""
    load_dotenv()
    api_key = os.getenv('FMP_API_KEY')
    
    if not api_key:
        logger.error("FMP_API_KEY not found in environment variables")
        return

    client = FMPStableClient(api_key)
    db = MarketDatabase()
    
    symbol = "QQQ"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching {days} days of 1min intraday data for {symbol}...")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    
    # Fetch data
    # Note: FMP historical-chart/1min usually returns limited data per call, 
    # but the client might handle date ranges or we might need to loop.
    # Let's try fetching in chunks of 5 days to be safe and avoid timeouts/limits
    
    current_start = start_date
    total_records = 0
    
    # Loop must include the ending day so we do not stop one day early
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=5), end_date)
        
        s_str = current_start.strftime('%Y-%m-%d')
        e_str = current_end.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching chunk: {s_str} to {e_str}")
        
        try:
            data = client.historical_chart_1min(symbol, s_str, e_str)
            
            if data:
                logger.info(f"Received {len(data)} records")
                db.insert_intraday_data(symbol, data)
                total_records += len(data)
            else:
                logger.warning(f"No data received for {s_str} to {e_str}")
                
        except Exception as e:
            logger.error(f"Error fetching chunk: {e}")
            
        current_start = current_end + timedelta(days=1) # Move to next day
        
    logger.info(f"Total records inserted: {total_records}")

def main():
    parser = argparse.ArgumentParser(description="Backfill 1-minute intraday data")
    parser.add_argument("--days", type=int, default=180, help="Number of trailing days to fetch")
    args = parser.parse_args()
    fetch_data(days=args.days)


if __name__ == "__main__":
    main()
