"""
Fetch and Store Historical Options Data for QQQ
Uses Polygon API for bars/snapshots and Massive API for trades data
"""
import os
import sys
import argparse
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sqlite3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients.database import MarketDatabase

STATE_FILE = "options_collection_state.json"

def fetch_options_snapshot(api_key, underlying, date=None):
    """
    Fetch options snapshot from Massive API
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
    params = {
        'apiKey': api_key,
        'limit': 250
    }
    
    print(f"Fetching options snapshot for {underlying}...")
    
    all_results = []
    next_url = None
    
    while True:
        if next_url:
            resp = requests.get(next_url)
        else:
            resp = requests.get(url, params=params)
        
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text[:200]}")
            break
        
        data = resp.json()
        results = data.get('results', [])
        all_results.extend(results)
        
        # Check for pagination
        next_url = data.get('next_url')
        if next_url:
            next_url = f"{next_url}&apiKey={api_key}"
        else:
            break
    
    return all_results


def fetch_option_bars(api_key, ticker, start_date, end_date, timespan='day', multiplier=1):
    """
    Fetch historical bars (aggregates) for an option contract
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    
    # print(f"Fetching bars for {ticker}...") 
    
    try:
        resp = requests.get(url, params=params)
        
        if resp.status_code != 200:
            print(f"Error fetching bars for {ticker}: {resp.status_code} - {resp.text}")
            return []
            
        data = resp.json()
        results = data.get('results', [])
        if not results:
            # Debug: print why no results?
            # print(f"No results for {ticker} in range {start_date}-{end_date}")
            pass
            
        return results
    except Exception as e:
        print(f"Exception fetching bars for {ticker}: {e}")
        return []


def fetch_all_contracts(api_key, underlying, limit=None, cutoff_date=None):
    """
    Fetch contracts (active and expired) for an underlying
    Sorted by expiration_date DESC to get recent ones first
    Stops when hitting contracts older than cutoff_date
    """
    print(f"Fetching contracts for {underlying} (Newest First)...")
    url = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        'underlying_ticker': underlying,
        'limit': 1000,
        'expired': 'true',
        'sort': 'expiration_date',
        'order': 'desc',
        'apiKey': api_key
    }
    
    all_contracts = []
    next_url = f"{url}?underlying_ticker={underlying}&limit=1000&expired=true&sort=expiration_date&order=desc&apiKey={api_key}"
    
    page = 0
    while True:
        if limit and len(all_contracts) >= limit:
            break
            
        try:
            resp = requests.get(next_url)
            if resp.status_code != 200:
                print(f"Error fetching contracts: {resp.status_code}")
                break
                
            data = resp.json()
            results = data.get('results', [])
            
            # If we have a cutoff date, check if we've gone past it
            if cutoff_date and results:
                # Check the last contract in this batch
                last_exp = results[-1].get('expiration_date')
                if last_exp and last_exp < cutoff_date:
                    # Add only contracts that are >= cutoff_date
                    for contract in results:
                        if contract.get('expiration_date') >= cutoff_date:
                            all_contracts.append(contract)
                        else:
                            print(f"Reached cutoff date {cutoff_date}. Stopping contract fetch.")
                            print(f"Total contracts found: {len(all_contracts)}")
                            return all_contracts
                    break
            
            all_contracts.extend(results)
            page += 1
            
            if page % 5 == 0:
                print(f"  Fetched {len(all_contracts)} contracts so far...")
            
            next_url = data.get('next_url')
            if next_url:
                next_url = f"{next_url}&apiKey={api_key}"
            else:
                break
        except Exception as e:
            print(f"Exception fetching contracts: {e}")
            break
            
    print(f"Total contracts found: {len(all_contracts)}")
    return all_contracts


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}


def store_options_data(db, underlying, options_data, date):
    # ... (Snapshot storage logic - keeping it for reference/future use)
    pass 


def main():
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    
    parser = argparse.ArgumentParser(description="Fetch Historical Options Data")
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Snapshot mode
    parser_snap = subparsers.add_parser('snapshot', help='Fetch option chain snapshots')
    parser_snap.add_argument("--symbol", type=str, default="QQQ")
    parser_snap.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser_snap.add_argument("--days", type=int, default=1)
    
    # History mode
    parser_hist = subparsers.add_parser('history', help='Fetch historical bars for a specific contract')
    parser_hist.add_argument("--contract", type=str, required=True)
    parser_hist.add_argument("--start", type=str, required=True)
    parser_hist.add_argument("--end", type=str, required=True)
    
    # Bulk mode
    parser_bulk = subparsers.add_parser('bulk', help='Bulk fetch history for ALL contracts')
    parser_bulk.add_argument("--symbol", type=str, default="QQQ")
    parser_bulk.add_argument("--limit", type=int, help="Limit number of contracts to process (for testing)")
    parser_bulk.add_argument("--resume", action="store_true", help="Resume from last saved state")
    
    args = parser.parse_args()
    
    if not api_key:
        print("Error: POLYGON_API_KEY not found")
        return
    
    db = MarketDatabase()
    
    if args.mode == 'bulk':
        print(f"=== BULK COLLECTION FOR {args.symbol} ===")
        
        # Define cutoff date
        cutoff_date = "2020-01-01"
        
        # 1. Get Contracts (only fetch until we hit cutoff date)
        contracts = fetch_all_contracts(api_key, args.symbol, args.limit, cutoff_date)
        
        # Apply limit strictly if specified
        if args.limit:
            contracts = contracts[:args.limit]
        
        # 2. Determine start index
        start_idx = 0
        state = load_state()
        if args.resume and state.get('symbol') == args.symbol:
            last_processed = state.get('last_processed_contract')
            if last_processed:
                # Find index of last processed
                for i, c in enumerate(contracts):
                    if c['ticker'] == last_processed:
                        start_idx = i + 1
                        print(f"Resuming from index {start_idx} (after {last_processed})")
                        break
        
        # 3. Iterate and Fetch
        total = len(contracts)
        print(f"Processing {total - start_idx} contracts...")
        
        processed_count = 0
        
        for i in range(start_idx, total):
            contract = contracts[i]
            ticker = contract['ticker']
            exp_date = contract['expiration_date']
            
            # Define range: 5 years ago to expiration
            end_d = exp_date
            start_d = (datetime.strptime(exp_date, "%Y-%m-%d") - timedelta(days=1825)).strftime("%Y-%m-%d")
            
            # print(f"[{i+1}/{total}] Fetching {ticker}...")
            
            bars = fetch_option_bars(api_key, ticker, start_d, end_d)
            
            if bars:
                db.insert_option_bars(ticker, bars)
                print(f"[{i+1}/{total}] {ticker}: Stored {len(bars)} bars")
            else:
                # print(f"[{i+1}/{total}] {ticker}: No data")
                pass
                
            processed_count += 1
            
            # Save state every 10 processed contracts
            if processed_count % 10 == 0:
                save_state({'symbol': args.symbol, 'last_processed_contract': ticker})
                
            # Rate limit niceness
            # time.sleep(0.05) 
            
        print(f"Bulk collection complete!")
        save_state({'symbol': args.symbol, 'last_processed_contract': 'DONE'})

    elif args.mode == 'history':
        # ... (existing history logic)
        bars = fetch_option_bars(api_key, args.contract, args.start, args.end)
        if bars:
            db.insert_option_bars(args.contract, bars)
            print(f"✓ Stored {len(bars)} bars")
        else:
            print("No bars found.")

    elif args.mode == 'snapshot':
        # ... (existing snapshot logic, simplified here)
        print("Snapshot mode not fully re-implemented in this update. Use bulk mode.")
        
    db.close()

if __name__ == "__main__":
    main()
