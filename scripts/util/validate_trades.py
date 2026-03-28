"""
Trade Validation Script
- For each trade in backtest_2026_jan_feb.csv, verify:
  1. Entry price matches raw option data + slippage
  2. Exit price matches the correct future bar + slippage
  3. Exit reason is correct (PROFIT/STOP/TIME hit at right bar)
  4. P&L calculation is correct (contracts * 100 * (exit-entry) - commission)
  5. Capital tracking is correct (cumulative)
  6. RSI signal direction matches (RSI>70→CALL, RSI<30→PUT)
  7. Trade window is within 10:00-11:00
  8. Option ticker format and strike make sense
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import sqlite3
from clients.database import MarketDatabase
from config import defaults as cfg

# Config constants
SLIPPAGE = 0.005       # 0.5%
COMMISSION = 0.65      # per contract per side
PROFIT_TARGET = 0.50   # 50%
STOP_LOSS = 0.35       # 35%
MAX_HOLD_BARS = 16
INITIAL_CAPITAL = cfg.initial_capital()
RSI_CALL = 70
RSI_PUT = 30

def main():
    db = MarketDatabase()
    
    # Load trades
    trades = pd.read_csv('output/backtest_2026_jan_feb.csv')
    print(f'Loaded {len(trades)} trades to validate')
    print(f'Date range: {trades["date"].min()} to {trades["date"].max()}')
    print()
    
    # Load raw option data (same query as engine)
    print('Loading raw option data...')
    conn = sqlite3.connect('data/market_data.db')
    query = """
    SELECT option_ticker, underlying, timestamp, date, time,
           open, high, low, close, volume, expiration, strike, option_type
    FROM options_intraday
    WHERE underlying = 'SPY' AND date = expiration
          AND date >= '2026-01-01' AND date <= '2026-02-15'
    ORDER BY date, time
    """
    raw_options = pd.read_sql_query(query, conn)
    raw_options = raw_options[raw_options['close'] > 0].copy()
    conn.close()
    print(f'  Raw options: {len(raw_options):,} rows')
    
    # Build lookup
    raw_options['time'] = raw_options['time'].astype(str)
    
    errors = []
    warnings = []
    checked = 0
    
    for i, trade in trades.iterrows():
        trade_id = f"T{i+1} {trade['date']} {trade['time']} {trade['direction']} {trade['strike']}"
        issues = []
        
        # ============================================================
        # CHECK 1: Trading window (10:00-11:00)
        # ============================================================
        hour = int(trade['time'].split(':')[0])
        minute = int(trade['time'].split(':')[1])
        if hour < 10 or hour > 11 or (hour == 11 and minute > 0):
            # Some trades at 11:xx are valid if the engine allows entries up to trade_end
            # The config says trade_end_hour=11, trade_end_minute=0, but entries at 11:00 may produce signals
            # Let's flag entries after 11:00 as warnings
            if hour > 11 or (hour == 11 and minute > 0):
                issues.append(f'WARN: Entry at {trade["time"]} outside 10:00-11:00 window')
        
        # ============================================================
        # CHECK 2: RSI-Direction consistency
        # ============================================================
        rsi = trade['rsi']
        direction = trade['direction']
        if direction == 'CALL' and rsi < RSI_CALL:
            issues.append(f'ERROR: CALL signal but RSI={rsi:.1f} < {RSI_CALL}')
        elif direction == 'PUT' and rsi > RSI_PUT:
            issues.append(f'ERROR: PUT signal but RSI={rsi:.1f} > {RSI_PUT}')
        
        # ============================================================
        # CHECK 3: Option ticker format
        # ============================================================
        ticker = trade['option_ticker']
        expected_type = 'C' if direction == 'CALL' else 'P'
        if expected_type not in ticker:
            issues.append(f'ERROR: Ticker {ticker} missing {expected_type} for {direction}')
        
        strike_str = f'{int(trade["strike"]):05d}000'
        if strike_str not in ticker:
            issues.append(f'ERROR: Ticker {ticker} does not match strike {trade["strike"]}')
        
        # ============================================================
        # CHECK 4: Entry price = raw close * (1 + slippage)
        # ============================================================
        opt_at_entry = raw_options[
            (raw_options['option_ticker'] == ticker) &
            (raw_options['date'] == trade['date']) &
            (raw_options['time'] == trade['time'])
        ]
        
        if len(opt_at_entry) == 0:
            issues.append(f'ERROR: No raw data found for {ticker} at {trade["date"]} {trade["time"]}')
        else:
            raw_close = opt_at_entry.iloc[0]['close']
            expected_entry = raw_close * (1 + SLIPPAGE)
            actual_entry = trade['entry']
            entry_diff = abs(actual_entry - expected_entry) / expected_entry * 100
            if entry_diff > 0.01:  # >0.01% tolerance
                issues.append(f'ERROR: Entry ${actual_entry:.4f} != raw ${raw_close:.4f} * 1.005 = ${expected_entry:.4f} (diff {entry_diff:.3f}%)')
        
        # ============================================================
        # CHECK 5: Exit price & reason validation
        # ============================================================
        # Get future bars for this option after entry
        future = raw_options[
            (raw_options['option_ticker'] == ticker) &
            (raw_options['date'] == trade['date']) &
            (raw_options['time'] > trade['time'])
        ].sort_values('time')
        
        if len(future) == 0 and trade['exit_reason'] != 'TIME':
            issues.append(f'WARN: No future bars found for exit validation')
        else:
            entry_price = trade['entry']
            expected_exit_reason = None
            expected_exit_bar = None
            expected_exit_raw_price = None
            
            for bar_idx, (_, bar) in enumerate(future.iterrows()):
                if bar_idx >= MAX_HOLD_BARS:
                    break
                bar_price = bar['close']
                pct_change = (bar_price - entry_price) / entry_price
                
                if pct_change >= PROFIT_TARGET:
                    expected_exit_reason = 'PROFIT'
                    expected_exit_bar = bar_idx + 1
                    expected_exit_raw_price = bar_price
                    break
                elif pct_change <= -STOP_LOSS:
                    expected_exit_reason = 'STOP'
                    expected_exit_bar = bar_idx + 1
                    expected_exit_raw_price = bar_price
                    break
                elif bar_idx + 1 >= MAX_HOLD_BARS:
                    expected_exit_reason = 'TIME'
                    expected_exit_bar = bar_idx + 1
                    expected_exit_raw_price = bar_price
                    break
            
            # If we ran out of bars before max_hold_bars
            if expected_exit_reason is None and len(future) > 0:
                last_bar = future.iloc[min(MAX_HOLD_BARS - 1, len(future) - 1)]
                expected_exit_reason = 'TIME'
                expected_exit_bar = min(MAX_HOLD_BARS, len(future))
                expected_exit_raw_price = last_bar['close']
            
            if expected_exit_reason and expected_exit_raw_price:
                # Check exit reason
                if trade['exit_reason'] != expected_exit_reason:
                    issues.append(f'ERROR: Exit reason {trade["exit_reason"]} != expected {expected_exit_reason}')
                
                # Check bars held
                if expected_exit_bar and trade['bars_held'] != expected_exit_bar:
                    issues.append(f'WARN: Bars held {trade["bars_held"]} != expected {expected_exit_bar}')
                
                # Check exit price = raw_price * (1 - slippage)
                expected_exit = expected_exit_raw_price * (1 - SLIPPAGE)
                actual_exit = trade['exit']
                exit_diff = abs(actual_exit - expected_exit) / expected_exit * 100
                if exit_diff > 0.01:
                    issues.append(f'ERROR: Exit ${actual_exit:.4f} != raw ${expected_exit_raw_price:.4f} * 0.995 = ${expected_exit:.4f} (diff {exit_diff:.3f}%)')
        
        # ============================================================
        # CHECK 6: P&L calculation
        # ============================================================
        contracts = trade['num_contracts']
        expected_gross = contracts * 100 * (trade['exit'] - trade['entry'])
        expected_commission = COMMISSION * contracts * 2
        expected_net = expected_gross - expected_commission
        pnl_diff = abs(trade['pnl'] - expected_net)
        if pnl_diff > 0.01:
            issues.append(f'ERROR: P&L ${trade["pnl"]:.2f} != expected ${expected_net:.2f} (gross=${expected_gross:.2f} - comm=${expected_commission:.2f})')
        
        # ============================================================
        # CHECK 7: Capital tracking
        # ============================================================
        if i == 0:
            expected_capital = INITIAL_CAPITAL + trade['pnl']
        else:
            expected_capital = trades.iloc[i-1]['capital'] + trade['pnl']
        
        cap_diff = abs(trade['capital'] - expected_capital)
        if cap_diff > 0.01:
            issues.append(f'ERROR: Capital ${trade["capital"]:.2f} != expected ${expected_capital:.2f}')
        
        # ============================================================
        # CHECK 8: Contracts > 0 and option price in range
        # ============================================================
        if contracts <= 0:
            issues.append(f'ERROR: num_contracts={contracts}')
        
        # Raw entry (before slippage) should be $0.50-$2.00
        raw_entry = trade['entry'] / (1 + SLIPPAGE)
        if raw_entry < 0.49 or raw_entry > 2.01:
            issues.append(f'WARN: Raw entry price ${raw_entry:.2f} outside $0.50-$2.00 range')
        
        # ============================================================
        # CHECK 9: Exit makes directional sense
        # ============================================================
        if trade['exit_reason'] == 'PROFIT' and trade['exit'] <= trade['entry']:
            issues.append(f'ERROR: PROFIT exit but exit ${trade["exit"]:.4f} <= entry ${trade["entry"]:.4f}')
        if trade['exit_reason'] == 'STOP' and trade['exit'] >= trade['entry']:
            issues.append(f'ERROR: STOP exit but exit ${trade["exit"]:.4f} >= entry ${trade["entry"]:.4f}')
        
        checked += 1
        
        if issues:
            error_issues = [x for x in issues if x.startswith('ERROR')]
            warn_issues = [x for x in issues if x.startswith('WARN')]
            if error_issues:
                errors.append((trade_id, error_issues))
            if warn_issues:
                warnings.append((trade_id, warn_issues))
            
            # Print all issues for this trade
            print(f'{"!" if error_issues else "?"} {trade_id}')
            for issue in issues:
                print(f'    {issue}')
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print('\n' + '=' * 70)
    print('  VALIDATION SUMMARY')
    print('=' * 70)
    print(f'  Trades checked:    {checked}')
    print(f'  Trades with errors: {len(errors)}')
    print(f'  Trades with warns:  {len(warnings)}')
    print(f'  Clean trades:      {checked - len(errors) - len(set(e[0] for e in errors) | set(w[0] for w in warnings))}')
    
    # Count error types
    all_error_msgs = [msg for _, msgs in errors for msg in msgs]
    all_warn_msgs = [msg for _, msgs in warnings for msg in msgs]
    
    if all_error_msgs:
        print(f'\n  ERROR BREAKDOWN:')
        error_types = {}
        for msg in all_error_msgs:
            key = msg.split(':')[1].strip().split(' ')[0]  # First word after ERROR:
            error_types[key] = error_types.get(key, 0) + 1
        for k, v in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f'    {k}: {v}')
    
    if all_warn_msgs:
        print(f'\n  WARNING BREAKDOWN:')
        warn_types = {}
        for msg in all_warn_msgs:
            key = msg.split(':')[1].strip().split(' ')[0]
            warn_types[key] = warn_types.get(key, 0) + 1
        for k, v in sorted(warn_types.items(), key=lambda x: -x[1]):
            print(f'    {k}: {v}')
    
    if not errors and not warnings:
        print('\n  ALL TRADES VALIDATED SUCCESSFULLY')
    elif not errors:
        print(f'\n  NO ERRORS - {len(warnings)} warnings (non-critical)')
    else:
        print(f'\n  {len(errors)} TRADES HAVE ERRORS - REVIEW NEEDED')
    
    # ============================================================
    # SPOT CHECK: Print 5 sample trades with full detail
    # ============================================================
    print('\n' + '=' * 70)
    print('  SPOT CHECK: 5 Sample Trades (full detail)')
    print('=' * 70)
    
    sample_indices = [0, len(trades)//4, len(trades)//2, 3*len(trades)//4, len(trades)-1]
    for idx in sample_indices:
        t = trades.iloc[idx]
        
        # Get raw data
        opt_entry = raw_options[
            (raw_options['option_ticker'] == t['option_ticker']) &
            (raw_options['date'] == t['date']) &
            (raw_options['time'] == t['time'])
        ]
        raw_entry_price = opt_entry.iloc[0]['close'] if len(opt_entry) > 0 else 'N/A'
        
        # Get exit bar raw data
        future = raw_options[
            (raw_options['option_ticker'] == t['option_ticker']) &
            (raw_options['date'] == t['date']) &
            (raw_options['time'] > t['time'])
        ].sort_values('time')
        
        bars_held = int(t['bars_held'])
        exit_bar = future.iloc[bars_held - 1] if len(future) >= bars_held else None
        raw_exit_price = exit_bar['close'] if exit_bar is not None else 'N/A'
        exit_time = exit_bar['time'] if exit_bar is not None else 'N/A'
        
        print(f'\n  Trade #{idx+1}: {t["date"]} {t["time"]} {t["direction"]} {t["strike"]:.0f}')
        print(f'    Ticker:      {t["option_ticker"]}')
        print(f'    RSI:         {t["rsi"]:.1f} ({"OK" if (t["direction"]=="CALL" and t["rsi"]>=RSI_CALL) or (t["direction"]=="PUT" and t["rsi"]<=RSI_PUT) else "MISMATCH"})')
        print(f'    Raw Entry:   ${raw_entry_price:.4f}' if isinstance(raw_entry_price, float) else f'    Raw Entry:   {raw_entry_price}')
        print(f'    Slipped Entry: ${t["entry"]:.4f} (raw * 1.005 = ${float(raw_entry_price) * 1.005:.4f})' if isinstance(raw_entry_price, float) else '')
        print(f'    Raw Exit:    ${raw_exit_price:.4f} at {exit_time}' if isinstance(raw_exit_price, float) else f'    Raw Exit:    {raw_exit_price}')
        print(f'    Slipped Exit: ${t["exit"]:.4f} (raw * 0.995 = ${float(raw_exit_price) * 0.995:.4f})' if isinstance(raw_exit_price, float) else '')
        print(f'    Exit Reason: {t["exit_reason"]} after {bars_held} bars')
        
        # Show pct change at exit
        pct = (t['exit'] - t['entry']) / t['entry'] * 100
        print(f'    Pct Change:  {pct:+.1f}% (PT=+50%, SL=-35%)')
        print(f'    Contracts:   {int(t["num_contracts"])}')
        print(f'    Gross P&L:   ${int(t["num_contracts"]) * 100 * (t["exit"] - t["entry"]):+.2f}')
        print(f'    Commission:  ${COMMISSION * int(t["num_contracts"]) * 2:.2f}')
        print(f'    Net P&L:     ${t["pnl"]:+.2f}')
        print(f'    Capital:     ${t["capital"]:,.2f}')
        
        # Show price path for first few bars
        if len(future) > 0:
            print(f'    Price path (entry → exit):')
            print(f'      Bar 0 (entry): ${raw_entry_price:.4f}' if isinstance(raw_entry_price, float) else '')
            for bi in range(min(bars_held + 2, len(future))):
                bar = future.iloc[bi]
                bp = bar['close']
                pct_from_entry = (bp - t['entry']) / t['entry'] * 100
                marker = ' ← EXIT' if bi + 1 == bars_held else ''
                trigger = ''
                if pct_from_entry >= PROFIT_TARGET * 100:
                    trigger = ' [PT HIT]'
                elif pct_from_entry <= -STOP_LOSS * 100:
                    trigger = ' [SL HIT]'
                elif bi + 1 >= MAX_HOLD_BARS:
                    trigger = ' [TIME]'
                print(f'      Bar {bi+1} ({bar["time"]}): ${bp:.4f} ({pct_from_entry:+.1f}%){trigger}{marker}')


if __name__ == '__main__':
    main()
