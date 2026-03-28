"""
Place an order on IBKR paper account.

Usage:
    python scripts/place_order.py                           # Show account info only
    python scripts/place_order.py --buy SPY --qty 1         # Buy 1 share of SPY at market
    python scripts/place_order.py --buy SPY --qty 1 --limit 580  # Buy 1 SPY at $580 limit
    python scripts/place_order.py --sell SPY --qty 1        # Sell 1 share of SPY at market
    python scripts/place_order.py --status                  # Show open orders
    python scripts/place_order.py --cancel ORDER_ID         # Cancel an order
    python scripts/place_order.py --positions               # Show positions
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.ibkr_client import IBKRClient


def get_port():
    """Get IBKR port from env or strategy.json."""
    env_port = os.environ.get("IBKR_PAPER_PORT")
    if env_port:
        return int(env_port)
    try:
        from config import defaults as cfg
        return cfg.ibkr_paper_port()
    except Exception:
        return 4004  # Docker socat default


def get_host():
    return os.environ.get("IBKR_HOST", "127.0.0.1")


def connect():
    """Connect to IBKR and return client."""
    host = get_host()
    port = get_port()
    # Try configured port first, fall back to 4004 (socat)
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    if s.connect_ex((host, port)) != 0:
        s.close()
        port = 4004
    else:
        s.close()

    print(f"Connecting to IBKR at {host}:{port}...")
    client = IBKRClient(host=host, port=port, client_id=52, connect_timeout=15)
    client.connect()
    print(f"Connected! Account: {client.wrapper.next_order_id}")
    return client


def show_account(client):
    """Display account summary."""
    summary = client.get_account_summary()
    print("\n=== ACCOUNT SUMMARY ===")
    for key in ['NetLiquidation', 'TotalCashValue', 'BuyingPower',
                'AvailableFunds', 'ExcessLiquidity', 'UnrealizedPnL',
                'RealizedPnL', 'DayTradesRemaining']:
        if key in summary:
            vals = summary[key]
            for currency, value in vals.items():
                print(f"  {key} ({currency}): {value}")
    return summary


def show_positions(client):
    """Display current positions."""
    positions = client.get_positions()
    print(f"\n=== POSITIONS ({len(positions)}) ===")
    if not positions:
        print("  (no positions)")
    for sym, info in positions.items():
        print(f"  {sym}: {info.get('position')} shares, avg_cost=${info.get('avg_cost', 0):.2f}, account={info.get('account')}")
    return positions


def get_spy_quote(client):
    """Get current SPY quote."""
    contract = IBKRClient.stock("SPY")
    client.set_market_data_type(3)  # Delayed data
    tick = client.get_quote(contract, timeout=10)
    if not tick:
        print("  Could not get SPY quote")
        return None

    # Extract prices
    bid = tick.get("price_1", tick.get("price_66", 0))
    ask = tick.get("price_2", tick.get("price_67", 0))
    last = tick.get("price_4", tick.get("price_68", 0))
    close = tick.get("price_9", tick.get("price_75", 0))
    print(f"\n=== SPY QUOTE ===")
    print(f"  Last: ${last:.2f}  Bid: ${bid:.2f}  Ask: ${ask:.2f}  Prev Close: ${close:.2f}")
    return {"bid": bid, "ask": ask, "last": last, "close": close}


def place_buy(client, symbol, qty, limit_price=None):
    """Place a buy order."""
    contract = IBKRClient.stock(symbol)
    # Resolve contract
    resolved = client.resolve_contract(contract, timeout=10)
    if resolved:
        contract = resolved

    if limit_price:
        order = client.limit_order("BUY", qty, limit_price)
        print(f"\nPlacing LIMIT BUY: {qty} x {symbol} @ ${limit_price:.2f}")
    else:
        order = client.market_order("BUY", qty)
        print(f"\nPlacing MARKET BUY: {qty} x {symbol}")

    oid = client.place_order(contract, order, wait_seconds=2.0)
    print(f"  Order ID: {oid}")

    # Check status
    status = client.get_order_status(oid)
    if status:
        print(f"  Status: {status.get('status')}, Filled: {status.get('filled')}, Avg Price: ${status.get('avg_fill_price', 0):.2f}")
    else:
        print("  Status: Pending (check --status for updates)")
    return oid


def place_sell(client, symbol, qty, limit_price=None):
    """Place a sell order."""
    contract = IBKRClient.stock(symbol)
    resolved = client.resolve_contract(contract, timeout=10)
    if resolved:
        contract = resolved

    if limit_price:
        order = client.limit_order("SELL", qty, limit_price)
        print(f"\nPlacing LIMIT SELL: {qty} x {symbol} @ ${limit_price:.2f}")
    else:
        order = client.market_order("SELL", qty)
        print(f"\nPlacing MARKET SELL: {qty} x {symbol}")

    oid = client.place_order(contract, order, wait_seconds=2.0)
    print(f"  Order ID: {oid}")

    status = client.get_order_status(oid)
    if status:
        print(f"  Status: {status.get('status')}, Filled: {status.get('filled')}, Avg Price: ${status.get('avg_fill_price', 0):.2f}")
    else:
        print("  Status: Pending (check --status for updates)")
    return oid


def show_orders(client):
    """Show open orders."""
    # Request open orders
    client._ec.reqAllOpenOrders()
    time.sleep(2)
    orders = client.wrapper.open_orders
    statuses = client.wrapper.order_statuses
    print(f"\n=== OPEN ORDERS ({len(orders)}) ===")
    if not orders:
        print("  (no open orders)")
    for oid, info in orders.items():
        contract = info.get("contract")
        order = info.get("order")
        st = statuses.get(oid, {})
        sym = contract.symbol if contract else "?"
        action = order.action if order else "?"
        qty = order.totalQuantity if order else 0
        otype = order.orderType if order else "?"
        lmt = order.lmtPrice if order else 0
        status = st.get("status", "Unknown")
        filled = st.get("filled", 0)
        print(f"  #{oid}: {action} {qty} {sym} ({otype} @ ${lmt:.2f}) — {status} (filled: {filled})")


def cancel_order_by_id(client, order_id):
    """Cancel an order by ID."""
    print(f"\nCancelling order #{order_id}...")
    client.cancel_order(order_id)
    time.sleep(1)
    print("  Cancel request sent.")


def main():
    parser = argparse.ArgumentParser(description="Place orders on IBKR paper account")
    parser.add_argument("--buy", metavar="SYMBOL", help="Buy a symbol (e.g. SPY)")
    parser.add_argument("--sell", metavar="SYMBOL", help="Sell a symbol (e.g. SPY)")
    parser.add_argument("--qty", type=int, default=1, help="Quantity (default: 1)")
    parser.add_argument("--limit", type=float, help="Limit price (omit for market order)")
    parser.add_argument("--status", action="store_true", help="Show open orders")
    parser.add_argument("--positions", action="store_true", help="Show positions")
    parser.add_argument("--cancel", type=int, help="Cancel an order by ID")
    parser.add_argument("--quote", metavar="SYMBOL", help="Get a quote for a symbol")
    args = parser.parse_args()

    client = connect()
    try:
        # Always show account summary
        show_account(client)

        if args.positions:
            show_positions(client)

        if args.status:
            show_orders(client)

        if args.cancel:
            cancel_order_by_id(client, args.cancel)

        if args.quote:
            contract = IBKRClient.stock(args.quote.upper())
            client.set_market_data_type(3)
            tick = client.get_quote(contract, timeout=10)
            if tick:
                bid = tick.get("price_1", tick.get("price_66", 0))
                ask = tick.get("price_2", tick.get("price_67", 0))
                last = tick.get("price_4", tick.get("price_68", 0))
                print(f"\n{args.quote.upper()}: Last=${last:.2f} Bid=${bid:.2f} Ask=${ask:.2f}")

        if args.buy:
            symbol = args.buy.upper()
            get_spy_quote(client) if symbol == "SPY" else None
            place_buy(client, symbol, args.qty, args.limit)
            time.sleep(3)
            show_orders(client)
            show_positions(client)

        if args.sell:
            symbol = args.sell.upper()
            place_sell(client, symbol, args.qty, args.limit)
            time.sleep(3)
            show_orders(client)
            show_positions(client)

        if not any([args.buy, args.sell, args.status, args.positions, args.cancel, args.quote]):
            # Just show everything
            show_positions(client)
            get_spy_quote(client)

    finally:
        client.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
