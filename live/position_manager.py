"""
Position Manager - Track and manage live positions

Handles:
- Position tracking from broker (IBKR or Questrade)
- Position sizing
- Risk management per position
- P&L calculation in real-time
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    symbol_id: int
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    day_pnl: float = 0.0
    # Option-specific
    is_option: bool = False
    underlying: Optional[str] = None
    underlying_price: Optional[float] = None
    strike: Optional[float] = None
    expiration: Optional[str] = None
    option_type: Optional[str] = None  # call/put
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    # Metadata
    account_id: Optional[str] = None
    last_updated: str = ""


class PositionManager:
    """
    Manages positions and provides real-time tracking.
    """
    
    def __init__(self, broker_client, trade_db=None, quote_client=None):
        """
        Initialize position manager.
        
        Args:
            broker_client: Broker client for order/position operations (IBKRAdapter or QuestradeClient)
            trade_db: Optional TradeDatabase for persistence
            quote_client: Separate client for real-time market data (e.g. Questrade).
                          If None, uses the main client for both.
        """
        self.client = broker_client
        self.quote_client = quote_client or broker_client
        self.db = trade_db
        self._positions: Dict[str, Position] = {}  # symbol -> Position
        self._account_id: Optional[str] = None
    
    def set_account(self, account_id: str):
        """Set the account to track"""
        self._account_id = account_id
        logger.info(f"Position manager set to account: {account_id}")
    
    def sync_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """
        Sync positions from Questrade.
        
        Args:
            account_id: Account to sync (uses default if not provided)
            
        Returns:
            List of current positions
        """
        account_id = account_id or self._account_id
        if not account_id:
            raise ValueError("No account ID set. Call set_account() first.")
        
        try:
            positions_data = self.client.get_account_positions(account_id)
            
            # PM-R1 fix: Build into a local dict first, then atomic-swap.
            # This avoids a window where self._positions is empty while
            # the loop is still running.
            new_positions = {}
            
            for p in positions_data:
                symbol = p.get('symbol', '')
                
                position = Position(
                    symbol=symbol,
                    symbol_id=p.get('symbolId', 0),
                    quantity=p.get('openQuantity', 0),
                    avg_cost=p.get('averageEntryPrice', 0),
                    current_price=p.get('currentPrice', 0),
                    market_value=p.get('currentMarketValue', 0),
                    unrealized_pnl=p.get('openPnl', 0),
                    unrealized_pnl_percent=p.get('openPnlPercent', 0),
                    day_pnl=p.get('dayPnl', 0),
                    account_id=account_id,
                    last_updated=datetime.now().isoformat()
                )
                
                # Check if it's an option
                if self._is_option_symbol(symbol):
                    position.is_option = True
                    parsed = self._parse_option_symbol(symbol)
                    if parsed:
                        position.underlying = parsed.get('underlying')
                        position.strike = parsed.get('strike')
                        position.expiration = parsed.get('expiration')
                        position.option_type = parsed.get('option_type')
                
                new_positions[symbol] = position
            
            self._positions = new_positions
            
            logger.info(f"Synced {len(self._positions)} positions")
            return list(self._positions.values())
            
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self._positions.values())
    
    def get_option_positions(self) -> List[Position]:
        """Get only option positions"""
        return [p for p in self._positions.values() if p.is_option]
    
    def get_stock_positions(self) -> List[Position]:
        """Get only stock positions"""
        return [p for p in self._positions.values() if not p.is_option]
    
    def get_positions_by_underlying(self, underlying: str) -> List[Position]:
        """Get all positions for an underlying"""
        return [
            p for p in self._positions.values()
            if p.underlying == underlying or p.symbol == underlying
        ]
    
    def update_quotes(self) -> List[Position]:
        """
        Update current prices for all positions.
        
        Returns:
            Updated positions
        """
        if not self._positions:
            return []
        
        # Get symbol IDs
        symbol_ids = [p.symbol_id for p in self._positions.values() if p.symbol_id]
        
        if not symbol_ids:
            return list(self._positions.values())
        
        try:
            quotes = self.quote_client.get_quotes(symbol_ids)
            
            # Map by symbol ID
            quote_map = {q.get('symbolId'): q for q in quotes}
            
            for pos in self._positions.values():
                quote = quote_map.get(pos.symbol_id)
                if quote:
                    pos.current_price = quote.get('lastTradePrice', pos.current_price)
                    pos.market_value = pos.current_price * pos.quantity
                    if pos.is_option:
                        pos.market_value *= 100  # Options multiplier
                    
                    # Recalculate P&L
                    if pos.avg_cost > 0:
                        cost_basis = pos.avg_cost * pos.quantity
                        if pos.is_option:
                            cost_basis *= 100
                        pos.unrealized_pnl = pos.market_value - cost_basis
                        pos.unrealized_pnl_percent = (pos.unrealized_pnl / cost_basis) * 100
                    
                    pos.last_updated = datetime.now().isoformat()
            
            logger.debug(f"Updated quotes for {len(quotes)} positions")
            
        except Exception as e:
            logger.error(f"Failed to update quotes: {e}")
        
        return list(self._positions.values())
    
    def update_option_greeks(self) -> List[Position]:
        """
        Update Greeks for option positions.
        
        Returns:
            Updated option positions
        """
        option_positions = self.get_option_positions()
        
        if not option_positions:
            return []
        
        for pos in option_positions:
            try:
                # Get option quote with Greeks
                if pos.symbol_id:
                    option_quotes = self.quote_client.get_option_quotes(option_ids=[pos.symbol_id])
                    
                    if option_quotes:
                        q = option_quotes[0]
                        pos.delta = q.get('delta')
                        pos.gamma = q.get('gamma')
                        pos.theta = q.get('theta')
                        pos.vega = q.get('vega')
                        pos.iv = q.get('volatility')
                        pos.current_price = q.get('lastTradePrice', pos.current_price)
                        pos.last_updated = datetime.now().isoformat()
                        
            except Exception as e:
                logger.warning(f"Failed to update Greeks for {pos.symbol}: {e}")
        
        return option_positions
    
    def get_total_exposure(self) -> Dict[str, float]:
        """
        Calculate total portfolio exposure.
        
        Returns:
            Dict with exposure metrics
        """
        total_value = sum(p.market_value for p in self._positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        
        # Delta exposure (options only)
        delta_exposure = 0
        for p in self.get_option_positions():
            if p.delta:
                delta_exposure += p.delta * p.quantity * 100
        
        return {
            "total_market_value": total_value,
            "total_unrealized_pnl": total_unrealized,
            "position_count": len(self._positions),
            "option_count": len(self.get_option_positions()),
            "stock_count": len(self.get_stock_positions()),
            "net_delta_exposure": delta_exposure
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk metrics for the portfolio.
        
        Returns:
            Dict with risk metrics
        """
        option_positions = self.get_option_positions()
        
        # Aggregate Greeks
        total_delta = sum((p.delta or 0) * p.quantity * 100 for p in option_positions)
        total_gamma = sum((p.gamma or 0) * p.quantity * 100 for p in option_positions)
        total_theta = sum((p.theta or 0) * p.quantity * 100 for p in option_positions)
        total_vega = sum((p.vega or 0) * p.quantity * 100 for p in option_positions)
        
        # Group by underlying
        by_underlying = {}
        for p in option_positions:
            ul = p.underlying or p.symbol
            if ul not in by_underlying:
                by_underlying[ul] = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "positions": 0}
            by_underlying[ul]["delta"] += (p.delta or 0) * p.quantity * 100
            by_underlying[ul]["gamma"] += (p.gamma or 0) * p.quantity * 100
            by_underlying[ul]["theta"] += (p.theta or 0) * p.quantity * 100
            by_underlying[ul]["vega"] += (p.vega or 0) * p.quantity * 100
            by_underlying[ul]["positions"] += 1
        
        return {
            "portfolio_delta": round(total_delta, 2),
            "portfolio_gamma": round(total_gamma, 4),
            "portfolio_theta": round(total_theta, 2),
            "portfolio_vega": round(total_vega, 2),
            "by_underlying": by_underlying
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        account_value: float,
        risk_percent: float = 2.0,
        stop_loss_percent: Optional[float] = None,
        max_contracts: int = 10
    ) -> int:
        """
        Calculate appropriate position size based on risk.
        
        Args:
            symbol: Symbol to trade
            account_value: Total account value
            risk_percent: Maximum risk per trade as % of account
            stop_loss_percent: Stop loss as % of entry price
            max_contracts: Maximum contracts allowed
            
        Returns:
            Recommended quantity
        """
        risk_amount = account_value * (risk_percent / 100)
        
        # Get current price
        quote = self.quote_client.get_quote_by_symbol(symbol)
        if not quote:
            return 1
        
        current_price = quote.get('lastTradePrice', 0)
        if current_price <= 0:
            return 1
        
        # If stop loss provided, calculate based on that
        if stop_loss_percent:
            loss_per_contract = current_price * (stop_loss_percent / 100) * 100
            quantity = int(risk_amount / loss_per_contract)
        else:
            # Default: risk the premium
            quantity = int(risk_amount / (current_price * 100))
        
        # Apply constraints
        quantity = max(1, min(quantity, max_contracts))
        
        return quantity
    
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an option"""
        import re
        # Questrade format: AAPL30Jan26C240.00
        if re.match(r'^[A-Z]+\d{2}[A-Za-z]{3}\d{2}[CP]\d+\.?\d*$', symbol):
            return True
        # OCC format: SPY20260310C680
        if re.match(r'^[A-Z]+\d{8}[CP]\d+\.?\d*$', symbol):
            return True
        return False
    
    def _parse_option_symbol(self, symbol: str) -> Optional[Dict]:
        """Parse option symbol (Questrade or OCC format)"""
        import re
        # OCC format: SPY20260310C680
        occ_match = re.match(r'^([A-Z]+)(\d{4})(\d{2})(\d{2})([CP])(\d+\.?\d*)$', symbol)
        if occ_match:
            underlying, year, month, day, opt_type, strike = occ_match.groups()
            return {
                "underlying": underlying,
                "expiration": f"{year}-{month}-{day}",
                "option_type": "call" if opt_type == 'C' else "put",
                "strike": float(strike)
            }
        # Questrade format: AAPL30Jan26C240.00
        qt_match = re.match(r'^([A-Z]+)(\d{2})([A-Za-z]{3})(\d{2})([CP])(\d+\.?\d*)$', symbol)
        if qt_match:
            underlying, day, month, year, opt_type, strike = qt_match.groups()
            months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                      'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            month_num = months.get(month, '01')
            return {
                "underlying": underlying,
                "expiration": f"20{year}-{month_num}-{day}",
                "option_type": "call" if opt_type == 'C' else "put",
                "strike": float(strike)
            }
        return None
    
    def print_positions(self):
        """Print positions summary"""
        positions = self.get_all_positions()
        
        if not positions:
            print("No positions")
            return
        
        print("\n" + "=" * 80)
        print("POSITIONS")
        print("=" * 80)
        print(f"{'Symbol':<25} {'Qty':>6} {'Avg Cost':>10} {'Current':>10} {'P&L':>12} {'%':>8}")
        print("-" * 80)
        
        for p in positions:
            pnl_str = f"${p.unrealized_pnl:,.2f}" if p.unrealized_pnl else "$0.00"
            pct_str = f"{p.unrealized_pnl_percent:.2f}%" if p.unrealized_pnl_percent else "0.00%"
            print(f"{p.symbol:<25} {p.quantity:>6} ${p.avg_cost:>9.2f} ${p.current_price:>9.2f} {pnl_str:>12} {pct_str:>8}")
        
        exposure = self.get_total_exposure()
        print("-" * 80)
        print(f"Total Value: ${exposure['total_market_value']:,.2f}  |  Unrealized P&L: ${exposure['total_unrealized_pnl']:,.2f}")
        print("=" * 80)
