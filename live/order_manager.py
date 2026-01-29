"""
Order Manager - Handle order execution and tracking

Manages:
- Order submission to Questrade
- Order status monitoring
- Order cancellation
- Fill tracking
"""
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"


class OrderStatus(Enum):
    PENDING = "Pending"
    OPEN = "Open"
    FILLED = "Filled"
    PARTIAL = "PartiallyFilled"
    CANCELLED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    UNKNOWN = "Unknown"


class TimeInForce(Enum):
    DAY = "Day"
    GTC = "GoodTillCanceled"
    GTD = "GoodTillDate"
    FOK = "FillOrKill"
    IOC = "ImmediateOrCancel"


@dataclass
class Order:
    """Represents an order"""
    order_id: Optional[int] = None
    symbol: str = ""
    symbol_id: int = 0
    quantity: int = 0
    filled_quantity: int = 0
    side: str = ""
    order_type: str = ""
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: float = 0.0
    status: str = "Pending"
    time_in_force: str = "Day"
    account_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    commission: float = 0.0
    notes: str = ""


class OrderManager:
    """
    Manages order submission, tracking, and execution.
    """
    
    def __init__(self, questrade_client, trade_db=None):
        """
        Initialize order manager.
        
        Args:
            questrade_client: QuestradeClient instance
            trade_db: Optional TradeDatabase for persistence
        """
        self.client = questrade_client
        self.db = trade_db
        self._account_id: Optional[str] = None
        self._orders: Dict[int, Order] = {}  # order_id -> Order
        self._pending_orders: List[Order] = []
        self._fill_callbacks: List[Callable] = []
    
    def set_account(self, account_id: str):
        """Set the account for order operations"""
        self._account_id = account_id
        logger.info(f"Order manager set to account: {account_id}")
    
    def on_fill(self, callback: Callable[[Order], None]):
        """Register callback for fill events"""
        self._fill_callbacks.append(callback)
    
    def _notify_fill(self, order: Order):
        """Notify registered callbacks of a fill"""
        for callback in self._fill_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def submit_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        account_id: Optional[str] = None,
        is_all_or_none: bool = False,
        notes: str = ""
    ) -> Order:
        """
        Submit an order to Questrade.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares/contracts
            side: Buy or Sell
            order_type: Market, Limit, Stop, StopLimit
            limit_price: Limit price (required for Limit/StopLimit)
            stop_price: Stop price (required for Stop/StopLimit)
            time_in_force: Order duration
            account_id: Account to use (uses default if not provided)
            is_all_or_none: All or none execution
            notes: Notes for this order
            
        Returns:
            Order object with result
        """
        account_id = account_id or self._account_id
        if not account_id:
            raise ValueError("No account ID set. Call set_account() first.")
        
        # Get symbol ID
        symbol_id = self.client.get_symbol_id(symbol)
        if not symbol_id:
            raise ValueError(f"Symbol not found: {symbol}")
        
        # Create order object
        order = Order(
            symbol=symbol,
            symbol_id=symbol_id,
            quantity=quantity,
            side=side.value,
            order_type=order_type.value,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force.value,
            account_id=account_id,
            created_at=datetime.now().isoformat(),
            notes=notes
        )
        
        try:
            # Submit to Questrade
            result = self.client.place_order(
                account_id=account_id,
                symbol_id=symbol_id,
                quantity=quantity,
                is_buy=(side == OrderSide.BUY),
                order_type=order_type.value,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force.value,
                is_all_or_none=is_all_or_none
            )
            
            if result and 'orderId' in result:
                order.order_id = result['orderId']
                order.status = result.get('orderState', 'Pending')
                self._orders[order.order_id] = order
                logger.info(f"Order submitted: {order.order_id} - {side.value} {quantity} {symbol}")
                
                # Save to database
                if self.db:
                    self.db.insert_order(
                        order_id=str(order.order_id),
                        trade_id=None,  # Will be linked when filled
                        symbol=symbol,
                        side=side.value,
                        order_type=order_type.value,
                        quantity=quantity,
                        limit_price=limit_price,
                        status=order.status,
                        submitted_at=order.created_at
                    )
            else:
                order.status = "Rejected"
                logger.error(f"Order rejected: {result}")
                
        except Exception as e:
            order.status = "Rejected"
            order.notes = str(e)
            logger.error(f"Order submission failed: {e}")
            raise
        
        return order
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """
        Submit a buy order (convenience method).
        """
        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET
        return self.submit_order(
            symbol=symbol,
            quantity=quantity,
            side=OrderSide.BUY,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """
        Submit a sell order (convenience method).
        """
        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET
        return self.submit_order(
            symbol=symbol,
            quantity=quantity,
            side=OrderSide.SELL,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=time_in_force
        )
    
    def buy_to_open(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Order:
        """Buy to open an option position"""
        return self.buy(symbol, quantity, limit_price)
    
    def sell_to_close(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Order:
        """Sell to close an option position"""
        return self.sell(symbol, quantity, limit_price)
    
    def sell_to_open(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Order:
        """Sell to open (short) an option position"""
        return self.sell(symbol, quantity, limit_price)
    
    def buy_to_close(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Order:
        """Buy to close a short option position"""
        return self.buy(symbol, quantity, limit_price)
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled successfully
        """
        if not self._account_id:
            raise ValueError("No account ID set")
        
        try:
            result = self.client.cancel_order(self._account_id, order_id)
            
            if order_id in self._orders:
                self._orders[order_id].status = "Canceled"
                self._orders[order_id].updated_at = datetime.now().isoformat()
            
            logger.info(f"Order cancelled: {order_id}")
            
            # Update database
            if self.db:
                self.db.update_order_status(str(order_id), "Canceled")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        open_orders = self.get_open_orders()
        cancelled = 0
        
        for order in open_orders:
            if order.order_id and self.cancel_order(order.order_id):
                cancelled += 1
        
        return cancelled
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)
    
    def get_open_orders(self, account_id: Optional[str] = None) -> List[Order]:
        """
        Get all open orders from Questrade.
        
        Returns:
            List of open orders
        """
        account_id = account_id or self._account_id
        if not account_id:
            return []
        
        try:
            orders_data = self.client.get_account_orders(account_id)
            
            orders = []
            for o in orders_data:
                status = o.get('state', 'Unknown')
                if status in ['Pending', 'Accepted', 'Open', 'PartiallyFilled']:
                    order = Order(
                        order_id=o.get('id'),
                        symbol=o.get('symbol', ''),
                        symbol_id=o.get('symbolId', 0),
                        quantity=o.get('totalQuantity', 0),
                        filled_quantity=o.get('filledQuantity', 0),
                        side=o.get('side', ''),
                        order_type=o.get('orderType', ''),
                        limit_price=o.get('limitPrice'),
                        stop_price=o.get('stopPrice'),
                        avg_fill_price=o.get('avgExecPrice', 0),
                        status=status,
                        time_in_force=o.get('timeInForce', ''),
                        account_id=account_id,
                        created_at=o.get('creationTime', ''),
                        updated_at=o.get('updateTime', '')
                    )
                    orders.append(order)
                    self._orders[order.order_id] = order
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def sync_orders(self, account_id: Optional[str] = None) -> List[Order]:
        """
        Sync all orders (open and filled) from Questrade.
        
        Returns:
            List of all orders
        """
        account_id = account_id or self._account_id
        if not account_id:
            return []
        
        try:
            orders_data = self.client.get_account_orders(account_id)
            
            for o in orders_data:
                order_id = o.get('id')
                existing = self._orders.get(order_id)
                
                order = Order(
                    order_id=order_id,
                    symbol=o.get('symbol', ''),
                    symbol_id=o.get('symbolId', 0),
                    quantity=o.get('totalQuantity', 0),
                    filled_quantity=o.get('filledQuantity', 0),
                    side=o.get('side', ''),
                    order_type=o.get('orderType', ''),
                    limit_price=o.get('limitPrice'),
                    stop_price=o.get('stopPrice'),
                    avg_fill_price=o.get('avgExecPrice', 0),
                    status=o.get('state', 'Unknown'),
                    time_in_force=o.get('timeInForce', ''),
                    account_id=account_id,
                    created_at=o.get('creationTime', ''),
                    updated_at=o.get('updateTime', ''),
                    commission=o.get('commissionCharged', 0)
                )
                
                # Check for fill event
                if existing and existing.status != 'Filled' and order.status == 'Filled':
                    logger.info(f"Order filled: {order.order_id} - {order.side} {order.filled_quantity} {order.symbol} @ ${order.avg_fill_price}")
                    self._notify_fill(order)
                
                self._orders[order_id] = order
            
            return list(self._orders.values())
            
        except Exception as e:
            logger.error(f"Failed to sync orders: {e}")
            return []
    
    def wait_for_fill(self, order_id: int, timeout: int = 60, poll_interval: float = 1.0) -> Optional[Order]:
        """
        Wait for an order to be filled.
        
        Args:
            order_id: Order ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks
            
        Returns:
            Filled order or None if timeout/cancelled
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            self.sync_orders()
            order = self._orders.get(order_id)
            
            if order:
                if order.status == 'Filled':
                    return order
                elif order.status in ['Canceled', 'Rejected', 'Expired']:
                    logger.warning(f"Order {order_id} {order.status}")
                    return None
            
            time.sleep(poll_interval)
        
        logger.warning(f"Order {order_id} timed out after {timeout}s")
        return None
    
    def modify_order(
        self,
        order_id: int,
        new_quantity: Optional[int] = None,
        new_limit_price: Optional[float] = None
    ) -> bool:
        """
        Modify an existing order.
        
        Note: Questrade may require cancel/replace for modifications.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order {order_id} not found")
            return False
        
        # Cancel and replace
        if self.cancel_order(order_id):
            new_order = self.submit_order(
                symbol=order.symbol,
                quantity=new_quantity or order.quantity,
                side=OrderSide(order.side),
                order_type=OrderType(order.order_type),
                limit_price=new_limit_price or order.limit_price,
                time_in_force=TimeInForce(order.time_in_force)
            )
            return new_order.status not in ['Rejected']
        
        return False
    
    def print_orders(self):
        """Print orders summary"""
        self.sync_orders()
        
        open_orders = [o for o in self._orders.values() if o.status in ['Pending', 'Open', 'PartiallyFilled']]
        
        if not open_orders:
            print("No open orders")
            return
        
        print("\n" + "=" * 90)
        print("OPEN ORDERS")
        print("=" * 90)
        print(f"{'ID':<10} {'Symbol':<20} {'Side':<6} {'Qty':>6} {'Type':<8} {'Limit':>10} {'Status':<12}")
        print("-" * 90)
        
        for o in open_orders:
            limit_str = f"${o.limit_price:.2f}" if o.limit_price else "-"
            print(f"{o.order_id:<10} {o.symbol:<20} {o.side:<6} {o.quantity:>6} {o.order_type:<8} {limit_str:>10} {o.status:<12}")
        
        print("=" * 90)
