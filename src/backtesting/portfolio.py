"""
COMPLETELY FIXED Portfolio Management for Backtesting

This replaces the broken src/backtesting/portfolio.py with correct calculations.
"""

import logging
from typing import Dict, Optional, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class Portfolio:
    """
    FIXED Portfolio management class for backtesting.
    
    The original had multiple calculation errors. This version provides:
    - Correct position averaging when buying multiple times
    - Proper total value calculations
    - Accurate transaction cost handling
    - Realized vs unrealized P&L tracking
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio with starting capital.
        
        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'total_cost': float}}
        self.transaction_history = []
        self.realized_pnl = 0.0  # Track realized gains/losses
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
        
    def reset(self, initial_capital: float):
        """Reset portfolio to initial state"""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions.clear()
        self.transaction_history.clear()
        self.realized_pnl = 0.0
        
    def buy_stock(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        transaction_cost_rate: float = 0.001
    ) -> bool:
        """
        Buy shares of a stock with FIXED calculations.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to buy
            price: Price per share
            transaction_cost_rate: Transaction cost as percentage of trade value
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        if shares <= 0:
            return False
            
        trade_value = shares * price
        transaction_cost = trade_value * transaction_cost_rate
        total_cost = trade_value + transaction_cost
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient funds for {symbol}: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # FIXED: Correct position averaging
        if symbol in self.positions:
            # Average down/up calculation
            old_shares = self.positions[symbol]['shares']
            old_total_value = old_shares * self.positions[symbol]['avg_price']
            old_total_cost = self.positions[symbol]['total_cost']
            
            new_total_shares = old_shares + shares
            new_total_value = old_total_value + trade_value
            new_avg_price = new_total_value / new_total_shares
            new_total_cost = old_total_cost + total_cost
            
            self.positions[symbol] = {
                'shares': new_total_shares,
                'avg_price': new_avg_price,
                'total_cost': new_total_cost,
                'purchase_date': self.positions[symbol].get('purchase_date', datetime.now())
            }
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'total_cost': total_cost,
                'purchase_date': datetime.now()
            }
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'action': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'cash_after': self.cash
        })
        
        logger.info(f"✅ BOUGHT {shares} shares of {symbol} at ${price:.2f} (cost: ${transaction_cost:.2f})")
        return True
    
    def sell_stock(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        transaction_cost_rate: float = 0.001
    ) -> bool:
        """
        Sell shares of a stock with FIXED P&L calculations.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to sell
            price: Price per share
            transaction_cost_rate: Transaction cost as percentage of trade value
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to sell")
            return False
        
        available_shares = self.positions[symbol]['shares']
        if shares > available_shares:
            logger.warning(f"Cannot sell {shares} shares of {symbol}, only have {available_shares}")
            return False
        
        trade_value = shares * price
        transaction_cost = trade_value * transaction_cost_rate
        net_proceeds = trade_value - transaction_cost
        
        # FIXED: Calculate realized P&L properly
        avg_cost_per_share = self.positions[symbol]['avg_price']
        cost_basis = shares * avg_cost_per_share
        realized_gain_loss = trade_value - cost_basis - transaction_cost
        self.realized_pnl += realized_gain_loss
        
        # Update cash
        self.cash += net_proceeds
        
        # FIXED: Update position correctly
        self.positions[symbol]['shares'] -= shares
        
        # If selling all shares, remove position
        if self.positions[symbol]['shares'] == 0:
            del self.positions[symbol]
        else:
            # Proportionally reduce total_cost
            remaining_ratio = self.positions[symbol]['shares'] / (self.positions[symbol]['shares'] + shares)
            self.positions[symbol]['total_cost'] *= remaining_ratio
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'action': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'transaction_cost': transaction_cost,
            'net_proceeds': net_proceeds,
            'realized_pnl': realized_gain_loss,
            'cash_after': self.cash
        })
        
        logger.info(f"✅ SOLD {shares} shares of {symbol} at ${price:.2f} "
                   f"(cost: ${transaction_cost:.2f}, P&L: ${realized_gain_loss:.2f})")
        return True
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current market value of a position"""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol]['shares'] * current_price
    
    def get_total_positions_value(self, current_prices: Dict[str, float]) -> float:
        """FIXED: Get total value of all positions"""
        total_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                market_value = position['shares'] * current_prices[symbol]
            else:
                # Fallback to average price if current price not available
                market_value = position['shares'] * position['avg_price']
            total_value += market_value
        return total_value
    
    def get_total_value(self, current_price_or_prices: Union[float, Dict[str, float]], symbol: str = 'STOCK') -> float:
        """
        FIXED: Get total portfolio value (cash + positions).
        
        Args:
            current_price_or_prices: Either a single price (float) for the main symbol,
                                   or a dictionary of {symbol: price}
            symbol: Main symbol to value (default 'STOCK')
            
        Returns:
            Total portfolio value
        """
        positions_value = 0.0
        
        # Handle both single price and price dictionary
        if isinstance(current_price_or_prices, dict):
            # Multiple prices provided
            for sym, position in self.positions.items():
                if sym in current_price_or_prices:
                    market_value = position['shares'] * current_price_or_prices[sym]
                else:
                    # Fallback to average price
                    market_value = position['shares'] * position['avg_price']
                positions_value += market_value
        else:
            # Single price provided - use for main symbol
            current_price = current_price_or_prices
            for sym, position in self.positions.items():
                if sym == symbol:
                    market_value = position['shares'] * current_price
                else:
                    # Use average price for other symbols
                    market_value = position['shares'] * position['avg_price']
                positions_value += market_value
        
        return self.cash + positions_value
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """FIXED: Get unrealized P&L for all positions"""
        pnl = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_market_value = position['shares'] * current_prices[symbol]
                cost_basis = position['shares'] * position['avg_price']
                pnl[symbol] = current_market_value - cost_basis
            else:
                pnl[symbol] = 0.0  # No current price available
        return pnl
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """FIXED: Get comprehensive portfolio summary"""
        if current_prices is None:
            current_prices = {symbol: pos['avg_price'] for symbol, pos in self.positions.items()}
        
        positions_value = self.get_total_positions_value(current_prices)
        total_value = self.cash + positions_value
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'realized_pnl': self.realized_pnl,
            'positions_count': len(self.positions),
            'positions': dict(self.positions),
            'unrealized_pnl': self.get_unrealized_pnl(current_prices)
        }
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific position"""
        return self.positions.get(symbol, None)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has a position in the symbol"""
        return symbol in self.positions and self.positions[symbol]['shares'] > 0
    
    def get_available_cash(self) -> float:
        """Get available cash for trading"""
        return self.cash
    
    def calculate_position_size(
        self, 
        price: float, 
        max_position_pct: float = 0.2, 
        confidence: float = 1.0
    ) -> int:
        """
        FIXED: Calculate optimal position size based on portfolio constraints.
        
        Args:
            price: Price per share
            max_position_pct: Maximum percentage of portfolio to risk
            confidence: Confidence in the trade (0-1)
            
        Returns:
            Number of shares to buy
        """
        total_value = self.get_total_value(price, 'STOCK')
        max_investment = total_value * max_position_pct * confidence
        
        # Account for transaction costs and ensure we have enough cash
        effective_price = price * (1 + 0.001)  # Assume 0.1% transaction cost
        max_shares_by_portfolio = int(max_investment / effective_price)
        max_shares_by_cash = int(self.cash / effective_price)
        
        return min(max_shares_by_portfolio, max_shares_by_cash)
    
    def get_trade_statistics(self) -> Dict:
        """Get trading statistics from transaction history"""
        if not self.transaction_history:
            return {'total_trades': 0, 'total_costs': 0, 'realized_pnl': self.realized_pnl}
        
        total_costs = sum(t.get('transaction_cost', 0) for t in self.transaction_history)
        buy_trades = len([t for t in self.transaction_history if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.transaction_history if t['action'] == 'SELL'])
        
        return {
            'total_trades': len(self.transaction_history),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_costs': total_costs,
            'realized_pnl': self.realized_pnl
        }
    