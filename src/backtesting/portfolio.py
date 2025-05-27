"""
Portfolio Management for Backtesting

Handles position tracking, cash management, and portfolio valuation
with realistic transaction costs and constraints.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio management class for backtesting.
    
    Tracks cash, positions, and portfolio value with realistic constraints.
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio with starting capital.
        
        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'purchase_date': datetime}}
        self.transaction_history = []
        
    def reset(self, initial_capital: float):
        """Reset portfolio to initial state"""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions.clear()
        self.transaction_history.clear()
        
    def buy_stock(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        transaction_cost_rate: float = 0.001
    ) -> bool:
        """
        Buy shares of a stock.
        
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
        
        # Update positions
        if symbol in self.positions:
            # Calculate new average price
            old_shares = self.positions[symbol]['shares']
            old_value = old_shares * self.positions[symbol]['avg_price']
            new_total_shares = old_shares + shares
            new_avg_price = (old_value + trade_value) / new_total_shares
            
            self.positions[symbol]['shares'] = new_total_shares
            self.positions[symbol]['avg_price'] = new_avg_price
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
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
            'cash_after': self.cash
        })
        
        logger.info(f"Bought {shares} shares of {symbol} at ${price:.2f} (cost: ${transaction_cost:.2f})")
        return True
    
    def sell_stock(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        transaction_cost_rate: float = 0.001
    ) -> bool:
        """
        Sell shares of a stock.
        
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
        
        # Update cash
        self.cash += net_proceeds
        
        # Update positions
        self.positions[symbol]['shares'] -= shares
        if self.positions[symbol]['shares'] == 0:
            del self.positions[symbol]
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'action': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'transaction_cost': transaction_cost,
            'cash_after': self.cash
        })
        
        logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f} (cost: ${transaction_cost:.2f})")
        return True
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position"""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol]['shares'] * current_price
    
    def get_total_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Get total value of all positions"""
        total_value = 0.0
        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position['avg_price'])
            total_value += position['shares'] * price
        return total_value
    
    def get_total_value(self, current_price: float, symbol: str = 'STOCK') -> float:
        """
        Get total portfolio value (cash + positions).
        
        Args:
            current_price: Current price of the main symbol
            symbol: Symbol to value (default 'STOCK')
            
        Returns:
            Total portfolio value
        """
        positions_value = 0.0
        for sym, position in self.positions.items():
            price = current_price if sym == symbol else position['avg_price']
            positions_value += position['shares'] * price
        
        return self.cash + positions_value
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get unrealized P&L for all positions"""
        pnl = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position['shares'] * current_prices[symbol]
                cost_basis = position['shares'] * position['avg_price']
                pnl[symbol] = current_value - cost_basis
        return pnl
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get comprehensive portfolio summary"""
        if current_prices is None:
            current_prices = {symbol: pos['avg_price'] for symbol, pos in self.positions.items()}
        
        positions_value = self.get_total_positions_value(current_prices)
        total_value = self.cash + positions_value
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'initial_capital': self.initial_capital,
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'positions_count': len(self.positions),
            'positions': dict(self.positions),
            'unrealized_pnl': self.get_unrealized_pnl(current_prices)
        }
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol].copy()
        return position
    
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
        Calculate optimal position size based on portfolio constraints.
        
        Args:
            price: Price per share
            max_position_pct: Maximum percentage of portfolio to risk
            confidence: Confidence in the trade (0-1)
            
        Returns:
            Number of shares to buy
        """
        total_value = self.get_total_value(price)
        max_investment = total_value * max_position_pct * confidence
        max_shares_by_portfolio = int(max_investment / price)
        max_shares_by_cash = int(self.cash / (price * 1.001))  # Account for transaction costs
        
        return min(max_shares_by_portfolio, max_shares_by_cash, max_shares_by_cash)