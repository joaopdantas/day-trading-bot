"""
FIXED strategies.py - Update existing file with correct logic
Keep the same class names to avoid import errors
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """Base class for trading strategies"""
    
    @abstractmethod
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Generate trading signal based on current and historical data"""
        pass
    
    def reset(self):
        """Reset strategy state for new backtest"""
        pass


class MLTradingStrategy(TradingStrategy):
    """
    FIXED ML Trading Strategy with CORRECT signal logic
    
    Original had inverted logic - this version implements proper buy low, sell high
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30,      # Buy when RSI < 30 (stock is CHEAP)
        rsi_overbought: float = 70,    # Sell when RSI > 70 (stock is EXPENSIVE)
        volume_threshold: float = 1.0, # Require average volume or higher
        confidence_threshold: float = 0.25,  # MUCH lower threshold
        use_ml_predictions: bool = False
    ):
        """Initialize with CORRECT parameters"""
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.confidence_threshold = confidence_threshold
        self.use_ml_predictions = use_ml_predictions
        
        # Strategy state
        self.last_signal = 'HOLD'
        self.position = None
        self.signal_count = 0
        
        # Debug logging
        logger.info(f"MLTradingStrategy initialized with:")
        logger.info(f"  RSI oversold: {rsi_oversold} (BUY when below)")
        logger.info(f"  RSI overbought: {rsi_overbought} (SELL when above)")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
    
    def reset(self):
        """Reset strategy state"""
        self.last_signal = 'HOLD'
        self.position = None
        self.signal_count = 0
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """
        Generate CORRECT trading signals - Buy LOW, Sell HIGH
        """
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'symbol': 'STOCK',
            'reasoning': [],
            'technical_score': 0.0
        }
        
        try:
            buy_score = 0.0
            sell_score = 0.0
            reasoning = []
            
            # Get current price for context
            current_price = current_data.get('close', 0)
            
            # RSI Analysis - FIXED LOGIC
            rsi = current_data.get('rsi', 50)
            if pd.notna(rsi):
                if rsi <= self.rsi_oversold:  # VERY CHEAP - Strong BUY
                    buy_score += 3.0
                    reasoning.append(f'RSI oversold ({rsi:.1f}) - Stock very cheap, strong BUY')
                elif rsi < 40:  # Somewhat cheap
                    buy_score += 1.5
                    reasoning.append(f'RSI below 40 ({rsi:.1f}) - Stock cheap, BUY signal')
                elif rsi >= self.rsi_overbought:  # VERY EXPENSIVE - Strong SELL
                    sell_score += 3.0
                    reasoning.append(f'RSI overbought ({rsi:.1f}) - Stock very expensive, strong SELL')
                elif rsi > 60:  # Somewhat expensive
                    sell_score += 1.5
                    reasoning.append(f'RSI above 60 ({rsi:.1f}) - Stock expensive, SELL signal')
            
            # MACD Analysis - FIXED LOGIC
            macd = current_data.get('macd', 0)
            macd_signal = current_data.get('macd_signal', 0)
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal:  # Bullish momentum
                    buy_score += 1.0
                    reasoning.append('MACD bullish - upward momentum')
                else:  # Bearish momentum
                    sell_score += 1.0
                    reasoning.append('MACD bearish - downward momentum')
            
            # Bollinger Bands - FIXED LOGIC
            bb_position = current_data.get('bb_position', 0.5)
            if pd.notna(bb_position):
                if bb_position <= 0.1:  # Near lower band = CHEAP
                    buy_score += 2.0
                    reasoning.append('Near lower Bollinger Band - very cheap')
                elif bb_position <= 0.3:  # Below middle = somewhat cheap
                    buy_score += 1.0
                    reasoning.append('Below Bollinger Band middle - cheap')
                elif bb_position >= 0.9:  # Near upper band = EXPENSIVE
                    sell_score += 2.0
                    reasoning.append('Near upper Bollinger Band - very expensive')
                elif bb_position >= 0.7:  # Above middle = somewhat expensive
                    sell_score += 1.0
                    reasoning.append('Above Bollinger Band middle - expensive')
            
            # Support/Resistance - FIXED LOGIC
            at_support = current_data.get('at_support', 0)
            at_resistance = current_data.get('at_resistance', 0)
            if at_support:  # At support = BUY opportunity
                buy_score += 2.0
                reasoning.append('At support level - BUY opportunity')
            if at_resistance:  # At resistance = SELL opportunity
                sell_score += 2.0
                reasoning.append('At resistance level - SELL opportunity')
            
            # Moving Average Position - FIXED LOGIC
            sma_20 = current_data.get('sma_20', 0)
            sma_50 = current_data.get('sma_50', 0)
            if pd.notna(sma_20) and pd.notna(sma_50) and current_price > 0:
                if current_price < sma_20 and sma_20 < sma_50:  # Price below both = CHEAP
                    buy_score += 1.5
                    reasoning.append('Price below moving averages - undervalued')
                elif current_price > sma_20 and sma_20 > sma_50:  # Price above both = EXPENSIVE
                    sell_score += 1.5
                    reasoning.append('Price above moving averages - overvalued')
            
            # Volume Confirmation
            volume_ratio = current_data.get('volume_ratio', 1.0)
            if pd.notna(volume_ratio) and volume_ratio >= self.volume_threshold:
                # Volume confirms the signal
                if buy_score > sell_score:
                    buy_score += 0.5
                    reasoning.append(f'Volume confirms BUY ({volume_ratio:.1f}x avg)')
                elif sell_score > buy_score:
                    sell_score += 0.5
                    reasoning.append(f'Volume confirms SELL ({volume_ratio:.1f}x avg)')
            
            # Calculate final signal
            if buy_score > sell_score and buy_score >= self.confidence_threshold:
                confidence = min(buy_score / 5.0, 1.0)
                signal['action'] = 'BUY'
                signal['confidence'] = confidence
                signal['reasoning'] = reasoning
                signal['technical_score'] = buy_score
                
                logger.info(f"🟢 BUY signal generated at ${current_price:.2f} - "
                           f"Score: {buy_score:.1f}, Confidence: {confidence:.2f}")
                
            elif sell_score > buy_score and sell_score >= self.confidence_threshold:
                confidence = min(sell_score / 5.0, 1.0)
                signal['action'] = 'SELL'
                signal['confidence'] = confidence
                signal['reasoning'] = reasoning
                signal['technical_score'] = sell_score
                
                logger.info(f"🔴 SELL signal generated at ${current_price:.2f} - "
                           f"Score: {sell_score:.1f}, Confidence: {confidence:.2f}")
            
            # Update state
            self.signal_count += 1
            self.last_signal = signal['action']
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            signal['reasoning'] = [f'Error: {str(e)}']
        
        return signal


class TechnicalAnalysisStrategy(TradingStrategy):
    """
    FIXED Technical Analysis Strategy - Proper buy low, sell high logic
    """
    
    def __init__(
        self,
        sma_short: int = 20,
        sma_long: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ):
        """Initialize with correct parameters"""
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        self.position = None
        logger.info(f"TechnicalAnalysisStrategy initialized: RSI {rsi_oversold}/{rsi_overbought}")
    
    def reset(self):
        """Reset strategy state"""
        self.position = None
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Generate CORRECT technical signals"""
        
        signal = {
            'action': 'HOLD',
            'confidence': 0.6,
            'symbol': 'STOCK',
            'reasoning': []
        }
        
        try:
            # Get current values
            sma_short = current_data.get(f'sma_{self.sma_short}', 0)
            sma_long = current_data.get(f'sma_{self.sma_long}', 0)
            price = current_data.get('close', 0)
            rsi = current_data.get('rsi', 50)
            
            # FIXED LOGIC: Strong RSI signals first
            if rsi <= self.rsi_oversold:  # Very cheap - STRONG BUY
                signal['action'] = 'BUY'
                signal['confidence'] = 0.9
                signal['reasoning'].append(f'RSI oversold ({rsi:.1f}) - Stock very cheap, strong BUY')
                logger.info(f"🟢 STRONG BUY: RSI {rsi:.1f} <= {self.rsi_oversold} at ${price:.2f}")
                
            elif rsi >= self.rsi_overbought:  # Very expensive - STRONG SELL
                signal['action'] = 'SELL'
                signal['confidence'] = 0.9
                signal['reasoning'].append(f'RSI overbought ({rsi:.1f}) - Stock very expensive, strong SELL')
                logger.info(f"🔴 STRONG SELL: RSI {rsi:.1f} >= {self.rsi_overbought} at ${price:.2f}")
            
            # Moving Average signals - FIXED LOGIC
            elif pd.notna(sma_short) and pd.notna(sma_long) and sma_short > 0 and sma_long > 0:
                
                if price < sma_short < sma_long and rsi < 50:  # Price below MAs + not overbought = CHEAP
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.7
                    signal['reasoning'].append(f'Price (${price:.2f}) below SMAs, RSI {rsi:.1f} - Value opportunity')
                    logger.info(f"🟢 BUY: Price below SMAs, RSI {rsi:.1f}")
                    
                elif price > sma_short > sma_long and rsi > 50:  # Price above MAs + not oversold = EXPENSIVE
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.7
                    signal['reasoning'].append(f'Price (${price:.2f}) above SMAs, RSI {rsi:.1f} - Taking profits')
                    logger.info(f"🔴 SELL: Price above SMAs, RSI {rsi:.1f}")
            
            # Moderate RSI signals
            elif rsi < 40:  # Somewhat cheap
                signal['action'] = 'BUY'
                signal['confidence'] = 0.6
                signal['reasoning'].append(f'RSI {rsi:.1f} below 40 - Moderate buy signal')
                logger.info(f"🟢 BUY: RSI {rsi:.1f} < 40")
                
            elif rsi > 60:  # Somewhat expensive
                signal['action'] = 'SELL'
                signal['confidence'] = 0.6
                signal['reasoning'].append(f'RSI {rsi:.1f} above 60 - Moderate sell signal')
                logger.info(f"🔴 SELL: RSI {rsi:.1f} > 60")
            
            if not signal['reasoning']:
                signal['reasoning'].append('No clear technical signal - market neutral')
        
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            signal['reasoning'].append(f'Error: {str(e)}')
        
        return signal


class BuyAndHoldStrategy(TradingStrategy):
    """
    FIXED Buy and Hold Strategy - Buy once and hold forever
    """
    
    def __init__(self):
        """Initialize strategy"""
        self.has_bought = False
        self.buy_price = 0
    
    def reset(self):
        """Reset strategy state"""
        self.has_bought = False
        self.buy_price = 0
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Buy once at the beginning and hold"""
        
        current_price = current_data.get('close', 0)
        
        if not self.has_bought:
            self.has_bought = True
            self.buy_price = current_price
            logger.info(f"🟢 BUY AND HOLD: Initial purchase at ${current_price:.2f}")
            
            return {
                'action': 'BUY',
                'confidence': 1.0,
                'symbol': 'STOCK',
                'reasoning': [f'Buy and hold - initial purchase at ${current_price:.2f}']
            }
        
        # Calculate unrealized return
        unrealized_return = ((current_price - self.buy_price) / self.buy_price * 100) if self.buy_price > 0 else 0
        
        return {
            'action': 'HOLD',
            'confidence': 1.0,
            'symbol': 'STOCK',
            'reasoning': [f'Holding since ${self.buy_price:.2f}, current: ${current_price:.2f} ({unrealized_return:+.1f}%)']
        }


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy using Bollinger Bands - FIXED"""
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        hold_period: int = 5
    ):
        """Initialize mean reversion strategy"""
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.hold_period = hold_period
        
        self.position = None
        self.hold_counter = 0
    
    def reset(self):
        """Reset strategy state"""
        self.position = None
        self.hold_counter = 0
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Generate mean reversion signals - FIXED LOGIC"""
        
        signal = {
            'action': 'HOLD',
            'confidence': 0.5,
            'symbol': 'STOCK',
            'reasoning': []
        }
        
        try:
            bb_position = current_data.get('bb_position', 0.5)
            rsi = current_data.get('rsi', 50)
            price = current_data.get('close', 0)
            
            # Update hold counter
            if self.position:
                self.hold_counter += 1
            
            # FIXED: Buy when cheap (low BB position + low RSI)
            if bb_position < 0.1 and rsi < 35:  # Very oversold
                if not self.position:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('Mean reversion BUY: very oversold conditions')
                    self.position = 'LONG'
                    self.hold_counter = 0
                    logger.info(f"🟢 MEAN REVERSION BUY at ${price:.2f} (BB: {bb_position:.2f}, RSI: {rsi:.1f})")
            
            # FIXED: Sell when expensive (high BB position + high RSI)
            elif bb_position > 0.9 and rsi > 65:  # Very overbought
                if self.position == 'LONG':
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('Mean reversion SELL: very overbought conditions')
                    self.position = None
                    self.hold_counter = 0
                    logger.info(f"🔴 MEAN REVERSION SELL at ${price:.2f} (BB: {bb_position:.2f}, RSI: {rsi:.1f})")
            
            # Exit conditions - take profits when moving back to middle
            elif self.position and self.hold_counter >= self.hold_period:
                if self.position == 'LONG' and bb_position > 0.6:  # Moved up from oversold
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.7
                    signal['reasoning'].append('Mean reversion exit: moved away from oversold')
                    self.position = None
                    self.hold_counter = 0
                    logger.info(f"🔴 MEAN REVERSION EXIT at ${price:.2f}")
            
            if not signal['reasoning']:
                signal['reasoning'].append('Mean reversion: waiting for extreme conditions')
        
        except Exception as e:
            logger.error(f"Mean reversion strategy error: {e}")
            signal['reasoning'].append(f'Error: {str(e)}')
        
        return signal


class MomentumStrategy(TradingStrategy):
    """Momentum strategy - FIXED"""
    
    def __init__(
        self,
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,
        volume_confirmation: bool = True
    ):
        """Initialize momentum strategy"""
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation = volume_confirmation
        
        self.position = None
    
    def reset(self):
        """Reset strategy state"""
        self.position = None
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """Generate momentum signals - FIXED"""
        
        signal = {
            'action': 'HOLD',
            'confidence': 0.5,
            'symbol': 'STOCK',
            'reasoning': []
        }
        
        try:
            if len(historical_data) < self.lookback_period:
                signal['reasoning'].append('Insufficient data for momentum')
                return signal
            
            # Calculate momentum
            recent_data = historical_data.tail(self.lookback_period)
            current_price = current_data['close']
            start_price = recent_data['close'].iloc[0]
            price_change = (current_price - start_price) / start_price
            
            # Volume confirmation
            volume_confirmed = True
            if self.volume_confirmation:
                avg_volume = recent_data['volume'].mean()
                current_volume = current_data.get('volume', avg_volume)
                volume_confirmed = current_volume > avg_volume * 1.2
            
            # FIXED: Buy on upward breakouts, sell on downward breakouts
            if price_change > self.breakout_threshold and volume_confirmed:
                if not self.position:
                    signal['action'] = 'BUY'
                    signal['confidence'] = min(abs(price_change) * 10, 0.9)
                    signal['reasoning'].append(f'Upward breakout: {price_change:.2%} with volume')
                    self.position = 'LONG'
                    logger.info(f"🟢 MOMENTUM BUY: {price_change:.2%} breakout at ${current_price:.2f}")
            
            elif price_change < -self.breakout_threshold and volume_confirmed:
                if self.position == 'LONG':
                    signal['action'] = 'SELL'
                    signal['confidence'] = min(abs(price_change) * 10, 0.9)
                    signal['reasoning'].append(f'Downward breakout: {price_change:.2%} with volume')
                    self.position = None
                    logger.info(f"🔴 MOMENTUM SELL: {price_change:.2%} breakdown at ${current_price:.2f}")
            
            if not signal['reasoning']:
                signal['reasoning'].append('Momentum: no clear breakout signal')
        
        except Exception as e:
            logger.error(f"Momentum strategy error: {e}")
            signal['reasoning'].append(f'Error: {str(e)}')
        
        return signal
    