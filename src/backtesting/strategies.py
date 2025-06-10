"""
FINAL FIXED strategies.py - Correcting Buy & Hold and Signal Logic
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
    """FIXED ML Trading Strategy with CORRECT signal logic"""
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        volume_threshold: float = 1.0,
        confidence_threshold: float = 0.25,
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
        Generate BALANCED trading signals - Both BUY and SELL
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
            
            # Get current market data
            current_price = current_data.get('close', 0)
            if current_price <= 0:
                return signal
            
            # RSI Analysis - BALANCED APPROACH
            rsi = current_data.get('rsi', 50)
            if pd.notna(rsi):
                # Strong signals at extremes
                if rsi <= 30:  # Very oversold = Strong BUY
                    buy_score += 5.0
                    reasoning.append(f'RSI very oversold ({rsi:.1f}) - Strong BUY')
                    
                elif rsi <= 40:  # Moderately oversold = BUY
                    buy_score += 3.0
                    reasoning.append(f'RSI oversold ({rsi:.1f}) - BUY signal')
                    
                elif rsi >= 70:  # Very overbought = Strong SELL  
                    sell_score += 5.0
                    reasoning.append(f'RSI very overbought ({rsi:.1f}) - Strong SELL')
                    
                elif rsi >= 60:  # Moderately overbought = SELL
                    sell_score += 3.0
                    reasoning.append(f'RSI overbought ({rsi:.1f}) - SELL signal')
                    
                # NEUTRAL ZONE - This is key for balance!
                elif 45 <= rsi <= 55:  # Neutral zone
                    # Look at price momentum for direction
                    if len(historical_data) >= 5:
                        recent_prices = historical_data['close'].tail(5)
                        price_change = (current_price - recent_prices.iloc[0]) / recent_prices.iloc[0]
                        
                        if price_change > 0.02:  # Rising > 2%
                            sell_score += 1.5
                            reasoning.append(f'Neutral RSI but rising momentum - SELL signal')
                        elif price_change < -0.02:  # Falling > 2%
                            buy_score += 1.5
                            reasoning.append(f'Neutral RSI but falling momentum - BUY signal')
                        else:
                            # Very neutral - look at RSI direction
                            if rsi < 50:
                                buy_score += 1.0
                                reasoning.append(f'Neutral RSI ({rsi:.1f}) slightly bearish - Weak BUY')
                            else:
                                sell_score += 1.0
                                reasoning.append(f'Neutral RSI ({rsi:.1f}) slightly bullish - Weak SELL')
                                
                elif 40 < rsi < 45:  # Leaning oversold
                    buy_score += 2.0
                    reasoning.append(f'RSI leaning oversold ({rsi:.1f}) - BUY signal')
                    
                elif 55 < rsi < 60:  # Leaning overbought
                    sell_score += 2.0
                    reasoning.append(f'RSI leaning overbought ({rsi:.1f}) - SELL signal')
            
            # MACD for additional confirmation
            macd = current_data.get('macd', 0)
            macd_signal = current_data.get('macd_signal', 0)
            if pd.notna(macd) and pd.notna(macd_signal):
                macd_diff = macd - macd_signal
                if macd_diff > 0.1:  # Strong bullish MACD
                    buy_score += 1.5
                    reasoning.append('Strong bullish MACD - BUY confirmation')
                elif macd_diff < -0.1:  # Strong bearish MACD
                    sell_score += 1.5
                    reasoning.append('Strong bearish MACD - SELL confirmation')
                elif macd_diff > 0:  # Weak bullish
                    buy_score += 0.5
                    reasoning.append('Weak bullish MACD')
                else:  # Weak bearish
                    sell_score += 0.5
                    reasoning.append('Weak bearish MACD')
            
            # Price trend analysis
            if len(historical_data) >= 10:
                sma_10 = historical_data['close'].tail(10).mean()
                if current_price > sma_10 * 1.02:  # 2% above SMA
                    sell_score += 1.0
                    reasoning.append('Price well above short-term average - SELL signal')
                elif current_price < sma_10 * 0.98:  # 2% below SMA
                    buy_score += 1.0
                    reasoning.append('Price well below short-term average - BUY signal')
            
            # Volume confirmation (but don't let it dominate)
            volume_ratio = current_data.get('volume_ratio', 1.0)
            if pd.notna(volume_ratio) and volume_ratio >= self.volume_threshold:
                if buy_score > sell_score:
                    buy_score += 0.5
                    reasoning.append(f'Volume confirms BUY')
                elif sell_score > buy_score:
                    sell_score += 0.5
                    reasoning.append(f'Volume confirms SELL')
            
            # CRITICAL: Ensure we can generate both BUY and SELL signals
            # by using proper scoring thresholds
            min_signal_score = 1.0  # Minimum score to generate signal
            
            if buy_score >= min_signal_score and buy_score > sell_score:
                confidence = min(buy_score / 5.0, 1.0)
                
                if confidence >= self.confidence_threshold:
                    signal['action'] = 'BUY'
                    signal['confidence'] = confidence
                    signal['reasoning'] = reasoning
                    signal['technical_score'] = buy_score
                    
                    logger.info(f"🟢 BUY signal generated at ${current_price:.2f} - "
                            f"Score: {buy_score:.1f}, Confidence: {confidence:.2f}")
            
            elif sell_score >= min_signal_score and sell_score > buy_score:
                confidence = min(sell_score / 5.0, 1.0)
                
                if confidence >= self.confidence_threshold:
                    signal['action'] = 'SELL'
                    signal['confidence'] = confidence
                    signal['reasoning'] = reasoning
                    signal['technical_score'] = sell_score
                    
                    logger.info(f"🔴 SELL signal generated at ${current_price:.2f} - "
                            f"Score: {sell_score:.1f}, Confidence: {confidence:.2f}")
            
            # DEBUG logging to understand balance
            else:
                logger.debug(f"HOLD at ${current_price:.2f} - BUY:{buy_score:.1f}, SELL:{sell_score:.1f}, RSI:{rsi:.1f}")
            
            self.signal_count += 1
            self.last_signal = signal['action']
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            signal['reasoning'] = [f'Error: {str(e)}']
        
        return signal


class TechnicalAnalysisStrategy(TradingStrategy):
    """FIXED Technical Analysis Strategy"""
    
    def __init__(
        self,
        sma_short: int = 20,
        sma_long: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position = None
        
        logger.info(f"TechnicalAnalysisStrategy initialized: RSI {rsi_oversold}/{rsi_overbought}")
    
    def reset(self):
        self.position = None
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        signal = {
            'action': 'HOLD',
            'confidence': 0.6,
            'symbol': 'STOCK',
            'reasoning': []
        }
        
        try:
            sma_short = current_data.get(f'sma_{self.sma_short}', 0)
            sma_long = current_data.get(f'sma_{self.sma_long}', 0)
            price = current_data.get('close', 0)
            rsi = current_data.get('rsi', 50)
            
            # Strong RSI signals first
            if rsi <= self.rsi_oversold:
                signal['action'] = 'BUY'
                signal['confidence'] = 0.9
                signal['reasoning'].append(f'RSI oversold ({rsi:.1f}) - Stock very cheap, strong BUY')
                logger.info(f"🟢 STRONG BUY: RSI {rsi:.1f} <= {self.rsi_oversold} at ${price:.2f}")
                
            elif rsi >= self.rsi_overbought:
                signal['action'] = 'SELL'
                signal['confidence'] = 0.9
                signal['reasoning'].append(f'RSI overbought ({rsi:.1f}) - Stock very expensive, strong SELL')
                logger.info(f"🔴 STRONG SELL: RSI {rsi:.1f} >= {self.rsi_overbought} at ${price:.2f}")
            
            # Moving Average signals
            elif pd.notna(sma_short) and pd.notna(sma_long) and sma_short > 0 and sma_long > 0:
                if price < sma_short < sma_long and rsi < 50:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.7
                    signal['reasoning'].append(f'Price (${price:.2f}) below SMAs, RSI {rsi:.1f} - Value opportunity')
                    logger.info(f"🟢 BUY: Price below SMAs, RSI {rsi:.1f}")
                    
                elif price > sma_short > sma_long and rsi > 50:
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.7
                    signal['reasoning'].append(f'Price (${price:.2f}) above SMAs, RSI {rsi:.1f} - Taking profits')
                    logger.info(f"🔴 SELL: Price above SMAs, RSI {rsi:.1f}")
            
            # Moderate RSI signals
            elif rsi < 40:
                signal['action'] = 'BUY'
                signal['confidence'] = 0.6
                signal['reasoning'].append(f'RSI {rsi:.1f} below 40 - Moderate buy signal')
                logger.info(f"🟢 BUY: RSI {rsi:.1f} < 40")
                
            elif rsi > 60:
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


"""
TRUE BUY & HOLD FIX - Replace the BuyAndHoldStrategy class in strategies.py

The problem: Buy & Hold was subject to stop losses and wrong entry timing.
The solution: Make it immune to all backtesting rules.
"""

class BuyAndHoldStrategy(TradingStrategy):
    """
    TRUE Buy and Hold Strategy - Immune to all backtesting rules
    
    This strategy should:
    1. Buy on the VERY FIRST day at opening price
    2. NEVER sell (ignore stop losses, take profits, etc.)
    3. Hold until the very end
    """
    
    def __init__(self):
        self.has_bought = False
        self.buy_price = 0
        self.is_first_call = True  # Track if this is the very first call
    
    def reset(self):
        """Reset strategy state"""
        self.has_bought = False
        self.buy_price = 0
        self.is_first_call = True
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """
        TRUE Buy and Hold Logic:
        - Buy ONLY on the first call ever
        - NEVER sell after that, no matter what
        """
        
        current_price = current_data.get('close', 0)
        
        # Buy ONLY on the very first call
        if self.is_first_call:
            self.is_first_call = False
            self.has_bought = True
            self.buy_price = current_price
            
            logger.info(f"🟢 TRUE BUY AND HOLD: Buying at ${current_price:.2f} on FIRST day")
            
            return {
                'action': 'BUY',
                'confidence': 1.0,
                'symbol': 'STOCK',
                'reasoning': [f'Buy and hold - purchasing at first data point: ${current_price:.2f}'],
                'ignore_stop_loss': True,  # Special flag to ignore risk management
                'ignore_take_profit': True,  # Special flag to ignore risk management
                'buy_and_hold': True  # Special flag for the backtester
            }
        
        # ALL subsequent calls: ALWAYS hold, never sell
        unrealized_return = ((current_price - self.buy_price) / self.buy_price * 100) if self.buy_price > 0 else 0
        
        return {
            'action': 'HOLD',  # NEVER sell
            'confidence': 1.0,
            'symbol': 'STOCK',
            'reasoning': [f'TRUE HOLD: Bought at ${self.buy_price:.2f}, now ${current_price:.2f} ({unrealized_return:+.1f}%)'],
            'ignore_stop_loss': True,  # Ignore stop losses
            'ignore_take_profit': True,  # Ignore take profits
            'buy_and_hold': True  # Flag for backtester
        }


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy using Bollinger Bands"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, hold_period: int = 5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.hold_period = hold_period
        self.position = None
        self.hold_counter = 0
    
    def reset(self):
        self.position = None
        self.hold_counter = 0
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
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
            
            if self.position:
                self.hold_counter += 1
            
            # Buy when very oversold
            if bb_position < 0.1 and rsi < 35:
                if not self.position:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('Mean reversion BUY: very oversold conditions')
                    self.position = 'LONG'
                    self.hold_counter = 0
                    logger.info(f"🟢 MEAN REVERSION BUY at ${price:.2f} (BB: {bb_position:.2f}, RSI: {rsi:.1f})")
            
            # Sell when very overbought
            elif bb_position > 0.9 and rsi > 65:
                if self.position == 'LONG':
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.8
                    signal['reasoning'].append('Mean reversion SELL: very overbought conditions')
                    self.position = None
                    self.hold_counter = 0
                    logger.info(f"🔴 MEAN REVERSION SELL at ${price:.2f} (BB: {bb_position:.2f}, RSI: {rsi:.1f})")
            
            # Exit conditions
            elif self.position and self.hold_counter >= self.hold_period:
                if self.position == 'LONG' and bb_position > 0.6:
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
    """Momentum strategy"""
    
    def __init__(self, lookback_period: int = 20, breakout_threshold: float = 0.02, volume_confirmation: bool = True):
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation = volume_confirmation
        self.position = None
    
    def reset(self):
        self.position = None
    
    def generate_signal(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
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
            
            recent_data = historical_data.tail(self.lookback_period)
            current_price = current_data['close']
            start_price = recent_data['close'].iloc[0]
            price_change = (current_price - start_price) / start_price
            
            volume_confirmed = True
            if self.volume_confirmation:
                avg_volume = recent_data['volume'].mean()
                current_volume = current_data.get('volume', avg_volume)
                volume_confirmed = current_volume > avg_volume * 1.2
            
            # Buy on upward breakouts
            if price_change > self.breakout_threshold and volume_confirmed:
                if not self.position:
                    signal['action'] = 'BUY'
                    signal['confidence'] = min(abs(price_change) * 10, 0.9)
                    signal['reasoning'].append(f'Upward breakout: {price_change:.2%} with volume')
                    self.position = 'LONG'
                    logger.info(f"🟢 MOMENTUM BUY: {price_change:.2%} breakout at ${current_price:.2f}")
            
            # Sell on downward breakouts
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