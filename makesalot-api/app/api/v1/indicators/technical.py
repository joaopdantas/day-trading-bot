"""
Technical Indicators for MakesALot Trading API
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Advanced technical indicators calculator"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = data.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(data), index=data.index)
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.fillna(0),
                'signal': signal_line.fillna(0),
                'histogram': histogram.fillna(0)
            }
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            zeros = pd.Series([0] * len(data), index=data.index)
            return {'macd': zeros, 'signal': zeros, 'histogram': zeros}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position within bands
            band_width = upper_band - lower_band
            position = (data - lower_band) / band_width
            position = position.fillna(0.5).clip(0, 1)
            
            return {
                'upper': upper_band.fillna(data),
                'middle': sma.fillna(data),
                'lower': lower_band.fillna(data),
                'position': position,
                'width': band_width.fillna(0)
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return {
                'upper': data,
                'middle': data,
                'lower': data,
                'position': pd.Series([0.5] * len(data), index=data.index),
                'width': pd.Series([0] * len(data), index=data.index)
            }
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            return data.rolling(window=period).mean().fillna(data)
        except Exception as e:
            logger.error(f"SMA calculation error: {e}")
            return data
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return data.ewm(span=period).mean().fillna(data)
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return data
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': k_percent.fillna(50),
                'd': d_percent.fillna(50)
            }
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            zeros = pd.Series([50] * len(close), index=close.index)
            return {'k': zeros, 'd': zeros}
    
    @staticmethod
    def calculate_volume_indicators(price: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        try:
            # Volume Moving Average
            volume_sma = volume.rolling(window=20).mean()
            
            # On-Balance Volume
            obv = pd.Series(index=price.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(price)):
                if price.iloc[i] > price.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif price.iloc[i] < price.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            # Volume Price Trend
            vpt = pd.Series(index=price.index, dtype=float)
            vpt.iloc[0] = 0
            
            for i in range(1, len(price)):
                if price.iloc[i-1] != 0:
                    vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * ((price.iloc[i] - price.iloc[i-1]) / price.iloc[i-1])
                else:
                    vpt.iloc[i] = vpt.iloc[i-1]
            
            return {
                'volume_sma': volume_sma.fillna(volume),
                'obv': obv.fillna(0),
                'vpt': vpt.fillna(0),
                'volume_ratio': (volume / volume_sma).fillna(1)
            }
        except Exception as e:
            logger.error(f"Volume indicators calculation error: {e}")
            zeros = pd.Series([0] * len(price), index=price.index)
            ones = pd.Series([1] * len(price), index=price.index)
            return {
                'volume_sma': volume,
                'obv': zeros,
                'vpt': zeros,
                'volume_ratio': ones
            }
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        try:
            if 'high' not in data.columns or 'low' not in data.columns:
                return {'support': [], 'resistance': []}
            
            highs = data['high'].rolling(window=window, center=True).max()
            lows = data['low'].rolling(window=window, center=True).min()
            
            # Find local maxima (resistance)
            resistance_levels = []
            for i in range(window, len(data) - window):
                if data['high'].iloc[i] == highs.iloc[i] and data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                    resistance_levels.append(float(data['high'].iloc[i]))
            
            # Find local minima (support)
            support_levels = []
            for i in range(window, len(data) - window):
                if data['low'].iloc[i] == lows.iloc[i] and data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                    support_levels.append(float(data['low'].iloc[i]))
            
            # Sort and remove duplicates
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)), reverse=True)[:5]
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
        except Exception as e:
            logger.error(f"Support/Resistance detection error: {e}")
            return {'support': [], 'resistance': []}
    
    @staticmethod
    def detect_rsi_divergences(data: pd.DataFrame, min_swing_pct: float = 2.5, 
                             max_lookback: int = 50) -> List[Dict]:
        """Detect RSI divergences"""
        try:
            if len(data) < max_lookback:
                return []
            
            divergences = []
            prices = data['close']
            rsi = TechnicalIndicators.calculate_rsi(prices)
            
            # Find swing highs and lows
            price_highs = []
            price_lows = []
            rsi_highs = []
            rsi_lows = []
            
            for i in range(5, len(data) - 5):
                # Price swing high
                if (prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1] and
                    prices.iloc[i] > prices.iloc[i-2] and prices.iloc[i] > prices.iloc[i+2]):
                    price_highs.append((i, prices.iloc[i]))
                    rsi_highs.append((i, rsi.iloc[i]))
                
                # Price swing low
                if (prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1] and
                    prices.iloc[i] < prices.iloc[i-2] and prices.iloc[i] < prices.iloc[i+2]):
                    price_lows.append((i, prices.iloc[i]))
                    rsi_lows.append((i, rsi.iloc[i]))
            
            # Check for bullish divergence (price lower low, RSI higher low)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                for i in range(1, len(price_lows)):
                    prev_price_low = price_lows[i-1][1]
                    curr_price_low = price_lows[i][1]
                    prev_rsi_low = rsi_lows[i-1][1]
                    curr_rsi_low = rsi_lows[i][1]
                    
                    price_change_pct = ((curr_price_low - prev_price_low) / prev_price_low) * 100
                    
                    if (curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low and 
                        abs(price_change_pct) >= min_swing_pct):
                        divergences.append({
                            'type': 'bullish',
                            'date': data.index[price_lows[i][0]],
                            'price': curr_price_low,
                            'rsi': curr_rsi_low,
                            'strength': abs(price_change_pct),
                            'confidence': min(0.9, 0.5 + abs(price_change_pct) / 10)
                        })
            
            # Check for bearish divergence (price higher high, RSI lower high)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                for i in range(1, len(price_highs)):
                    prev_price_high = price_highs[i-1][1]
                    curr_price_high = price_highs[i][1]
                    prev_rsi_high = rsi_highs[i-1][1]
                    curr_rsi_high = rsi_highs[i][1]
                    
                    price_change_pct = ((curr_price_high - prev_price_high) / prev_price_high) * 100
                    
                    if (curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high and 
                        price_change_pct >= min_swing_pct):
                        divergences.append({
                            'type': 'bearish',
                            'date': data.index[price_highs[i][0]],
                            'price': curr_price_high,
                            'rsi': curr_rsi_high,
                            'strength': price_change_pct,
                            'confidence': min(0.9, 0.5 + price_change_pct / 10)
                        })
            
            return sorted(divergences, key=lambda x: x['date'], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"RSI divergence detection error: {e}")
            return []
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for a dataset"""
        try:
            if data.empty or 'close' not in data.columns:
                logger.warning("Invalid data for technical indicators")
                return {}
            
            close = data['close']
            high = data.get('high', close)
            low = data.get('low', close)
            volume = data.get('volume', pd.Series([1000000] * len(close), index=close.index))
            
            # Calculate all indicators
            rsi = TechnicalIndicators.calculate_rsi(close)
            macd = TechnicalIndicators.calculate_macd(close)
            bb = TechnicalIndicators.calculate_bollinger_bands(close)
            stoch = TechnicalIndicators.calculate_stochastic(high, low, close)
            volume_ind = TechnicalIndicators.calculate_volume_indicators(close, volume)
            
            # Support/Resistance
            support_resistance = TechnicalIndicators.detect_support_resistance(data)
            
            # Divergences
            divergences = TechnicalIndicators.detect_rsi_divergences(data)
            
            # Get latest values
            latest_idx = -1
            
            return {
                'rsi': float(rsi.iloc[latest_idx]) if not rsi.empty else 50.0,
                'macd': float(macd['macd'].iloc[latest_idx]) if not macd['macd'].empty else 0.0,
                'macd_signal': float(macd['signal'].iloc[latest_idx]) if not macd['signal'].empty else 0.0,
                'macd_histogram': float(macd['histogram'].iloc[latest_idx]) if not macd['histogram'].empty else 0.0,
                'bb_upper': float(bb['upper'].iloc[latest_idx]) if not bb['upper'].empty else float(close.iloc[latest_idx]),
                'bb_middle': float(bb['middle'].iloc[latest_idx]) if not bb['middle'].empty else float(close.iloc[latest_idx]),
                'bb_lower': float(bb['lower'].iloc[latest_idx]) if not bb['lower'].empty else float(close.iloc[latest_idx]),
                'bb_position': float(bb['position'].iloc[latest_idx]) if not bb['position'].empty else 0.5,
                'stoch_k': float(stoch['k'].iloc[latest_idx]) if not stoch['k'].empty else 50.0,
                'stoch_d': float(stoch['d'].iloc[latest_idx]) if not stoch['d'].empty else 50.0,
                'sma_20': float(TechnicalIndicators.calculate_sma(close, 20).iloc[latest_idx]) if len(close) >= 20 else float(close.iloc[latest_idx]),
                'sma_50': float(TechnicalIndicators.calculate_sma(close, 50).iloc[latest_idx]) if len(close) >= 50 else float(close.iloc[latest_idx]),
                'ema_12': float(TechnicalIndicators.calculate_ema(close, 12).iloc[latest_idx]) if len(close) >= 12 else float(close.iloc[latest_idx]),
                'ema_26': float(TechnicalIndicators.calculate_ema(close, 26).iloc[latest_idx]) if len(close) >= 26 else float(close.iloc[latest_idx]),
                'volume_sma': float(volume_ind['volume_sma'].iloc[latest_idx]) if not volume_ind['volume_sma'].empty else float(volume.iloc[latest_idx]),
                'volume_ratio': float(volume_ind['volume_ratio'].iloc[latest_idx]) if not volume_ind['volume_ratio'].empty else 1.0,
                'obv': float(volume_ind['obv'].iloc[latest_idx]) if not volume_ind['obv'].empty else 0.0,
                'support_resistance': support_resistance,
                'divergences': divergences
            }
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            # Return safe defaults
            price = float(data['close'].iloc[-1]) if not data.empty and 'close' in data.columns else 100.0
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bb_upper': price * 1.02,
                'bb_middle': price,
                'bb_lower': price * 0.98,
                'bb_position': 0.5,
                'stoch_k': 50.0,
                'stoch_d': 50.0,
                'sma_20': price,
                'sma_50': price,
                'ema_12': price,
                'ema_26': price,
                'volume_sma': 1000000.0,
                'volume_ratio': 1.0,
                'obv': 0.0,
                'support_resistance': {'support': [], 'resistance': []},
                'divergences': []
            }