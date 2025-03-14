"""
Technical Indicators Module for Day Trading Bot.

This module calculates various technical indicators used for trading decisions.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class for calculating technical indicators on market data."""
    
    @staticmethod
    def add_moving_averages(
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20, 50, 200],
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add simple moving averages for specified periods.
        
        Args:
            df: DataFrame with market data
            periods: List of periods for SMA calculation
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with added moving average columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot calculate SMA: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        for period in periods:
            result_df[f'sma_{period}'] = ta.sma(result_df[price_column], length=period)
        
        return result_df
    
    @staticmethod
    def add_exponential_moving_averages(
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20, 50, 200],
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add exponential moving averages for specified periods.
        
        Args:
            df: DataFrame with market data
            periods: List of periods for EMA calculation
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with added EMA columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot calculate EMA: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        for period in periods:
            result_df[f'ema_{period}'] = ta.ema(result_df[price_column], length=period)
        
        return result_df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Args:
            df: DataFrame with market data
            period: Period for moving average calculation
            std_dev: Number of standard deviations for bands
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot calculate Bollinger Bands: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Calculate Bollinger Bands
        bbands = ta.bbands(result_df[price_column], length=period, std=std_dev)
        
        # Add the columns to the DataFrame
        result_df['bb_upper'] = bbands['BBU_'+str(period)+'_'+str(std_dev)]
        result_df['bb_middle'] = bbands['BBM_'+str(period)+'_'+str(std_dev)]
        result_df['bb_lower'] = bbands['BBL_'+str(period)+'_'+str(std_dev)]
        
        # Calculate bandwidth and percent B
        result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_percent'] = (result_df[price_column] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        return result_df
    
    @staticmethod
    def add_rsi(
        df: pd.DataFrame,
        period: int = 14,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with market data
            period: Period for RSI calculation
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with added RSI column
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot calculate RSI: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Calculate RSI
        result_df['rsi'] = ta.rsi(result_df[price_column], length=period)
        
        # Add RSI categories for easier interpretation
        result_df['rsi_oversold'] = (result_df['rsi'] < 30).astype(int)
        result_df['rsi_overbought'] = (result_df['rsi'] > 70).astype(int)
        
        return result_df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame with market data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with added MACD columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot calculate MACD: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Calculate MACD
        macd = ta.macd(result_df[price_column], fast=fast_period, slow=slow_period, signal=signal_period)
        
        # Add the columns to the DataFrame
        result_df['macd'] = macd[f'MACD_{fast_period}_{slow_period}_{signal_period}']
        result_df['macd_signal'] = macd[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
        result_df['macd_histogram'] = macd[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
        
        # Add MACD signal
        result_df['macd_bullish'] = (result_df['macd'] > result_df['macd_signal']).astype(int)
        result_df['macd_bearish'] = (result_df['macd'] < result_df['macd_signal']).astype(int)
        
        return result_df
    
    @staticmethod
    def add_stochastic_oscillator(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        
        Args:
            df: DataFrame with market data
            k_period: Period for %K
            d_period: Period for %D
            smooth_k: Smoothing factor for %K
            
        Returns:
            DataFrame with added Stochastic Oscillator columns
        """
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("Cannot calculate Stochastic Oscillator: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        # Calculate Stochastic Oscillator
        stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'], 
                        k=k_period, d=d_period, smooth_k=smooth_k)
        
        # Add the columns to the DataFrame
        result_df['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{smooth_k}']
        result_df['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{smooth_k}']
        
        # Add Stochastic signals
        result_df['stoch_oversold'] = ((result_df['stoch_k'] < 20) & (result_df['stoch_d'] < 20)).astype(int)
        result_df['stoch_overbought'] = ((result_df['stoch_k'] > 80) & (result_df['stoch_d'] > 80)).astype(int)
        result_df['stoch_bullish'] = ((result_df['stoch_k'] > result_df['stoch_d']) & 
                                     (result_df['stoch_k'].shift(1) <= result_df['stoch_d'].shift(1))).astype(int)
        
        return result_df
    
    @staticmethod
    def add_average_true_range(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR).
        
        Args:
            df: DataFrame with market data
            period: Period for ATR calculation
            
        Returns:
            DataFrame with added ATR column
        """
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("Cannot calculate ATR: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        # Calculate ATR
        result_df['atr'] = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=period)
        
        # Add normalized ATR (ATR as percentage of closing price)
        result_df['atr_percent'] = result_df['atr'] / result_df['close'] * 100
        
        return result_df
    
    @staticmethod
    def add_on_balance_volume(
        df: pd.DataFrame,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV).
        
        Args:
            df: DataFrame with market data
            price_column: Price column to determine volume direction
            
        Returns:
            DataFrame with added OBV column
        """
        if df.empty or not all(col in df.columns for col in [price_column, 'volume']):
            logger.warning(f"Cannot calculate OBV: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        # Calculate OBV
        result_df['obv'] = ta.obv(result_df[price_column], result_df['volume'])
        
        # Add OBV moving average
        result_df['obv_ema'] = ta.ema(result_df['obv'], length=20)
        
        return result_df
    
    @staticmethod
    def add_ichimoku_cloud(
        df: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        lagging_span_period: int = 52,
        displacement: int = 26
    ) -> pd.DataFrame:
        """
        Add Ichimoku Cloud indicators.
        
        Args:
            df: DataFrame with market data
            conversion_period: Period for Tenkan-sen (Conversion Line)
            base_period: Period for Kijun-sen (Base Line)
            lagging_span_period: Period for Senkou Span B (Leading Span B)
            displacement: Displacement for Senkou Span (Leading Spans)
            
        Returns:
            DataFrame with added Ichimoku Cloud columns
        """
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("Cannot calculate Ichimoku Cloud: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        try:
            # Calculate Ichimoku Cloud - handle both DataFrame and tuple return types
            ichimoku = ta.ichimoku(result_df['high'], result_df['low'], result_df['close'],
                                 conversion=conversion_period,
                                 base=base_period,
                                 lagging=lagging_span_period,
                                 displacement=displacement)
            
            # Check if ichimoku is a tuple (older versions of pandas-ta) or DataFrame (newer versions)
            if isinstance(ichimoku, tuple):
                # Unpack the tuple (assuming standard order of components)
                if len(ichimoku) >= 5:
                    result_df['tenkan_sen'] = ichimoku[0]  # Conversion line
                    result_df['kijun_sen'] = ichimoku[1]   # Base line
                    result_df['senkou_span_a'] = ichimoku[2]  # Leading span A
                    result_df['senkou_span_b'] = ichimoku[3]  # Leading span B
                    result_df['chikou_span'] = ichimoku[4]   # Lagging span
                else:
                    logger.warning("Ichimoku Cloud tuple has unexpected length")
            else:
                # It's a DataFrame, so add columns to the result DataFrame
                for column in ichimoku.columns:
                    result_df[column.lower()] = ichimoku[column]
            
            # Add signal columns - adjust column names based on how they were added
            if 'senkou_span_a' in result_df.columns and 'senkou_span_b' in result_df.columns:
                # Use these column names if they exist
                result_df['ichimoku_bullish'] = ((result_df['senkou_span_a'] > result_df['senkou_span_b']) & 
                                             (result_df['close'] > result_df['senkou_span_a'])).astype(int)
                result_df['ichimoku_bearish'] = ((result_df['senkou_span_a'] < result_df['senkou_span_b']) & 
                                             (result_df['close'] < result_df['senkou_span_a'])).astype(int)
            elif 'isa' in result_df.columns and 'isb' in result_df.columns:
                # Use these column names as a fallback
                result_df['ichimoku_bullish'] = ((result_df['isa'] > result_df['isb']) & 
                                             (result_df['close'] > result_df['isa'])).astype(int)
                result_df['ichimoku_bearish'] = ((result_df['isa'] < result_df['isb']) & 
                                             (result_df['close'] < result_df['isa'])).astype(int)
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {e}")
        
        return result_df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with market data (OHLCV)
            
        Returns:
            DataFrame with all technical indicators added
        """
        if df.empty:
            logger.warning("Cannot add indicators: Empty DataFrame")
            return df
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for some indicators: {missing_columns}")
        
        # Add all indicators
        result_df = df.copy()
        
        # Moving averages
        result_df = TechnicalIndicators.add_moving_averages(result_df)
        result_df = TechnicalIndicators.add_exponential_moving_averages(result_df)
        
        # Bollinger Bands
        result_df = TechnicalIndicators.add_bollinger_bands(result_df)
        
        # RSI
        result_df = TechnicalIndicators.add_rsi(result_df)
        
        # MACD
        result_df = TechnicalIndicators.add_macd(result_df)
        
        # Other indicators
        if all(col in df.columns for col in ['high', 'low']):
            result_df = TechnicalIndicators.add_stochastic_oscillator(result_df)
            result_df = TechnicalIndicators.add_average_true_range(result_df)
            result_df = TechnicalIndicators.add_ichimoku_cloud(result_df)
        
        if 'volume' in df.columns:
            result_df = TechnicalIndicators.add_on_balance_volume(result_df)
        
        return result_df


class PatternRecognition:
    """Class for recognizing candlestick patterns and chart patterns."""
    
    @staticmethod
    def recognize_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Recognize common candlestick patterns.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with candlestick pattern columns added
        """
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.warning("Cannot recognize patterns: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        # Doji pattern
        result_df['pattern_doji'] = ((abs(result_df['close'] - result_df['open']) / 
                                    (result_df['high'] - result_df['low']) < 0.1) &
                                   (result_df['high'] > result_df['open']) & 
                                   (result_df['high'] > result_df['close']) &
                                   (result_df['low'] < result_df['open']) & 
                                   (result_df['low'] < result_df['close'])).astype(int)
        
        # Hammer pattern (bullish)
        result_df['pattern_hammer'] = ((result_df['high'] - result_df['low'] > 3 * (result_df['open'] - result_df['close'])) &
                                     (result_df['close'] > result_df['open']) &
                                     (result_df['open'] - result_df['low'] > 2 * (result_df['high'] - result_df['close']))).astype(int)
        
        # Shooting Star pattern (bearish)
        result_df['pattern_shooting_star'] = ((result_df['high'] - result_df['low'] > 3 * (result_df['open'] - result_df['close'])) &
                                            (result_df['close'] < result_df['open']) &
                                            (result_df['high'] - result_df['open'] > 2 * (result_df['close'] - result_df['low']))).astype(int)
        
        # Engulfing patterns
        result_df['pattern_bullish_engulfing'] = ((result_df['close'] > result_df['open']) &
                                                (result_df['open'].shift(1) > result_df['close'].shift(1)) &
                                                (result_df['close'] > result_df['open'].shift(1)) &
                                                (result_df['open'] < result_df['close'].shift(1))).astype(int)
        
        result_df['pattern_bearish_engulfing'] = ((result_df['close'] < result_df['open']) &
                                                (result_df['open'].shift(1) < result_df['close'].shift(1)) &
                                                (result_df['close'] < result_df['open'].shift(1)) &
                                                (result_df['open'] > result_df['close'].shift(1))).astype(int)
        
        # Morning Star pattern (bullish)
        result_df['pattern_morning_star'] = ((result_df['close'].shift(2) < result_df['open'].shift(2)) & # First day bearish
                                           (abs(result_df['close'].shift(1) - result_df['open'].shift(1)) /
                                           (result_df['high'].shift(1) - result_df['low'].shift(1)) < 0.1) & # Second day doji
                                           (result_df['close'] > result_df['open']) & # Third day bullish
                                           (result_df['close'] > (result_df['close'].shift(2) + result_df['open'].shift(2)) / 2)
                                          ).astype(int)
        
        # Evening Star pattern (bearish)
        result_df['pattern_evening_star'] = ((result_df['close'].shift(2) > result_df['open'].shift(2)) & # First day bullish
                                           (abs(result_df['close'].shift(1) - result_df['open'].shift(1)) /
                                           (result_df['high'].shift(1) - result_df['low'].shift(1)) < 0.1) & # Second day doji
                                           (result_df['close'] < result_df['open']) & # Third day bearish
                                           (result_df['close'] < (result_df['close'].shift(2) + result_df['open'].shift(2)) / 2)
                                          ).astype(int)
        
        return result_df
    
    @staticmethod
    def detect_support_resistance(
        df: pd.DataFrame,
        window: int = 10,
        price_column: str = 'close',
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Detect support and resistance levels.
        
        Args:
            df: DataFrame with market data
            window: Window size for local minima/maxima detection
            price_column: Column to use for level detection
            threshold: Threshold for level proximity (as percentage)
            
        Returns:
            DataFrame with support and resistance level columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot detect support/resistance: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Find local minima and maxima
        result_df['local_min'] = result_df[price_column].rolling(window=window, center=True).min() == result_df[price_column]
        result_df['local_max'] = result_df[price_column].rolling(window=window, center=True).max() == result_df[price_column]
        
        # Initialize support and resistance columns
        result_df['support_level'] = np.nan
        result_df['resistance_level'] = np.nan
        
        # Extract support and resistance levels
        support_levels = result_df.loc[result_df['local_min'], price_column].tolist()
        resistance_levels = result_df.loc[result_df['local_max'], price_column].tolist()
        
        # Identify current price near support or resistance
        current_price = result_df[price_column].iloc[-1]
        
        # Check if current price is near any support level
        for level in support_levels:
            if abs(current_price - level) / current_price < threshold:
                result_df.loc[result_df.index[-1], 'support_level'] = level
                break
        
        # Check if current price is near any resistance level
        for level in resistance_levels:
            if abs(current_price - level) / current_price < threshold:
                result_df.loc[result_df.index[-1], 'resistance_level'] = level
                break
        
        # Add columns indicating if price is near support or resistance
        result_df['at_support'] = (~result_df['support_level'].isna()).astype(int)
        result_df['at_resistance'] = (~result_df['resistance_level'].isna()).astype(int)
        
        return result_df
    
    @staticmethod
    def detect_trend(
        df: pd.DataFrame,
        short_period: int = 20,
        long_period: int = 50,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Detect price trends using moving averages.
        
        Args:
            df: DataFrame with market data
            short_period: Period for short-term moving average
            long_period: Period for long-term moving average
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with trend indication columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot detect trend: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Calculate moving averages if not already present
        if f'sma_{short_period}' not in result_df.columns:
            result_df[f'sma_{short_period}'] = ta.sma(result_df[price_column], length=short_period)
        
        if f'sma_{long_period}' not in result_df.columns:
            result_df[f'sma_{long_period}'] = ta.sma(result_df[price_column], length=long_period)
        
        # Determine trend based on moving average relationship
        result_df['uptrend'] = (result_df[f'sma_{short_period}'] > result_df[f'sma_{long_period}']).astype(int)
        result_df['downtrend'] = (result_df[f'sma_{short_period}'] < result_df[f'sma_{long_period}']).astype(int)
        
        # Determine trend strength based on price distance from moving averages
        result_df['trend_strength'] = abs(result_df[f'sma_{short_period}'] - result_df[f'sma_{long_period}']) / result_df[price_column] * 100
        
        # Detect crossovers
        result_df['golden_cross'] = ((result_df[f'sma_{short_period}'] > result_df[f'sma_{long_period}']) & 
                                    (result_df[f'sma_{short_period}'].shift(1) <= result_df[f'sma_{long_period}'].shift(1))).astype(int)
        
        result_df['death_cross'] = ((result_df[f'sma_{short_period}'] < result_df[f'sma_{long_period}']) & 
                                   (result_df[f'sma_{short_period}'].shift(1) >= result_df[f'sma_{long_period}'].shift(1))).astype(int)
        
        return result_df