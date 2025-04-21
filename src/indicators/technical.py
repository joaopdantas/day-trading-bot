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
        
        # Check if enough data points exist for calculation
        if len(df) < max(fast_period, slow_period, signal_period):
            logger.warning(f"Cannot calculate MACD: Not enough data points. Need at least {max(fast_period, slow_period, signal_period)}, but got {len(df)}.")
            return df
        
        result_df = df.copy()
        
        try:
            # Calculate MACD
            macd = ta.macd(result_df[price_column], fast=fast_period, slow=slow_period, signal=signal_period)
            
            # Check if MACD calculation returned valid results
            if macd is None:
                logger.warning(f"MACD calculation returned None. Using manual calculation as fallback.")
                # Manual calculation as fallback
                fast_ema = result_df[price_column].ewm(span=fast_period, adjust=False).mean()
                slow_ema = result_df[price_column].ewm(span=slow_period, adjust=False).mean()
                result_df['macd'] = fast_ema - slow_ema
                result_df['macd_signal'] = result_df['macd'].ewm(span=signal_period, adjust=False).mean()
                result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
            else:
                # Try to extract values from return data structure
                try:
                    # Different versions of pandas-ta might have different column naming
                    if f'MACD_{fast_period}_{slow_period}_{signal_period}' in macd.columns:
                        result_df['macd'] = macd[f'MACD_{fast_period}_{slow_period}_{signal_period}']
                        result_df['macd_signal'] = macd[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
                        result_df['macd_histogram'] = macd[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
                    elif 'MACD' in macd.columns:
                        result_df['macd'] = macd['MACD']
                        result_df['macd_signal'] = macd['MACDs']
                        result_df['macd_histogram'] = macd['MACDh']
                    else:
                        # General case - try to use the first three columns
                        macd_cols = macd.columns.tolist()
                        if len(macd_cols) >= 3:
                            result_df['macd'] = macd[macd_cols[0]]
                            result_df['macd_signal'] = macd[macd_cols[1]]
                            result_df['macd_histogram'] = macd[macd_cols[2]]
                        else:
                            raise ValueError("Unexpected MACD result format")
                except Exception as e:
                    logger.warning(f"Error processing MACD results: {e}. Using manual calculation.")
                    # Manual calculation as fallback
                    fast_ema = result_df[price_column].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = result_df[price_column].ewm(span=slow_period, adjust=False).mean()
                    result_df['macd'] = fast_ema - slow_ema
                    result_df['macd_signal'] = result_df['macd'].ewm(span=signal_period, adjust=False).mean()
                    result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
            
            # Add MACD signal
            result_df['macd_bullish'] = (result_df['macd'] > result_df['macd_signal']).astype(int)
            result_df['macd_bearish'] = (result_df['macd'] < result_df['macd_signal']).astype(int)
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            # Don't modify the DataFrame if calculation failed
        
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
        
        # Check if we have enough data
        if len(df) < max(conversion_period, base_period, lagging_span_period, displacement):
            logger.warning(f"Not enough data for Ichimoku Cloud. Need at least {max(conversion_period, base_period, lagging_span_period, displacement)} points, but got {len(df)}.")
            return df
            
        result_df = df.copy()
        
        try:
            # Calculate Ichimoku Cloud components manually if needed
            # This is a fallback in case pandas_ta has issues
            if True:  # Always use manual calculation for consistency
                # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
                high_9 = result_df['high'].rolling(window=conversion_period).max()
                low_9 = result_df['low'].rolling(window=conversion_period).min()
                result_df['tenkan_sen'] = (high_9 + low_9) / 2
                
                # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
                high_26 = result_df['high'].rolling(window=base_period).max()
                low_26 = result_df['low'].rolling(window=base_period).min()
                result_df['kijun_sen'] = (high_26 + low_26) / 2
                
                # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 displaced forward 26 periods
                result_df['senkou_span_a'] = ((result_df['tenkan_sen'] + result_df['kijun_sen']) / 2).shift(displacement)
                
                # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, displaced forward 26 periods
                high_52 = result_df['high'].rolling(window=lagging_span_period).max()
                low_52 = result_df['low'].rolling(window=lagging_span_period).min()
                result_df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(displacement)
                
                # Chikou Span (Lagging Span): Close price shifted backwards 26 periods
                result_df['chikou_span'] = result_df['close'].shift(-displacement)
                
                # Add signal columns
                result_df['ichimoku_bullish'] = ((result_df['senkou_span_a'] > result_df['senkou_span_b']) & 
                                             (result_df['close'] > result_df['senkou_span_a'])).astype(int)
                result_df['ichimoku_bearish'] = ((result_df['senkou_span_a'] < result_df['senkou_span_b']) & 
                                             (result_df['close'] < result_df['senkou_span_a'])).astype(int)
            else:
                # Try using pandas_ta implementation (kept as reference but not used)
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
                        # Fall back to manual calculation (code would go here)
                else:
                    # It's a DataFrame, so add columns to the result DataFrame
                    for column in ichimoku.columns:
                        result_df[column.lower()] = ichimoku[column]
            
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
        
        # Check if we have enough data for the required moving averages
        if len(df) < max(short_period, long_period):
            logger.warning(f"Not enough data for trend detection. Need at least {max(short_period, long_period)} points, but got {len(df)}.")
            # Use shorter periods if we don't have enough data
            adjusted_short_period = min(short_period, len(df) // 3)
            adjusted_long_period = min(long_period, len(df) // 2)
            
            if adjusted_short_period >= 5 and adjusted_long_period > adjusted_short_period:
                logger.info(f"Using adjusted periods: short_period={adjusted_short_period}, long_period={adjusted_long_period}")
                short_period = adjusted_short_period
                long_period = adjusted_long_period
            else:
                # Add placeholder columns with neutral values
                result_df['uptrend'] = 0
                result_df['downtrend'] = 0
                result_df['trend_strength'] = 0
                result_df['golden_cross'] = 0
                result_df['death_cross'] = 0
                return result_df
        
        # Calculate moving averages if not already present
        if f'sma_{short_period}' not in result_df.columns:
            try:
                result_df[f'sma_{short_period}'] = ta.sma(result_df[price_column], length=short_period)
            except Exception as e:
                logger.warning(f"Error calculating SMA-{short_period}: {e}. Using pandas implementation.")
                result_df[f'sma_{short_period}'] = result_df[price_column].rolling(window=short_period).mean()
        
        if f'sma_{long_period}' not in result_df.columns:
            try:
                result_df[f'sma_{long_period}'] = ta.sma(result_df[price_column], length=long_period)
            except Exception as e:
                logger.warning(f"Error calculating SMA-{long_period}: {e}. Using pandas implementation.")
                result_df[f'sma_{long_period}'] = result_df[price_column].rolling(window=long_period).mean()
        
        # Handle NaN values in moving averages
        result_df[f'sma_{short_period}'] = result_df[f'sma_{short_period}'].fillna(result_df[price_column])
        result_df[f'sma_{long_period}'] = result_df[f'sma_{long_period}'].fillna(result_df[price_column])
        
        # Determine trend based on moving average relationship
        result_df['uptrend'] = (result_df[f'sma_{short_period}'] > result_df[f'sma_{long_period}']).astype(int)
        result_df['downtrend'] = (result_df[f'sma_{short_period}'] < result_df[f'sma_{long_period}']).astype(int)
        
        # Determine trend strength based on price distance from moving averages
        result_df['trend_strength'] = (abs(result_df[f'sma_{short_period}'] - result_df[f'sma_{long_period}']) / 
                                     result_df[price_column] * 100).fillna(0)
        
        # Detect crossovers - handle potential NaN values from shift operation
        short_ma_curr = result_df[f'sma_{short_period}'].fillna(0)
        short_ma_prev = result_df[f'sma_{short_period}'].shift(1).fillna(0)
        long_ma_curr = result_df[f'sma_{long_period}'].fillna(0)
        long_ma_prev = result_df[f'sma_{long_period}'].shift(1).fillna(0)
        
        result_df['golden_cross'] = ((short_ma_curr > long_ma_curr) & 
                                   (short_ma_prev <= long_ma_prev)).astype(int)
        
        result_df['death_cross'] = ((short_ma_curr < long_ma_curr) & 
                                  (short_ma_prev >= long_ma_prev)).astype(int)
        
        return result_df
    
    
    #Control Point 3
    @staticmethod
    def detect_support_resistance_advanced(
        df: pd.DataFrame,
        window: int = 10,
        price_column: str = 'close',
        threshold: float = 0.02,
        clustering_threshold: float = 0.01,
        test_lookback: int = 50
    ) -> pd.DataFrame:
        """
        Advanced detection of support and resistance levels with clustering and validation.
        
        Args:
            df: DataFrame with market data
            window: Window size for local minima/maxima detection
            price_column: Column to use for level detection
            threshold: Threshold for level proximity (as percentage)
            clustering_threshold: Threshold for clustering nearby levels (as percentage)
            test_lookback: Lookback period for testing how many times a level has been tested
            
        Returns:
            DataFrame with support and resistance level columns and additional metrics
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot detect support/resistance: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Find local minima and maxima
        result_df['local_min'] = result_df[price_column].rolling(window=window, center=True).min() == result_df[price_column]
        result_df['local_max'] = result_df[price_column].rolling(window=window, center=True).max() == result_df[price_column]
        
        # Extract support and resistance price points
        support_points = result_df.loc[result_df['local_min'], [price_column, 'volume']].copy()
        resistance_points = result_df.loc[result_df['local_max'], [price_column, 'volume']].copy()
        
        # Cluster nearby support levels
        if not support_points.empty:
            support_points = support_points.sort_values(by=price_column)
            clustered_supports = []
            current_cluster = [support_points.iloc[0][price_column]]
            current_cluster_volume = [support_points.iloc[0]['volume']]
            
            for i in range(1, len(support_points)):
                current_price = support_points.iloc[i][price_column]
                current_volume = support_points.iloc[i]['volume']
                prev_price = support_points.iloc[i-1][price_column]
                
                # If this level is close to previous one, add to current cluster
                if abs(current_price - prev_price) / prev_price < clustering_threshold:
                    current_cluster.append(current_price)
                    current_cluster_volume.append(current_volume)
                else:
                    # Calculate volume-weighted average price for the cluster
                    if sum(current_cluster_volume) > 0:
                        vwap = sum(p * v for p, v in zip(current_cluster, current_cluster_volume)) / sum(current_cluster_volume)
                    else:
                        vwap = sum(current_cluster) / len(current_cluster)
                    clustered_supports.append(vwap)
                    current_cluster = [current_price]
                    current_cluster_volume = [current_volume]
            
            # Add the last cluster
            if current_cluster:
                if sum(current_cluster_volume) > 0:
                    vwap = sum(p * v for p, v in zip(current_cluster, current_cluster_volume)) / sum(current_cluster_volume)
                else:
                    vwap = sum(current_cluster) / len(current_cluster)
                clustered_supports.append(vwap)
        else:
            clustered_supports = []
        
        # Cluster nearby resistance levels with same approach
        if not resistance_points.empty:
            resistance_points = resistance_points.sort_values(by=price_column)
            clustered_resistances = []
            current_cluster = [resistance_points.iloc[0][price_column]]
            current_cluster_volume = [resistance_points.iloc[0]['volume']]
            
            for i in range(1, len(resistance_points)):
                current_price = resistance_points.iloc[i][price_column]
                current_volume = resistance_points.iloc[i]['volume']
                prev_price = resistance_points.iloc[i-1][price_column]
                
                if abs(current_price - prev_price) / prev_price < clustering_threshold:
                    current_cluster.append(current_price)
                    current_cluster_volume.append(current_volume)
                else:
                    if sum(current_cluster_volume) > 0:
                        vwap = sum(p * v for p, v in zip(current_cluster, current_cluster_volume)) / sum(current_cluster_volume)
                    else:
                        vwap = sum(current_cluster) / len(current_cluster)
                    clustered_resistances.append(vwap)
                    current_cluster = [current_price]
                    current_cluster_volume = [current_volume]
            
            if current_cluster:
                if sum(current_cluster_volume) > 0:
                    vwap = sum(p * v for p, v in zip(current_cluster, current_cluster_volume)) / sum(current_cluster_volume)
                else:
                    vwap = sum(current_cluster) / len(current_cluster)
                clustered_resistances.append(vwap)
        else:
            clustered_resistances = []
        
        # Initialize columns
        result_df['support_level'] = np.nan
        result_df['resistance_level'] = np.nan
        result_df['support_strength'] = 0
        result_df['resistance_strength'] = 0
        result_df['support_tests'] = 0
        result_df['resistance_tests'] = 0
        
        # Current price for comparison
        current_price = result_df[price_column].iloc[-1]
        
        # Evaluate each support level
        for level in clustered_supports:
            # Calculate how many times this level was tested (price approached within threshold)
            price_near_level = abs(result_df[price_column] - level) / level < threshold
            tests = price_near_level.rolling(window=test_lookback).sum().iloc[-1] if len(result_df) > 0 else 0
            
            # Check if current price is near this support level
            if abs(current_price - level) / current_price < threshold:
                result_df.loc[result_df.index[-1], 'support_level'] = level
                result_df.loc[result_df.index[-1], 'support_tests'] = tests
                # Strength is determined by number of tests and proximity to current price
                strength = tests * (1 - abs(current_price - level) / level / threshold)
                result_df.loc[result_df.index[-1], 'support_strength'] = strength
        
        # Evaluate each resistance level
        for level in clustered_resistances:
            # Calculate how many times this level was tested
            price_near_level = abs(result_df[price_column] - level) / level < threshold
            tests = price_near_level.rolling(window=test_lookback).sum().iloc[-1] if len(result_df) > 0 else 0
            
            # Check if current price is near this resistance level
            if abs(current_price - level) / current_price < threshold:
                result_df.loc[result_df.index[-1], 'resistance_level'] = level
                result_df.loc[result_df.index[-1], 'resistance_tests'] = tests
                # Strength is determined by number of tests and proximity to current price
                strength = tests * (1 - abs(current_price - level) / level / threshold)
                result_df.loc[result_df.index[-1], 'resistance_strength'] = strength
        
        # Add boolean flags for proximity to levels
        result_df['at_support'] = (~result_df['support_level'].isna()).astype(int)
        result_df['at_resistance'] = (~result_df['resistance_level'].isna()).astype(int)
        
        # Add fractal-based support and resistance (Bill Williams' fractals)
        result_df = PatternRecognition._add_fractals(result_df)
        
        return result_df

    @staticmethod
    def _add_fractals(df: pd.DataFrame) -> pd.DataFrame:
        """Helper method to identify support/resistance using fractals."""
        if len(df) < 5:
            return df
        
        result_df = df.copy()
        
        # Bullish fractal (potential support) - lowest low with 2 higher lows on each side
        result_df['fractal_support'] = 0
        for i in range(2, len(result_df) - 2):
            if (result_df['low'].iloc[i] < result_df['low'].iloc[i-1] and
                result_df['low'].iloc[i] < result_df['low'].iloc[i-2] and
                result_df['low'].iloc[i] < result_df['low'].iloc[i+1] and
                result_df['low'].iloc[i] < result_df['low'].iloc[i+2]):
                result_df.loc[result_df.index[i], 'fractal_support'] = 1
        
        # Bearish fractal (potential resistance) - highest high with 2 lower highs on each side
        result_df['fractal_resistance'] = 0
        for i in range(2, len(result_df) - 2):
            if (result_df['high'].iloc[i] > result_df['high'].iloc[i-1] and
                result_df['high'].iloc[i] > result_df['high'].iloc[i-2] and
                result_df['high'].iloc[i] > result_df['high'].iloc[i+1] and
                result_df['high'].iloc[i] > result_df['high'].iloc[i+2]):
                result_df.loc[result_df.index[i], 'fractal_resistance'] = 1
        
        return result_df

class CandlestickPatterns:
    """Class dedicated to candlestick pattern recognition with detailed analysis."""
    
    @staticmethod
    def recognize_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Recognize comprehensive set of candlestick patterns and provide probability metrics.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with candlestick pattern columns and probability scores
        """
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.warning("Cannot recognize patterns: Empty DataFrame or missing required columns")
            return df
        
        result_df = df.copy()
        
        # Calculate basic candle properties
        result_df['body_size'] = abs(result_df['close'] - result_df['open'])
        result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        result_df['range'] = result_df['high'] - result_df['low']
        result_df['body_pct'] = result_df['body_size'] / result_df['range'].replace(0, np.nan)
        result_df['body_pct'] = result_df['body_pct'].fillna(0)
        result_df['is_bullish'] = (result_df['close'] > result_df['open']).astype(int)
        result_df['is_bearish'] = (result_df['close'] < result_df['open']).astype(int)
        
        # Add individual pattern recognition
        result_df = CandlestickPatterns._add_single_candle_patterns(result_df)
        result_df = CandlestickPatterns._add_double_candle_patterns(result_df)
        result_df = CandlestickPatterns._add_triple_candle_patterns(result_df)
        
        # Add pattern strength and probability metrics
        result_df = CandlestickPatterns._add_pattern_strength(result_df)
        
        return result_df
    
    @staticmethod
    def _add_single_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add recognition for single-candle patterns."""
        result_df = df.copy()
        
        # Doji pattern (small body relative to range)
        result_df['pattern_doji'] = (result_df['body_pct'] < 0.1).astype(int)
        
        # Long-legged Doji (small body with long shadows)
        result_df['pattern_long_legged_doji'] = ((result_df['pattern_doji'] == 1) & 
                                              (result_df['upper_shadow'] > 2 * result_df['body_size']) &
                                              (result_df['lower_shadow'] > 2 * result_df['body_size'])).astype(int)
        
        # Dragonfly Doji (open and close near high, long lower shadow)
        result_df['pattern_dragonfly_doji'] = ((result_df['pattern_doji'] == 1) &
                                            (result_df['upper_shadow'] < 0.1 * result_df['range']) &
                                            (result_df['lower_shadow'] > 0.6 * result_df['range'])).astype(int)
        
        # Gravestone Doji (open and close near low, long upper shadow)
        result_df['pattern_gravestone_doji'] = ((result_df['pattern_doji'] == 1) &
                                             (result_df['lower_shadow'] < 0.1 * result_df['range']) &
                                             (result_df['upper_shadow'] > 0.6 * result_df['range'])).astype(int)
        
        # Hammer pattern (bullish: small body near high with long lower shadow)
        result_df['pattern_hammer'] = ((result_df['is_bullish'] == 1) &
                                     (result_df['body_pct'] < 0.3) &
                                     (result_df['lower_shadow'] > 2 * result_df['body_size']) &
                                     (result_df['upper_shadow'] < 0.1 * result_df['body_size'])).astype(int)
        
        # Inverted Hammer pattern (potentially bullish: small body near low with long upper shadow)
        result_df['pattern_inverted_hammer'] = ((result_df['is_bullish'] == 1) &
                                             (result_df['body_pct'] < 0.3) &
                                             (result_df['upper_shadow'] > 2 * result_df['body_size']) &
                                             (result_df['lower_shadow'] < 0.1 * result_df['body_size'])).astype(int)
        
        # Shooting Star pattern (bearish: small body near low with long upper shadow)
        result_df['pattern_shooting_star'] = ((result_df['is_bearish'] == 1) &
                                           (result_df['body_pct'] < 0.3) &
                                           (result_df['upper_shadow'] > 2 * result_df['body_size']) &
                                           (result_df['lower_shadow'] < 0.1 * result_df['body_size'])).astype(int)
        
        # Hanging Man pattern (bearish: small body near high with long lower shadow)
        result_df['pattern_hanging_man'] = ((result_df['is_bearish'] == 1) &
                                         (result_df['body_pct'] < 0.3) &
                                         (result_df['lower_shadow'] > 2 * result_df['body_size']) &
                                         (result_df['upper_shadow'] < 0.1 * result_df['body_size'])).astype(int)
        
        # Marubozu (long body with very small or no shadows)
        result_df['pattern_marubozu'] = ((result_df['body_pct'] > 0.8) &
                                      (result_df['upper_shadow'] < 0.1 * result_df['body_size']) &
                                      (result_df['lower_shadow'] < 0.1 * result_df['body_size'])).astype(int)
        
        # Bullish Marubozu (long bullish body with very small or no shadows)
        result_df['pattern_bullish_marubozu'] = ((result_df['pattern_marubozu'] == 1) &
                                              (result_df['is_bullish'] == 1)).astype(int)
        
        # Bearish Marubozu (long bearish body with very small or no shadows)
        result_df['pattern_bearish_marubozu'] = ((result_df['pattern_marubozu'] == 1) &
                                              (result_df['is_bearish'] == 1)).astype(int)
        
        # Spinning Top (small body with shadows on both sides)
        result_df['pattern_spinning_top'] = ((result_df['body_pct'] < 0.3) &
                                          (result_df['upper_shadow'] > result_df['body_size']) &
                                          (result_df['lower_shadow'] > result_df['body_size'])).astype(int)
        
        return result_df
    
    @staticmethod
    def _add_double_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add recognition for patterns involving two consecutive candles."""
        if len(df) < 2:
            return df
            
        result_df = df.copy()
        
        # Bullish Engulfing pattern
        result_df['pattern_bullish_engulfing'] = ((result_df['is_bullish'] == 1) &
                                               (result_df['is_bearish'].shift(1) == 1) &
                                               (result_df['open'] < result_df['close'].shift(1)) &
                                               (result_df['close'] > result_df['open'].shift(1))).astype(int)
        
        # Bearish Engulfing pattern
        result_df['pattern_bearish_engulfing'] = ((result_df['is_bearish'] == 1) &
                                               (result_df['is_bullish'].shift(1) == 1) &
                                               (result_df['open'] > result_df['close'].shift(1)) &
                                               (result_df['close'] < result_df['open'].shift(1))).astype(int)
        
        # Bullish Harami pattern (small body contained within prior bearish body)
        result_df['pattern_bullish_harami'] = ((result_df['is_bullish'] == 1) &
                                            (result_df['is_bearish'].shift(1) == 1) &
                                            (result_df['open'] > result_df['close'].shift(1)) &
                                            (result_df['close'] < result_df['open'].shift(1)) &
                                            (result_df['body_size'] < 0.5 * result_df['body_size'].shift(1))).astype(int)
        
        # Bearish Harami pattern (small body contained within prior bullish body)
        result_df['pattern_bearish_harami'] = ((result_df['is_bearish'] == 1) &
                                            (result_df['is_bullish'].shift(1) == 1) &
                                            (result_df['open'] < result_df['close'].shift(1)) &
                                            (result_df['close'] > result_df['open'].shift(1)) &
                                            (result_df['body_size'] < 0.5 * result_df['body_size'].shift(1))).astype(int)
        
        # Tweezer Top (similar highs with first bullish, second bearish)
        result_df['pattern_tweezer_top'] = ((result_df['is_bearish'] == 1) &
                                         (result_df['is_bullish'].shift(1) == 1) &
                                         (abs(result_df['high'] - result_df['high'].shift(1)) / result_df['high'] < 0.001)).astype(int)
        
        # Tweezer Bottom (similar lows with first bearish, second bullish)
        result_df['pattern_tweezer_bottom'] = ((result_df['is_bullish'] == 1) &
                                            (result_df['is_bearish'].shift(1) == 1) &
                                            (abs(result_df['low'] - result_df['low'].shift(1)) / result_df['low'] < 0.001)).astype(int)
        
        # Piercing Line pattern (bullish reversal pattern)
        result_df['pattern_piercing_line'] = ((result_df['is_bullish'] == 1) &
                                           (result_df['is_bearish'].shift(1) == 1) &
                                           (result_df['open'] < result_df['low'].shift(1)) &
                                           (result_df['close'] > (result_df['open'].shift(1) + result_df['close'].shift(1)) / 2) &
                                           (result_df['close'] < result_df['open'].shift(1))).astype(int)
        
        # Dark Cloud Cover pattern (bearish reversal pattern)
        result_df['pattern_dark_cloud_cover'] = ((result_df['is_bearish'] == 1) &
                                              (result_df['is_bullish'].shift(1) == 1) &
                                              (result_df['open'] > result_df['high'].shift(1)) &
                                              (result_df['close'] < (result_df['open'].shift(1) + result_df['close'].shift(1)) / 2) &
                                              (result_df['close'] > result_df['close'].shift(1))).astype(int)
                                             
        return result_df
    
    @staticmethod
    def _add_triple_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add recognition for patterns involving three consecutive candles."""
        if len(df) < 3:
            return df
            
        result_df = df.copy()
        
        # Morning Star pattern (bullish reversal)
        result_df['pattern_morning_star'] = ((result_df['is_bearish'].shift(2) == 1) &  # First candle: bearish
                                          (result_df['body_size'].shift(1) < 0.3 * result_df['body_size'].shift(2)) &  # Second: small body
                                          (result_df['is_bullish'] == 1) &  # Third candle: bullish
                                          (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)).astype(int)
        
        # Evening Star pattern (bearish reversal)
        result_df['pattern_evening_star'] = ((result_df['is_bullish'].shift(2) == 1) &  # First candle: bullish
                                          (result_df['body_size'].shift(1) < 0.3 * result_df['body_size'].shift(2)) &  # Second: small body
                                          (result_df['is_bearish'] == 1) &  # Third candle: bearish
                                          (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)).astype(int)
        
        # Three White Soldiers (strong bullish reversal)
        result_df['pattern_three_white_soldiers'] = ((result_df['is_bullish'] == 1) &
                                                  (result_df['is_bullish'].shift(1) == 1) &
                                                  (result_df['is_bullish'].shift(2) == 1) &
                                                  (result_df['close'] > result_df['close'].shift(1)) &
                                                  (result_df['close'].shift(1) > result_df['close'].shift(2)) &
                                                  (result_df['open'] > result_df['open'].shift(1)) &
                                                  (result_df['open'].shift(1) > result_df['open'].shift(2)) &
                                                  (result_df['open'] < result_df['close'].shift(1)) &
                                                  (result_df['open'].shift(1) < result_df['close'].shift(2))).astype(int)
        
        # Three Black Crows (strong bearish reversal)
        result_df['pattern_three_black_crows'] = ((result_df['is_bearish'] == 1) &
                                               (result_df['is_bearish'].shift(1) == 1) &
                                               (result_df['is_bearish'].shift(2) == 1) &
                                               (result_df['close'] < result_df['close'].shift(1)) &
                                               (result_df['close'].shift(1) < result_df['close'].shift(2)) &
                                               (result_df['open'] < result_df['open'].shift(1)) &
                                               (result_df['open'].shift(1) < result_df['open'].shift(2)) &
                                               (result_df['open'] > result_df['close'].shift(1)) &
                                               (result_df['open'].shift(1) > result_df['close'].shift(2))).astype(int)
        
        # Three Inside Up (bullish continuation)
        result_df['pattern_three_inside_up'] = ((result_df['pattern_bullish_harami'].shift(1) == 1) &  # First two form bullish harami
                                             (result_df['is_bullish'] == 1) &  # Third is bullish
                                             (result_df['close'] > result_df['close'].shift(1))).astype(int)  # Closes above second candle
        
        # Three Inside Down (bearish continuation)
        result_df['pattern_three_inside_down'] = ((result_df['pattern_bearish_harami'].shift(1) == 1) &  # First two form bearish harami
                                               (result_df['is_bearish'] == 1) &  # Third is bearish
                                               (result_df['close'] < result_df['close'].shift(1))).astype(int)  # Closes below second candle
        
        # Three Outside Up (bullish reversal)
        result_df['pattern_three_outside_up'] = ((result_df['pattern_bullish_engulfing'].shift(1) == 1) &  # First two form bullish engulfing
                                              (result_df['is_bullish'] == 1) &  # Third is bullish
                                              (result_df['close'] > result_df['close'].shift(1))).astype(int)  # Confirms trend
        
        # Three Outside Down (bearish reversal)
        result_df['pattern_three_outside_down'] = ((result_df['pattern_bearish_engulfing'].shift(1) == 1) &  # First two form bearish engulfing
                                                (result_df['is_bearish'] == 1) &  # Third is bearish
                                                (result_df['close'] < result_df['close'].shift(1))).astype(int)  # Confirms trend
        
        return result_df
    
    @staticmethod
    def _add_pattern_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern strength indicators and trend context analysis.
        This evaluates how reliable each pattern is likely to be based on market context.
        """
        result_df = df.copy()
        
        # Get all pattern columns
        pattern_columns = [col for col in result_df.columns if col.startswith('pattern_')]
        
        # Initialize columns for bullish and bearish signals
        result_df['candlestick_bullish_signals'] = 0
        result_df['candlestick_bearish_signals'] = 0
        
        # Define patterns and their strengths/directions
        pattern_properties = {
            # Pattern name: (is_bullish, strength_value)
            'pattern_hammer': (True, 2),
            'pattern_inverted_hammer': (True, 1),
            'pattern_bullish_engulfing': (True, 3),
            'pattern_piercing_line': (True, 2),
            'pattern_morning_star': (True, 4),
            'pattern_bullish_harami': (True, 2),
            'pattern_bullish_marubozu': (True, 3),
            'pattern_three_white_soldiers': (True, 5),
            'pattern_tweezer_bottom': (True, 2),
            'pattern_three_inside_up': (True, 3),
            'pattern_three_outside_up': (True, 4),
            'pattern_dragonfly_doji': (True, 1),
            
            'pattern_shooting_star': (False, 2),
            'pattern_hanging_man': (False, 2),
            'pattern_bearish_engulfing': (False, 3),
            'pattern_dark_cloud_cover': (False, 2),
            'pattern_evening_star': (False, 4),
            'pattern_bearish_harami': (False, 2),
            'pattern_bearish_marubozu': (False, 3),
            'pattern_three_black_crows': (False, 5),
            'pattern_tweezer_top': (False, 2),
            'pattern_three_inside_down': (False, 3),
            'pattern_three_outside_down': (False, 4),
            'pattern_gravestone_doji': (False, 1)
        }
        
        # Calculate cumulative signal strengths
        for pattern, (is_bullish, strength) in pattern_properties.items():
            if pattern in result_df.columns:
                if is_bullish:
                    result_df['candlestick_bullish_signals'] += result_df[pattern] * strength
                else:
                    result_df['candlestick_bearish_signals'] += result_df[pattern] * strength
        
        # Calculate overall signal strength (positive for bullish, negative for bearish)
        result_df['candlestick_signal_strength'] = result_df['candlestick_bullish_signals'] - result_df['candlestick_bearish_signals']
        
        # Add context-aware pattern significance (example: patterns are more significant at support/resistance)
        if 'at_support' in result_df.columns and 'at_resistance' in result_df.columns:
            # Bullish patterns are more significant near support
            result_df['candlestick_bullish_significance'] = result_df['candlestick_bullish_signals'] * (1 + 0.5 * result_df['at_support'])
            # Bearish patterns are more significant near resistance
            result_df['candlestick_bearish_significance'] = result_df['candlestick_bearish_signals'] * (1 + 0.5 * result_df['at_resistance'])
        else:
            result_df['candlestick_bullish_significance'] = result_df['candlestick_bullish_signals']
            result_df['candlestick_bearish_significance'] = result_df['candlestick_bearish_signals']
        
        # Add a combined significance score
        result_df['candlestick_signal_significance'] = result_df['candlestick_bullish_significance'] - result_df['candlestick_bearish_significance']
        
        return result_df
    
    @staticmethod
    def get_active_patterns(row: pd.Series) -> Dict[str, Dict]:
        """
        Extract active patterns from a row with additional metadata.
        
        Args:
            row: Single row from DataFrame with pattern recognition columns
            
        Returns:
            Dictionary of active patterns with metadata about each pattern
        """
        pattern_metadata = {
            'pattern_doji': {
                'name': 'Doji',
                'type': 'indecision',
                'description': 'Open and close at nearly the same price, indicating market indecision',
                'reliability': 'medium'
            },
            'pattern_hammer': {
                'name': 'Hammer',
                'type': 'bullish_reversal',
                'description': 'Small body near the high with a long lower shadow, potential bullish reversal signal',
                'reliability': 'high'
            },
            'pattern_shooting_star': {
                'name': 'Shooting Star',
                'type': 'bearish_reversal',
                'description': 'Small body near the low with a long upper shadow, potential bearish reversal signal',
                'reliability': 'high'
            },
            'pattern_bullish_engulfing': {
                'name': 'Bullish Engulfing',
                'type': 'bullish_reversal',
                'description': 'Second candle completely engulfs the body of the first bearish candle',
                'reliability': 'very_high'
            },
            'pattern_bearish_engulfing': {
                'name': 'Bearish Engulfing',
                'type': 'bearish_reversal',
                'description': 'Second candle completely engulfs the body of the first bullish candle',
                'reliability': 'very_high'
            },
            'pattern_morning_star': {
                'name': 'Morning Star',
                'type': 'bullish_reversal',
                'description': 'Three-candle pattern showing a potential bottom reversal',
                'reliability': 'very_high'
            },
            'pattern_evening_star': {
                'name': 'Evening Star',
                'type': 'bearish_reversal',
                'description': 'Three-candle pattern showing a potential top reversal',
                'reliability': 'very_high'
            },
            # Add metadata for all other patterns...
        }
        
        active_patterns = {}
        
        for column, value in row.items():
            if column.startswith('pattern_') and value == 1:
                pattern_key = column
                if pattern_key in pattern_metadata:
                    active_patterns[pattern_key] = pattern_metadata[pattern_key]
                else:
                    # Generic metadata for patterns not explicitly defined
                    pattern_name = ' '.join(word.capitalize() for word in pattern_key.replace('pattern_', '').split('_'))
                    active_patterns[pattern_key] = {
                        'name': pattern_name,
                        'type': 'unknown',
                        'description': f'{pattern_name} candlestick pattern',
                        'reliability': 'medium'
                    }
        
        return active_patterns
    
    @staticmethod
    def visualize_patterns(
        df: pd.DataFrame,
        window_size: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of candlestick patterns in the data.
        
        Args:
            df: DataFrame with pattern recognition columns
            window_size: Number of candles to show in visualization
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure object
        """
        if len(df) < 2:
            logger.warning("Not enough data for candlestick visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Insufficient data for visualization", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Get last window_size candles
        plot_df = df.iloc[-window_size:].copy() if len(df) > window_size else df.copy()
        
        # Setup figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlesticks
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates
        
        # Convert index to numeric format for plotting
        plot_df = plot_df.reset_index()
        if pd.api.types.is_datetime64_any_dtype(plot_df['index']):
            plot_df['date_num'] = mdates.date2num(plot_df['index'])
        else:
            plot_df['date_num'] = range(len(plot_df))
        
        # Create OHLC data format
        ohlc = plot_df[['date_num', 'open', 'high', 'low', 'close']].values
        
        # Plot candlesticks
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red')
        
        # Format x axis for dates if available
        if pd.api.types.is_datetime64_any_dtype(plot_df['index']):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        
        # Annotate patterns
        y_range = plot_df['high'].max() - plot_df['low'].min()
        
        # Find pattern columns
        pattern_columns = [col for col in plot_df.columns if col.startswith('pattern_')]
        
        # Loop through each candle and add annotations for patterns
        for i, row in plot_df.iterrows():
            patterns_found = []
            for pattern in pattern_columns:
                if row[pattern] == 1:
                    # Convert pattern_snake_case to Pattern Name
                    pattern_name = ' '.join(word.capitalize() for word in pattern.replace('pattern_', '').split('_'))
                    patterns_found.append(pattern_name)
            
            if patterns_found:
                # Determine if it should be above or below the candle
                if row['close'] > row['open']:  # bullish candle
                    y_pos = row['high'] + y_range * 0.02
                    va = 'bottom'
                else:  # bearish candle
                    y_pos = row['low'] - y_range * 0.02
                    va = 'top'
                
                # Add marker
                ax1.annotate('⭐', (row['date_num'], y_pos), ha='center', va=va, fontsize=12)
                
                # If it's the last few candles, add pattern names
                if i >= len(plot_df) - 5:
                    ax1.annotate('\n'.join(patterns_found), 
                               (row['date_num'], y_pos + y_range * (0.03 if va == 'bottom' else -0.03)), 
                               ha='center', va=va, fontsize=8, rotation=45)
        
        # Add pattern strength indicators in bottom subplot
        if 'candlestick_signal_strength' in plot_df.columns:
            signal_strength = plot_df['candlestick_signal_strength']
            bars = ax2.bar(plot_df['date_num'], signal_strength, width=0.6, 
                         color=np.where(signal_strength >= 0, 'green', 'red'))
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_title('Candlestick Pattern Signal Strength (Positive=Bullish, Negative=Bearish)')
        
        # Set titles
        ax1.set_title('Candlestick Chart with Pattern Recognition')
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Signal Strength')
        
        # Remove x-axis labels from top plot
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
class TrendIdentification:
    """Advanced trend identification and analysis for trading algorithms."""
    
    @staticmethod
    def identify_trend(
        df: pd.DataFrame,
        short_period: int = 20,
        medium_period: int = 50,
        long_period: int = 200,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Identify market trends using multiple technical indicators.
        
        Args:
            df: DataFrame with market data
            short_period: Period for short-term moving average
            medium_period: Period for medium-term moving average
            long_period: Period for long-term moving average
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            price_column: Column to use for calculations
            
        Returns:
            DataFrame with comprehensive trend indicators
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot identify trend: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        logger.info(f"Identifying trend on data with {len(result_df)} rows")
        
        # Calculate required indicators if not already present
        # Moving averages
        for period in [short_period, medium_period, long_period]:
            if f'ema_{period}' not in result_df.columns:
                try:
                    result_df = TechnicalIndicators.add_exponential_moving_averages(
                        result_df, periods=[period], price_column=price_column
                    )
                except Exception as e:
                    logger.error(f"Error calculating EMA-{period}: {e}")
        
        # RSI
        if 'rsi' not in result_df.columns:
            try:
                result_df = TechnicalIndicators.add_rsi(result_df, period=rsi_period)
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
        
        # MACD
        if 'macd' not in result_df.columns:
            try:
                result_df = TechnicalIndicators.add_macd(result_df)
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")
        
        # Support/Resistance
        try:
            result_df = PatternRecognition.detect_support_resistance_advanced(result_df)
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
        
        # Initialize trend columns
        result_df['trend_direction'] = 0  # -2: Strong downtrend, -1: Downtrend, 0: Neutral, 1: Uptrend, 2: Strong uptrend
        result_df['trend_strength'] = 0.0  # 0.0 to 1.0
        result_df['trend_duration'] = 0  # Number of periods the trend has persisted
        result_df['trend_reversal_probability'] = 0.0  # 0.0 to 1.0
        
        # -----------------------------------------------------
        # Primary Trend Identification (Moving Average Analysis)
        # -----------------------------------------------------
        result_df['ma_alignment'] = 0
        
        # Check for aligned moving averages (stronger trend indication)
        if all(col in result_df.columns for col in [f'ema_{short_period}', f'ema_{medium_period}', f'ema_{long_period}']):
            # Bullish alignment: short > medium > long
            result_df['bullish_alignment'] = ((result_df[f'ema_{short_period}'] > result_df[f'ema_{medium_period}']) & 
                                             (result_df[f'ema_{medium_period}'] > result_df[f'ema_{long_period}'])).astype(int)
            
            # Bearish alignment: short < medium < long
            result_df['bearish_alignment'] = ((result_df[f'ema_{short_period}'] < result_df[f'ema_{medium_period}']) & 
                                             (result_df[f'ema_{medium_period}'] < result_df[f'ema_{long_period}'])).astype(int)
            
            # Calculate alignment score: 2 for full alignment, 1 for partial, 0 for none, negative for opposite
            result_df['ma_alignment'] = result_df.apply(
                lambda row: 2 if row['bullish_alignment'] == 1 else 
                          (-2 if row['bearish_alignment'] == 1 else 
                           1 if (row[f'ema_{short_period}'] > row[f'ema_{medium_period}']) else
                          (-1 if (row[f'ema_{short_period}'] < row[f'ema_{medium_period}']) else 0)),
                axis=1
            )
            
            # Identify price in relation to moving averages
            result_df['price_above_short_ma'] = (result_df[price_column] > result_df[f'ema_{short_period}']).astype(int)
            result_df['price_above_medium_ma'] = (result_df[price_column] > result_df[f'ema_{medium_period}']).astype(int)
            result_df['price_above_long_ma'] = (result_df[price_column] > result_df[f'ema_{long_period}']).astype(int)
            
            # Calculate price-MA relationship score
            result_df['price_ma_score'] = (result_df['price_above_short_ma'] + 
                                          result_df['price_above_medium_ma'] + 
                                          result_df['price_above_long_ma'] - 1.5)  # -1.5 to center around 0
        
        # -----------------------------------------------------
        # Momentum Analysis
        # -----------------------------------------------------
        result_df['momentum_score'] = 0
        
        # Add RSI momentum component
        if 'rsi' in result_df.columns:
            # RSI trend component: Map RSI from 0-100 to -1 to 1
            result_df['rsi_trend'] = (result_df['rsi'] - 50) / 50
            
            # RSI extreme levels (can indicate potential reversal)
            result_df['rsi_overbought'] = (result_df['rsi'] > rsi_overbought).astype(int)
            result_df['rsi_oversold'] = (result_df['rsi'] < rsi_oversold).astype(int)
        else:
            result_df['rsi_trend'] = 0
            result_df['rsi_overbought'] = 0
            result_df['rsi_oversold'] = 0
        
        # Add MACD momentum component
        if all(col in result_df.columns for col in ['macd', 'macd_signal']):
            # MACD relation to signal line
            result_df['macd_above_signal'] = (result_df['macd'] > result_df['macd_signal']).astype(int)
            
            # MACD normalized by price for better comparison across different assets
            result_df['macd_normalized'] = result_df['macd'] / result_df[price_column] * 100
            
            # MACD zero-line relationship (positive or negative territory)
            result_df['macd_positive'] = (result_df['macd'] > 0).astype(int)
            
            # Combined MACD trend signal: 1 for bullish (above signal & positive), -1 for bearish
            result_df['macd_trend'] = result_df.apply(
                lambda row: 1 if row['macd_positive'] == 1 and row['macd_above_signal'] == 1 else
                          (-1 if row['macd_positive'] == 0 and row['macd_above_signal'] == 0 else 0),
                axis=1
            )
        else:
            result_df['macd_trend'] = 0
        
        # Calculate overall momentum score (-1 to 1)
        result_df['momentum_score'] = result_df.apply(
            lambda row: (row['rsi_trend'] * 0.5 + row['macd_trend'] * 0.5),
            axis=1
        )
        
        # -----------------------------------------------------
        # Volume Analysis
        # -----------------------------------------------------
        if 'volume' in result_df.columns:
            # Calculate volume moving average
            result_df['volume_sma'] = result_df['volume'].rolling(window=short_period).mean()
            
            # Volume trend (increasing or decreasing)
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume_sma']
            
            # Volume confirmation (higher in direction of trend)
            result_df['volume_confirms_price'] = result_df.apply(
                lambda row: (row['volume_ratio'] > 1 and row[price_column] > row[price_column.replace('close', 'open')]) or
                           (row['volume_ratio'] <= 1 and row[price_column] <= row[price_column.replace('close', 'open')]),
                axis=1
            ).astype(int)
        else:
            result_df['volume_confirms_price'] = 0
        
        # -----------------------------------------------------
        # Support/Resistance Context
        # -----------------------------------------------------
        # Check if price is near support or resistance
        if 'at_support' in result_df.columns and 'at_resistance' in result_df.columns:
            # No change needed
            pass
        else:
            result_df['at_support'] = 0
            result_df['at_resistance'] = 0
        
        # -----------------------------------------------------
        # Combine Components for Final Trend Assessment
        # -----------------------------------------------------
        # Calculate primary trend direction
        result_df['trend_direction'] = result_df.apply(
            lambda row: (
                # Strong uptrend
                2 if (row['ma_alignment'] >= 1.5 and row['momentum_score'] > 0.3 and row['price_ma_score'] > 1) else
                # Uptrend
                1 if (row['ma_alignment'] > 0 and row['momentum_score'] > 0 and row['price_ma_score'] > 0) else
                # Strong downtrend
                -2 if (row['ma_alignment'] <= -1.5 and row['momentum_score'] < -0.3 and row['price_ma_score'] < -1) else
                # Downtrend
                -1 if (row['ma_alignment'] < 0 and row['momentum_score'] < 0 and row['price_ma_score'] < 0) else
                # Neutral
                0
            ),
            axis=1
        )
        
        # Calculate trend strength (0 to 1)
        result_df['trend_strength'] = result_df.apply(
            lambda row: min(1.0, abs(
                0.4 * abs(row['ma_alignment']) / 2 +
                0.3 * abs(row['momentum_score']) +
                0.2 * abs(row['price_ma_score']) / 3 +
                0.1 * row['volume_confirms_price']
            )),
            axis=1
        )
        
        # Calculate trend duration (number of periods with same trend direction)
        # Initialize with 1 for the first row
        result_df['trend_duration'] = 1
        
        # For each subsequent row, check if trend direction matches previous
        for i in range(1, len(result_df)):
            if result_df['trend_direction'].iloc[i] == result_df['trend_direction'].iloc[i-1] and result_df['trend_direction'].iloc[i] != 0:
                result_df.loc[result_df.index[i], 'trend_duration'] = result_df['trend_duration'].iloc[i-1] + 1
        
        # Calculate reversal probability (higher when trend gets extended or near support/resistance)
        result_df['trend_reversal_probability'] = result_df.apply(
            lambda row: min(0.9, (
                # Base probability
                0.1 +
                # Extended trend increases reversal probability
                (min(0.3, row['trend_duration'] / 100)) +
                # RSI extreme levels increase reversal probability
                (0.3 if (row['trend_direction'] > 0 and row['rsi_overbought'] == 1) or 
                        (row['trend_direction'] < 0 and row['rsi_oversold'] == 1) else 0) +
                # Price approaching S/R increases reversal probability
                (0.3 if (row['trend_direction'] > 0 and row['at_resistance'] == 1) or 
                        (row['trend_direction'] < 0 and row['at_support'] == 1) else 0)
            )),
            axis=1
        )
        
        # -----------------------------------------------------
        # Add Trend Signals for Trading Decisions
        # -----------------------------------------------------
        # Trend change signals
        result_df['trend_change_bullish'] = ((result_df['trend_direction'] > 0) & 
                                            (result_df['trend_direction'].shift(1) <= 0)).astype(int)
        
        result_df['trend_change_bearish'] = ((result_df['trend_direction'] < 0) & 
                                           (result_df['trend_direction'].shift(1) >= 0)).astype(int)
        
        # High-probability entry signals (trend confirmation with momentum)
        result_df['trend_entry_long'] = ((result_df['trend_direction'] > 0) & 
                                         (result_df['momentum_score'] > 0.2) & 
                                         (result_df['trend_strength'] > 0.6) &
                                         (result_df['at_support'] == 1)).astype(int)
        
        result_df['trend_entry_short'] = ((result_df['trend_direction'] < 0) & 
                                          (result_df['momentum_score'] < -0.2) & 
                                          (result_df['trend_strength'] > 0.6) &
                                          (result_df['at_resistance'] == 1)).astype(int)
        
        # Exit signals when trend weakens
        result_df['trend_exit_long'] = ((result_df['trend_direction'].shift(1) > 0) & 
                                        ((result_df['trend_direction'] <= 0) | 
                                         (result_df['trend_reversal_probability'] > 0.7) |
                                         (result_df['at_resistance'] == 1))).astype(int)
        
        result_df['trend_exit_short'] = ((result_df['trend_direction'].shift(1) < 0) & 
                                         ((result_df['trend_direction'] >= 0) | 
                                          (result_df['trend_reversal_probability'] > 0.7) |
                                          (result_df['at_support'] == 1))).astype(int)
        
        # Create trend_status text description for easy interpretation
        result_df['trend_status'] = result_df.apply(
            lambda row: (
                "Strong Uptrend" if row['trend_direction'] == 2 else
                "Uptrend" if row['trend_direction'] == 1 else
                "Strong Downtrend" if row['trend_direction'] == -2 else
                "Downtrend" if row['trend_direction'] == -1 else
                "Neutral/Sideways"
            ),
            axis=1
        )
        
        # Add a concise trading recommendation based on all signals
        result_df['trend_recommendation'] = result_df.apply(
            lambda row: (
                "Strong Buy" if row['trend_entry_long'] == 1 and row['trend_strength'] > 0.8 else
                "Buy" if row['trend_direction'] > 0 and row['trend_strength'] > 0.6 else
                "Weak Buy" if row['trend_direction'] > 0 and row['trend_strength'] <= 0.6 else
                "Strong Sell" if row['trend_entry_short'] == 1 and row['trend_strength'] > 0.8 else
                "Sell" if row['trend_direction'] < 0 and row['trend_strength'] > 0.6 else
                "Weak Sell" if row['trend_direction'] < 0 and row['trend_strength'] <= 0.6 else
                "Hold/Neutral"
            ),
            axis=1
        )
        
        logger.info(f"Trend identification complete: {result_df['trend_status'].iloc[-1]} with strength {result_df['trend_strength'].iloc[-1]:.2f}")
        return result_df
    
    @staticmethod
    def add_breakout_detection(
        df: pd.DataFrame,
        lookback_period: int = 20,
        price_column: str = 'close',
        volume_factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect potential price breakouts from consolidation patterns.
        
        Args:
            df: DataFrame with market data
            lookback_period: Period for range calculation
            price_column: Column to use for breakout detection
            volume_factor: Factor for volume confirmation (volume > volume_factor * avg_volume)
            
        Returns:
            DataFrame with breakout detection columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(f"Cannot detect breakouts: Empty DataFrame or missing '{price_column}' column")
            return df
        
        result_df = df.copy()
        
        # Calculate price range over lookback period
        result_df['price_range'] = result_df[price_column].rolling(window=lookback_period).max() - \
                                  result_df[price_column].rolling(window=lookback_period).min()
        
        # Calculate price range as percentage of current price
        result_df['price_range_pct'] = result_df['price_range'] / result_df[price_column] * 100
        
        # Calculate average range as baseline
        result_df['avg_range_pct'] = result_df['price_range_pct'].rolling(window=lookback_period*2).mean()
        
        # Define consolidation as periods where range is significantly lower than average
        result_df['is_consolidating'] = (result_df['price_range_pct'] < 0.7 * result_df['avg_range_pct']).astype(int)
        
        # Calculate consecutive consolidation periods
        result_df['consolidation_count'] = 0
        current_count = 0
        
        for i in range(len(result_df)):
            if result_df['is_consolidating'].iloc[i] == 1:
                current_count += 1
            else:
                current_count = 0
            result_df.loc[result_df.index[i], 'consolidation_count'] = current_count
        
        # Check for breakouts from consolidation
        # Breakout definition: Price moves outside recent range with higher volume
        if len(result_df) >= lookback_period:
            for i in range(lookback_period, len(result_df)):
                # Only consider breakouts after some consolidation
                if result_df['consolidation_count'].iloc[i-1] >= lookback_period // 4:
                    recent_high = result_df[price_column].iloc[i-lookback_period:i].max()
                    recent_low = result_df[price_column].iloc[i-lookback_period:i].min()
                    current_price = result_df[price_column].iloc[i]
                    
                    # Check for volume confirmation if volume data is available
                    volume_confirmed = True
                    if 'volume' in result_df.columns:
                        avg_volume = result_df['volume'].iloc[i-lookback_period:i].mean()
                        current_volume = result_df['volume'].iloc[i]
                        volume_confirmed = current_volume > volume_factor * avg_volume
                    
                    # Bullish breakout
                    if current_price > recent_high and volume_confirmed:
                        result_df.loc[result_df.index[i], 'breakout_bullish'] = 1
                        # Calculate potential breakout target (range projection)
                        breakout_range = recent_high - recent_low
                        result_df.loc[result_df.index[i], 'breakout_target'] = current_price + breakout_range
                    else:
                        result_df.loc[result_df.index[i], 'breakout_bullish'] = 0
                    
                    # Bearish breakout
                    if current_price < recent_low and volume_confirmed:
                        result_df.loc[result_df.index[i], 'breakout_bearish'] = 1
                        # Calculate potential breakdown target
                        breakout_range = recent_high - recent_low
                        result_df.loc[result_df.index[i], 'breakout_target'] = current_price - breakout_range
                    else:
                        result_df.loc[result_df.index[i], 'breakout_bearish'] = 0
                else:
                    result_df.loc[result_df.index[i], 'breakout_bullish'] = 0
                    result_df.loc[result_df.index[i], 'breakout_bearish'] = 0
        
        # Fill NaN values in new columns
        result_df['breakout_bullish'] = result_df['breakout_bullish'].fillna(0).astype(int)
        result_df['breakout_bearish'] = result_df['breakout_bearish'].fillna(0).astype(int)
        result_df['breakout_target'] = result_df['breakout_target'].fillna(0)
        
        return result_df
    
    @staticmethod
    def analyze_trend_potential(df: pd.DataFrame) -> dict:
        """
        Analyze the current trend potential based on multiple indicators.
        
        Args:
            df: DataFrame with calculated technical indicators
            
        Returns:
            Dictionary with trend analysis results
        """
        if df.empty:
            return {
                'trend_status': 'Unknown',
                'confidence': 0,
                'potential': 0,
                'risk_reward': 0,
                'recommendation': 'Insufficient data'
            }
        
        # Get most recent values
        last_row = df.iloc[-1]
        
        # Extract trend information
        trend_status = last_row.get('trend_status', 'Unknown')
        trend_direction = last_row.get('trend_direction', 0)
        trend_strength = last_row.get('trend_strength', 0)
        trend_duration = last_row.get('trend_duration', 0)
        reversal_probability = last_row.get('trend_reversal_probability', 0.5)
        
        # Calculate trend confidence (0-100%)
        confidence = int(trend_strength * 100)
        
        # Calculate trend potential (remaining opportunity)
        # Lower reversal probability and strong trend means higher potential
        potential = int((1 - reversal_probability) * trend_strength * 100)
        
        # Calculate risk/reward ratio based on trend characteristics
        if trend_direction > 0:  # Uptrend
            risk = reversal_probability * 100
            reward = potential
        elif trend_direction < 0:  # Downtrend
            risk = reversal_probability * 100
            reward = potential
        else:  # Neutral
            risk = 50
            reward = 50
        
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        
        # Generate detailed recommendation
        if trend_direction > 1:  # Strong uptrend
            if trend_duration > 50:
                recommendation = "Strong uptrend but extended; consider partial profit taking"
            else:
                recommendation = "Strong uptrend with good momentum; maintain long positions"
        elif trend_direction > 0:  # Uptrend
            if last_row.get('at_resistance', 0) == 1:
                recommendation = "Uptrend approaching resistance; prepare for potential pullback"
            else:
                recommendation = "Uptrend in progress; hold long positions"
        elif trend_direction < -1:  # Strong downtrend
            if trend_duration > 50:
                recommendation = "Strong downtrend but extended; watch for potential reversal"
            else:
                recommendation = "Strong downtrend; maintain short positions or stay on sidelines"
        elif trend_direction < 0:  # Downtrend
            if last_row.get('at_support', 0) == 1:
                recommendation = "Downtrend approaching support; prepare for potential bounce"
            else:
                recommendation = "Downtrend in progress; maintain short positions or cash"
        else:  # Neutral
            if last_row.get('is_consolidating', 0) == 1:
                recommendation = "Market in consolidation; prepare for potential breakout"
            else:
                recommendation = "Sideways movement without clear direction; wait for trend development"
        
        return {
            'trend_status': trend_status,
            'confidence': confidence,
            'potential': potential,
            'risk_reward': risk_reward,
            'recommendation': recommendation
        }
        
class SignalGenerator:
    """
    Generates trading signals by combining multiple technical indicators.
    
    This class aggregates signals from various technical indicators and applies
    weighting schemes to produce a final trading decision signal.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the SignalGenerator with optional custom weights.
        
        Args:
            weights: Dictionary mapping indicator names to their weights in signal calculation.
                    If None, default equal weights will be used.
        """
        # Default weights if none provided
        self.weights = weights or {
            "macd": 1.0,
            "rsi": 1.0,
            "bollinger": 1.0,
            "adx": 0.8,
            "vwap": 1.2,
            "ema": 1.0,
            "volume": 0.7
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Signal generator initialized with weights: {self.weights}")
    
    def generate_signal(self, indicators: Dict[str, float]) -> Dict:
        """
        Generate a combined trading signal based on multiple indicators.
        
        Args:
            indicators: Dictionary with indicator names as keys and their signal values as values.
                       Signal values should be normalized where:
                       -1.0 = Strong sell
                       -0.5 = Weak sell
                       0.0 = Neutral
                       0.5 = Weak buy
                       1.0 = Strong buy
        
        Returns:
            Dictionary containing the combined signal and additional metadata.
        """
        if not indicators:
            logger.warning("No indicators provided for signal generation")
            return {"signal": 0, "strength": 0, "confidence": 0}
        
        # Calculate weighted signal
        weighted_sum = 0
        total_applied_weight = 0
        
        active_indicators = {}
        for indicator, value in indicators.items():
            if indicator in self.weights:
                weight = self.weights.get(indicator, 0)
                weighted_sum += value * weight
                total_applied_weight += weight
                active_indicators[indicator] = {"value": value, "weight": weight}
            else:
                logger.warning(f"Unknown indicator '{indicator}' ignored in signal generation")
        
        # Avoid division by zero
        if total_applied_weight == 0:
            logger.error("No valid indicators found for signal generation")
            return {"signal": 0, "strength": 0, "confidence": 0, "active_indicators": {}}
        
        # Normalize by actually applied weights
        combined_signal = weighted_sum / total_applied_weight
        
        # Calculate signal strength (absolute value)
        signal_strength = abs(combined_signal)
        
        # Calculate confidence based on agreement between indicators
        signal_agreement = 0
        signal_count = 0
        
        for value in indicators.values():
            if (value > 0 and combined_signal > 0) or (value < 0 and combined_signal < 0):
                signal_agreement += 1
            signal_count += 1
        
        confidence = signal_agreement / signal_count if signal_count > 0 else 0
        
        # Determine final decision
        decision = "hold"
        if combined_signal >= 0.7:
            decision = "strong_buy"
        elif combined_signal >= 0.3:
            decision = "buy"
        elif combined_signal <= -0.7:
            decision = "strong_sell"
        elif combined_signal <= -0.3:
            decision = "sell"
        
        return {
            "signal": combined_signal,
            "strength": signal_strength,
            "confidence": confidence,
            "decision": decision,
            "active_indicators": active_indicators
        }
    
    def calculate_indicator_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate individual indicator signals from price data.
        
        Args:
            data: DataFrame containing OHLCV price data.
            
        Returns:
            Dictionary mapping indicator names to their signal values (-1.0 to 1.0).
        """
        signals = {}
        
        try:
            # MACD signal
            if len(data) >= 26:  # Minimum data needed for MACD
                macd = ta.macd(data['close'])
                macd_signal = macd['MACDh_12_26_9'].iloc[-1]  # MACD histogram
                # Normalize MACD histogram to [-1, 1] range
                max_hist = max(1, abs(macd['MACDh_12_26_9']).max())
                signals["macd"] = max(min(macd_signal / max_hist, 1.0), -1.0)
            
            # RSI signal
            if len(data) >= 14:  # Minimum data needed for RSI
                rsi = ta.rsi(data['close'], length=14).iloc[-1]
                # Convert RSI to a signal: Overbought (>70) = sell, Oversold (<30) = buy
                if rsi > 70:
                    signals["rsi"] = -1.0
                elif rsi < 30:
                    signals["rsi"] = 1.0
                else:
                    # Linear mapping from 30-70 to 1.0 to -1.0
                    signals["rsi"] = 1.0 - 2.0 * (rsi - 30) / 40
            
            # Bollinger Bands signal
            if len(data) >= 20:  # Minimum data needed for Bollinger Bands
                bbands = ta.bbands(data['close'], length=20)
                close = data['close'].iloc[-1]
                upper = bbands['BBU_20_2.0'].iloc[-1]
                lower = bbands['BBL_20_2.0'].iloc[-1]
                middle = bbands['BBM_20_2.0'].iloc[-1]
                
                # Calculate percent from middle to bands
                band_width = upper - lower
                if band_width > 0:
                    # How far price is from middle band, normalized to [-1, 1]
                    signals["bollinger"] = -2.0 * (close - middle) / band_width
                    signals["bollinger"] = max(min(signals["bollinger"], 1.0), -1.0)
            
            # ADX signal
            if len(data) >= 14:  # Minimum data needed for ADX
                adx = ta.adx(data['high'], data['low'], data['close'])
                adx_value = adx['ADX_14'].iloc[-1]
                plus_di = adx['DMP_14'].iloc[-1]
                minus_di = adx['DMN_14'].iloc[-1]
                
                # Strong trend if ADX > 25
                trend_strength = min(adx_value / 50, 1.0)  # Normalize ADX
                
                # Direction based on DI+ vs DI-
                if plus_di > minus_di:
                    signals["adx"] = trend_strength  # Positive/bullish trend
                else:
                    signals["adx"] = -trend_strength  # Negative/bearish trend
            
            # Volume signal
            if len(data) >= 10:
                volume = data['volume']
                avg_volume = volume.rolling(10).mean()
                rel_volume = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
                
                # Price direction
                price_change = data['close'].pct_change().iloc[-1]
                
                # Volume confirms price direction
                volume_signal = 0.0
                if rel_volume > 1.2:  # Higher than average volume
                    volume_signal = 1.0 if price_change > 0 else -1.0
                elif rel_volume < 0.8:  # Lower than average volume
                    volume_signal = 0.5 if price_change > 0 else -0.5
                
                signals["volume"] = volume_signal
            
            # EMA crossing signal
            if len(data) >= 50:
                ema_short = ta.ema(data['close'], length=10).iloc[-1]
                ema_long = ta.ema(data['close'], length=50).iloc[-1]
                
                # Calculate ratio of short to long EMA
                ema_ratio = ema_short / ema_long - 1 if ema_long > 0 else 0
                # Scale to signal range
                signals["ema"] = max(min(ema_ratio * 5, 1.0), -1.0)
            
            # VWAP signal
            if 'volume' in data.columns and len(data) >= 20:
                vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
                close = data['close'].iloc[-1]
                
                # Calculate percent difference from VWAP
                vwap_diff = (close / vwap - 1) if vwap > 0 else 0
                # Scale to signal range
                signals["vwap"] = max(min(vwap_diff * 10, 1.0), -1.0)
                
        except Exception as e:
            logger.error(f"Error calculating indicator signals: {e}")
        
        return signals
    
    def generate_trading_decision(self, data: pd.DataFrame) -> Dict:
        """
        Generate a complete trading decision based on the provided price data.
        
        Args:
            data: DataFrame containing OHLCV price data.
            
        Returns:
            Dictionary containing the final trading decision and supporting data.
        """
        try:
            # Calculate individual indicator signals
            indicator_signals = self.calculate_indicator_signals(data)
            
            # Generate combined signal
            signal_data = self.generate_signal(indicator_signals)
            
            # Add timestamp
            signal_data["timestamp"] = pd.Timestamp.now()
            
            # Add price info
            if not data.empty:
                signal_data["price"] = data['close'].iloc[-1]
                
            return signal_data
            
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return {"signal": 0, "strength": 0, "confidence": 0, "decision": "hold", 
                    "error": str(e), "timestamp": pd.Timestamp.now()}


def backtest_signals(data: pd.DataFrame, signal_generator: SignalGenerator, 
                    threshold: float = 0.5) -> pd.DataFrame:
    """
    Backtest the signal generator on historical data.
    
    Args:
        data: DataFrame containing OHLCV price data.
        signal_generator: Initialized SignalGenerator instance.
        threshold: Signal threshold for trade decisions.
    
    Returns:
        DataFrame with signals and performance metrics.
    """
    results = pd.DataFrame(index=data.index)
    results['close'] = data['close']
    
    # Calculate signals for each period
    signals = []
    decisions = []
    
    for i in range(50, len(data)):  # Start after warmup period
        subset = data.iloc[:i+1]
        signal_data = signal_generator.generate_trading_decision(subset)
        signals.append(signal_data['signal'])
        decisions.append(signal_data['decision'])
    
    # Pad the beginning with NaN values
    pad_length = len(data) - len(signals)
    results['signal'] = pad_length * [np.nan] + signals
    results['decision'] = pad_length * [np.nan] + decisions
    
    # Calculate returns
    results['position'] = np.where(results['signal'] > threshold, 1, 
                         np.where(results['signal'] < -threshold, -1, 0))
    results['position'] = results['position'].ffill().fillna(0)
    
    # Calculate returns
    results['return'] = data['close'].pct_change()
    results['strategy_return'] = results['position'].shift(1) * results['return']
    
    # Calculate cumulative returns
    results['cum_market_return'] = (1 + results['return']).cumprod() - 1
    results['cum_strategy_return'] = (1 + results['strategy_return']).cumprod() - 1
    
    return results  

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


class DatasetPreparation:
    """
    Handles the preparation of financial data for machine learning models.
    
    This class provides methods for cleaning, transforming, engineering features,
    and formatting data for training ML models in the context of day trading.
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 target_column: str = 'return_next_period',
                 test_size: float = 0.2,
                 validation_size: float = 0.15,
                 random_state: int = 42,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1):
        """
        Initialize the dataset preparation pipeline.
        
        Args:
            scaling_method: Method for scaling features ('standard', 'minmax', or None)
            target_column: Column name to be used as the prediction target
            test_size: Proportion of data to be used for testing
            validation_size: Proportion of data to be used for validation
            random_state: Random seed for reproducibility
            sequence_length: Number of time steps for sequence models (LSTM, etc.)
            prediction_horizon: Number of periods ahead to predict
        """
        self.scaling_method = scaling_method
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Initialize scalers as None
        self.feature_scaler = None
        self.target_scaler = None
        
        logger.info(f"Dataset preparation pipeline initialized with {scaling_method} scaling")
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by handling missing values and outliers.
        
        Args:
            data: Raw DataFrame containing OHLCV and indicator data
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return data
        
        logger.info(f"Cleaning dataset with shape {data.shape}")
        
        # Create a copy to avoid modifying the original
        cleaned = data.copy()
        
        # Handle missing values
        initial_missing = cleaned.isna().sum().sum()
        
        # For OHLCV data, forward fill is appropriate
        ohlcv_cols = [col for col in cleaned.columns if col.lower() in 
                     ['open', 'high', 'low', 'close', 'volume']]
        
        cleaned[ohlcv_cols] = cleaned[ohlcv_cols].ffill()
        
        # For other columns, use more sophisticated methods
        # First try forward fill
        cleaned = cleaned.ffill()
        
        # If still have NaNs, use backward fill
        cleaned = cleaned.bfill()
        
        # If still have NaNs after both fills, replace with column median
        for col in cleaned.columns:
            if cleaned[col].isna().any():
                median_val = cleaned[col].median()
                cleaned[col].fillna(median_val, inplace=True)
        
        # Handle infinite values
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # If any NaNs remain after replacing infinities, fill with column median
        for col in cleaned.columns:
            if cleaned[col].isna().any():
                median_val = cleaned[col].dropna().median()
                cleaned[col].fillna(median_val, inplace=True)
        
        final_missing = cleaned.isna().sum().sum()
        logger.info(f"Cleaned missing values: {initial_missing} -> {final_missing}")
        
        # Handle outliers using IQR method for non-OHLCV columns
        indicator_cols = [col for col in cleaned.columns if col not in ohlcv_cols]
        
        for col in indicator_cols:
            if cleaned[col].dtype.kind in 'if':  # Only for numeric columns
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Count outliers before replacement
                outliers = ((cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers instead of removing
                    cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped {outliers} outliers in column {col}")
        
        return cleaned
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from the raw data.
        
        Args:
            data: Cleaned DataFrame containing OHLCV and indicator data
            
        Returns:
            DataFrame with additional engineered features
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return data
        
        logger.info(f"Engineering features for dataset with shape {data.shape}")
        
        # Create a copy to avoid modifying the original
        enhanced = data.copy()
        
        try:
            # Price-based features
            if all(col in enhanced.columns for col in ['open', 'high', 'low', 'close']):
                # Candle body and wick size
                enhanced['body_size'] = abs(enhanced['close'] - enhanced['open']) / enhanced['open']
                enhanced['upper_wick'] = (enhanced['high'] - enhanced[['open', 'close']].max(axis=1)) / enhanced['open']
                enhanced['lower_wick'] = (enhanced[['open', 'close']].min(axis=1) - enhanced['low']) / enhanced['open']
                
                # Volatility features
                enhanced['daily_range'] = (enhanced['high'] - enhanced['low']) / enhanced['open']
                enhanced['close_to_high'] = (enhanced['high'] - enhanced['close']) / enhanced['open']
                enhanced['close_to_low'] = (enhanced['close'] - enhanced['low']) / enhanced['open']
            
            # Time-based features
            if enhanced.index.dtype.kind == 'M':  # Check if index is datetime
                enhanced['hour'] = enhanced.index.hour
                enhanced['minute'] = enhanced.index.minute
                enhanced['day_of_week'] = enhanced.index.dayofweek
                
                # Market session (assuming US market hours)
                enhanced['market_session'] = 0  # Default: after hours
                
                # Pre-market: 4:00 AM to 9:30 AM
                enhanced.loc[(enhanced['hour'] >= 4) & 
                            ((enhanced['hour'] < 9) | 
                            ((enhanced['hour'] == 9) & (enhanced['minute'] < 30))), 
                            'market_session'] = 1
                
                # Regular hours: 9:30 AM to 4:00 PM
                enhanced.loc[(((enhanced['hour'] == 9) & (enhanced['minute'] >= 30)) | 
                            (enhanced['hour'] > 9 & enhanced['hour'] < 16)), 
                            'market_session'] = 2
                
                # After hours: 4:00 PM to 8:00 PM
                enhanced.loc[(enhanced['hour'] >= 16) & (enhanced['hour'] < 20), 
                            'market_session'] = 3
            
            # Return features
            if 'close' in enhanced.columns:
                # Returns over different periods
                enhanced['return_1p'] = enhanced['close'].pct_change(1)
                enhanced['return_5p'] = enhanced['close'].pct_change(5)
                enhanced['return_10p'] = enhanced['close'].pct_change(10)
                enhanced['return_20p'] = enhanced['close'].pct_change(20)
                
                # Future returns (target variables)
                enhanced['return_next_period'] = enhanced['close'].pct_change(1).shift(-1)
                enhanced['return_next_5p'] = enhanced['close'].pct_change(5).shift(-5)
                enhanced['return_next_10p'] = enhanced['close'].pct_change(10).shift(-10)
                
                # Return volatility
                enhanced['return_volatility_5p'] = enhanced['return_1p'].rolling(5).std()
                enhanced['return_volatility_10p'] = enhanced['return_1p'].rolling(10).std()
            
            # Volume-based features
            if 'volume' in enhanced.columns:
                enhanced['volume_ma5'] = enhanced['volume'].rolling(5).mean()
                enhanced['volume_ma10'] = enhanced['volume'].rolling(10).mean()
                enhanced['relative_volume'] = enhanced['volume'] / enhanced['volume_ma10']
                
                # Money flow
                if all(col in enhanced.columns for col in ['close', 'high', 'low']):
                    typical_price = (enhanced['high'] + enhanced['low'] + enhanced['close']) / 3
                    enhanced['money_flow'] = typical_price * enhanced['volume']
                    enhanced['money_flow_10p'] = enhanced['money_flow'].rolling(10).sum()
                
                # Price-volume correlation
                if 'return_1p' in enhanced.columns:
                    enhanced['price_volume_corr_5p'] = enhanced['return_1p'].rolling(5).corr(
                        enhanced['volume'].pct_change(1))
            
            # Technical indicator interaction features
            indicator_cols = [col for col in enhanced.columns if col.lower() in 
                             ['rsi', 'macd', 'adx', 'cci', 'atr', 'obv']]
            
            if len(indicator_cols) >= 2:
                # Create pairwise ratios for key indicators
                for i, ind1 in enumerate(indicator_cols):
                    for ind2 in indicator_cols[i+1:]:
                        if all(col in enhanced.columns for col in [ind1, ind2]):
                            # Avoid division by zero
                            denom = enhanced[ind2].replace(0, np.nan)
                            enhanced[f'{ind1}_to_{ind2}_ratio'] = enhanced[ind1] / denom
                            
                            # Replace infinities and fill NaNs
                            enhanced[f'{ind1}_to_{ind2}_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
                            enhanced[f'{ind1}_to_{ind2}_ratio'].fillna(enhanced[f'{ind1}_to_{ind2}_ratio'].median(), inplace=True)
            
            # Signal convergence/divergence
            # Check if common indicators are present
            rsi_present = any('rsi' in col.lower() for col in enhanced.columns)
            macd_present = any('macd' in col.lower() for col in enhanced.columns)
            bbands_present = any('bband' in col.lower() or 'bb_' in col.lower() for col in enhanced.columns)
            
            # Create composite signal features if indicators are present
            if rsi_present and macd_present:
                rsi_col = next((col for col in enhanced.columns if 'rsi' in col.lower()), None)
                macd_col = next((col for col in enhanced.columns if 'macd' == col.lower() or 
                                'macd_hist' in col.lower() or 'macdh' in col.lower()), None)
                
                if rsi_col and macd_col:
                    # Normalize columns for comparison
                    enhanced['rsi_norm'] = (enhanced[rsi_col] - 50) / 50  # Center RSI around 0
                    
                    # Normalize MACD using its standard deviation
                    macd_std = enhanced[macd_col].rolling(20).std()
                    enhanced['macd_norm'] = enhanced[macd_col] / (macd_std.replace(0, 1))
                    
                    # Signal agreement
                    enhanced['rsi_macd_agreement'] = enhanced['rsi_norm'] * enhanced['macd_norm']
                    
                    # Clean up temporary columns
                    enhanced.drop(['rsi_norm', 'macd_norm'], axis=1, inplace=True)
            
            logger.info(f"Feature engineering complete, new shape: {enhanced.shape}")
            return enhanced
        
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            # Return original data if error occurs
            return data
    
    def prepare_for_ml(self, 
                       data: pd.DataFrame, 
                       feature_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[List[str]] = None) -> Dict:
        """
        Prepare data for machine learning model training and evaluation.
        
        Args:
            data: DataFrame with features and target
            feature_columns: List of columns to use as features (if None, uses all except target)
            categorical_columns: List of categorical columns requiring encoding
            
        Returns:
            Dictionary containing train/validation/test splits and related metadata
        """
        if data.empty:
            logger.warning("Empty DataFrame provided for ML preparation")
            return {}
        
        logger.info(f"Preparing dataset with shape {data.shape} for ML models")
        
        # Create a copy to avoid modifying the original
        prepared_data = data.copy()
        
        # Drop rows with NaN in target column
        if self.target_column in prepared_data.columns:
            original_len = len(prepared_data)
            prepared_data = prepared_data.dropna(subset=[self.target_column])
            dropped_rows = original_len - len(prepared_data)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing target values")
        
        # Handle feature columns
        if feature_columns is None:
            # Use all columns except target and datetime cols
            feature_columns = [col for col in prepared_data.columns 
                              if col != self.target_column and 
                              prepared_data[col].dtype.kind in 'bifc']  # bool, int, float, complex
        
        logger.info(f"Using {len(feature_columns)} feature columns")
        
        # Handle categorical columns
        if categorical_columns:
            # One-hot encode categorical variables
            prepared_data = pd.get_dummies(prepared_data, columns=categorical_columns, 
                                         drop_first=True)
            logger.info(f"One-hot encoded {len(categorical_columns)} categorical columns")
            
            # Update feature columns after encoding
            new_columns = set(prepared_data.columns) - set(data.columns) + set(feature_columns) - set(categorical_columns)
            feature_columns = list(new_columns - {self.target_column})
        
        # Extract features and target
        X = prepared_data[feature_columns]
        y = prepared_data[self.target_column] if self.target_column in prepared_data.columns else None
        
        # Handle feature scaling
        if self.scaling_method:
            try:
                if self.scaling_method.lower() == 'standard':
                    self.feature_scaler = StandardScaler()
                elif self.scaling_method.lower() == 'minmax':
                    self.feature_scaler = MinMaxScaler()
                
                if self.feature_scaler:
                    X_scaled = self.feature_scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                    logger.info(f"Applied {self.scaling_method} scaling to features")
            
            except Exception as e:
                logger.error(f"Error during feature scaling: {e}")
                # Continue with unscaled features if scaling fails
            
            # Scale target for regression problems if it's numeric
            if y is not None and y.dtype.kind in 'if':
                try:
                    self.target_scaler = MinMaxScaler()
                    y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
                    y = pd.Series(y_scaled, index=y.index)
                    logger.info("Applied scaling to target variable")
                except Exception as e:
                    logger.error(f"Error during target scaling: {e}")
        
        # Split data into train, validation, and test sets
        result = {}
        
        if y is not None:
            try:
                # First split to get train+val and test
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
                )
                
                # Then split train+val to get train and val
                val_size_adjusted = self.validation_size / (1 - self.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_size_adjusted, 
                    random_state=self.random_state, shuffle=False
                )
                
                result['X_train'] = X_train
                result['y_train'] = y_train
                result['X_val'] = X_val
                result['y_val'] = y_val
                result['X_test'] = X_test
                result['y_test'] = y_test
                
                logger.info(f"Data split into train ({len(X_train)}), "
                           f"validation ({len(X_val)}), and test ({len(X_test)}) sets")
                
            except Exception as e:
                logger.error(f"Error during data splitting: {e}")
                # Return unsplit data if an error occurs
                result['X'] = X
                result['y'] = y
        else:
            # If no target is provided, just return the features
            result['X'] = X
        
        # Add metadata
        result['feature_columns'] = feature_columns
        result['feature_scaler'] = self.feature_scaler
        result['target_scaler'] = self.target_scaler
        result['target_column'] = self.target_column
        
        return result
    
    def prepare_sequences(self, data_dict: Dict) -> Dict:
        """
        Convert data into sequences for time series models like LSTM.
        
        Args:
            data_dict: Dictionary containing train/val/test splits
            
        Returns:
            Dictionary with sequence data for time series models
        """
        sequence_data = {}
        
        # Function to create sequences
        def create_sequences(features, targets=None):
            X_seq = []
            y_seq = []
            
            for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
                # Extract sequence
                X_seq.append(features[i:(i + self.sequence_length)].values)
                
                if targets is not None:
                    # Extract target (future value)
                    y_seq.append(targets[i + self.sequence_length + self.prediction_horizon - 1])
            
            return np.array(X_seq), np.array(y_seq) if targets is not None else None
        
        try:
            # Process training data
            if 'X_train' in data_dict and 'y_train' in data_dict:
                X_train_seq, y_train_seq = create_sequences(
                    data_dict['X_train'], data_dict['y_train'])
                sequence_data['X_train_seq'] = X_train_seq
                sequence_data['y_train_seq'] = y_train_seq
                
                logger.info(f"Created training sequences with shape {X_train_seq.shape}")
            
            # Process validation data
            if 'X_val' in data_dict and 'y_val' in data_dict:
                X_val_seq, y_val_seq = create_sequences(
                    data_dict['X_val'], data_dict['y_val'])
                sequence_data['X_val_seq'] = X_val_seq
                sequence_data['y_val_seq'] = y_val_seq
                
                logger.info(f"Created validation sequences with shape {X_val_seq.shape}")
            
            # Process test data
            if 'X_test' in data_dict and 'y_test' in data_dict:
                X_test_seq, y_test_seq = create_sequences(
                    data_dict['X_test'], data_dict['y_test'])
                sequence_data['X_test_seq'] = X_test_seq
                sequence_data['y_test_seq'] = y_test_seq
                
                logger.info(f"Created test sequences with shape {X_test_seq.shape}")
            
            # Add metadata
            sequence_data['sequence_length'] = self.sequence_length
            sequence_data['prediction_horizon'] = self.prediction_horizon
            sequence_data['feature_columns'] = data_dict.get('feature_columns')
            sequence_data['feature_scaler'] = data_dict.get('feature_scaler')
            sequence_data['target_scaler'] = data_dict.get('target_scaler')
            
            return sequence_data
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return {}
    
    def prepare_dataset(self, 
                       data: pd.DataFrame, 
                       feature_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[List[str]] = None,
                       create_sequences: bool = False) -> Dict:
        """
        Complete pipeline for preparing datasets for ML models.
        
        Args:
            data: Raw DataFrame with OHLCV and indicator data
            feature_columns: List of columns to use as features
            categorical_columns: List of categorical columns requiring encoding
            create_sequences: Whether to create sequences for time series models
            
        Returns:
            Dictionary containing prepared data for ML model training
        """
        try:
            # Step 1: Clean the data
            cleaned_data = self.clean_data(data)
            
            # Step 2: Engineer features
            enhanced_data = self.engineer_features(cleaned_data)
            
            # Step 3: Prepare for ML
            ml_data = self.prepare_for_ml(
                enhanced_data, feature_columns, categorical_columns)
            
            # Step 4: Create sequences if requested
            if create_sequences:
                sequence_data = self.prepare_sequences(ml_data)
                return {**ml_data, **sequence_data}
            
            return ml_data
            
        except Exception as e:
            logger.error(f"Error in dataset preparation pipeline: {e}")
            return {}


def create_feature_importance_dataset(data: pd.DataFrame, 
                                     signal_generator, 
                                     window_sizes: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Create a dataset with technical indicators and their signals for feature importance analysis.
    
    Args:
        data: DataFrame containing OHLCV data
        signal_generator: Instance of SignalGenerator class
        window_sizes: List of window sizes for rolling features
        
    Returns:
        DataFrame with all features and target variables
    """
    logger.info("Creating feature importance dataset")
    
    result_df = data.copy()
    
    try:
        # Calculate basic indicators and signals
        for i in range(max(window_sizes) + 20, len(data)):
            subset = data.iloc[:i+1]
            signal_data = signal_generator.calculate_indicator_signals(subset)
            
            # Add each indicator signal to the result dataframe
            for indicator, value in signal_data.items():
                result_df.loc[subset.index[-1], f'signal_{indicator}'] = value
        
        # Calculate future returns (target variables)
        for window in window_sizes:
            result_df[f'future_return_{window}'] = result_df['close'].pct_change(window).shift(-window)
            
            # Create binary target (1 for positive return, 0 for negative)
            result_df[f'future_direction_{window}'] = (result_df[f'future_return_{window}'] > 0).astype(int)
        
        # Drop rows with NaN values
        result_df = result_df.dropna()
        
        logger.info(f"Created feature importance dataset with shape {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error creating feature importance dataset: {e}")
        return pd.DataFrame()      
    
    