"""
Technical Indicators Module for Day Trading Bot.

This module calculates various technical indicators used for trading decisions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        Add simple moving averages for specified periods, showing the curve from the beginning.

        Args:
            df: DataFrame with market data
            periods: List of periods for SMA calculation
            price_column: Column to use for calculations

        Returns:
            DataFrame with added moving average columns
        """
        if df.empty or price_column not in df.columns:
            logger.warning(
                f"Cannot calculate SMA: Empty DataFrame or missing '{price_column}' column")
            return df

        result_df = df.copy()

        for period in periods:
            result_df[f'sma_{period}'] = result_df[price_column].rolling(
                window=period, min_periods=1).mean()

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
            logger.warning(
                f"Cannot calculate EMA: Empty DataFrame or missing '{price_column}' column")
            return df

        result_df = df.copy()

        for period in periods:
            result_df[f'ema_{period}'] = ta.ema(
                result_df[price_column], length=period)

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
            logger.warning(
                f"Cannot calculate Bollinger Bands: Empty DataFrame or missing '{price_column}' column")
            return df

        result_df = df.copy()

        # Calculate Bollinger Bands
        bbands = ta.bbands(result_df[price_column], length=period, std=std_dev)

        # Add the columns to the DataFrame
        result_df['bb_upper'] = bbands['BBU_'+str(period)+'_'+str(std_dev)]
        result_df['bb_middle'] = bbands['BBM_'+str(period)+'_'+str(std_dev)]
        result_df['bb_lower'] = bbands['BBL_'+str(period)+'_'+str(std_dev)]

        # Calculate bandwidth and percent B
        result_df['bb_bandwidth'] = (
            result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_percent'] = (result_df[price_column] - result_df['bb_lower']) / (
            result_df['bb_upper'] - result_df['bb_lower'])

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
            logger.warning(
                f"Cannot calculate RSI: Empty DataFrame or missing '{price_column}' column")
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
            logger.warning(
                f"Cannot calculate MACD: Empty DataFrame or missing '{price_column}' column")
            return df

        # Check if enough data points exist for calculation
        if len(df) < max(fast_period, slow_period, signal_period):
            logger.warning(
                f"Cannot calculate MACD: Not enough data points. Need at least {max(fast_period, slow_period, signal_period)}, but got {len(df)}.")
            return df

        result_df = df.copy()

        try:
            # Calculate MACD
            macd = ta.macd(
                result_df[price_column], fast=fast_period, slow=slow_period, signal=signal_period)

            # Check if MACD calculation returned valid results
            if macd is None:
                logger.warning(
                    f"MACD calculation returned None. Using manual calculation as fallback.")
                # Manual calculation as fallback
                fast_ema = result_df[price_column].ewm(
                    span=fast_period, adjust=False).mean()
                slow_ema = result_df[price_column].ewm(
                    span=slow_period, adjust=False).mean()
                result_df['macd'] = fast_ema - slow_ema
                result_df['macd_signal'] = result_df['macd'].ewm(
                    span=signal_period, adjust=False).mean()
                result_df['macd_histogram'] = result_df['macd'] - \
                    result_df['macd_signal']
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
                    logger.warning(
                        f"Error processing MACD results: {e}. Using manual calculation.")
                    # Manual calculation as fallback
                    fast_ema = result_df[price_column].ewm(
                        span=fast_period, adjust=False).mean()
                    slow_ema = result_df[price_column].ewm(
                        span=slow_period, adjust=False).mean()
                    result_df['macd'] = fast_ema - slow_ema
                    result_df['macd_signal'] = result_df['macd'].ewm(
                        span=signal_period, adjust=False).mean()
                    result_df['macd_histogram'] = result_df['macd'] - \
                        result_df['macd_signal']

            # Add MACD signal
            result_df['macd_bullish'] = (
                result_df['macd'] > result_df['macd_signal']).astype(int)
            result_df['macd_bearish'] = (
                result_df['macd'] < result_df['macd_signal']).astype(int)

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
            logger.warning(
                "Cannot calculate Stochastic Oscillator: Empty DataFrame or missing required columns")
            return df

        result_df = df.copy()

        # Calculate Stochastic Oscillator
        stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'],
                         k=k_period, d=d_period, smooth_k=smooth_k)

        # Add the columns to the DataFrame
        result_df['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{smooth_k}']
        result_df['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{smooth_k}']

        # Add Stochastic signals
        result_df['stoch_oversold'] = ((result_df['stoch_k'] < 20) & (
            result_df['stoch_d'] < 20)).astype(int)
        result_df['stoch_overbought'] = (
            (result_df['stoch_k'] > 80) & (result_df['stoch_d'] > 80)).astype(int)
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
            logger.warning(
                "Cannot calculate ATR: Empty DataFrame or missing required columns")
            return df

        result_df = df.copy()

        # Calculate ATR
        result_df['atr'] = ta.atr(
            result_df['high'], result_df['low'], result_df['close'], length=period)

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
            logger.warning(
                f"Cannot calculate OBV: Empty DataFrame or missing required columns")
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
            logger.warning(
                "Cannot calculate Ichimoku Cloud: Empty DataFrame or missing required columns")
            return df

        # Check if we have enough data
        if len(df) < max(conversion_period, base_period, lagging_span_period, displacement):
            logger.warning(
                f"Not enough data for Ichimoku Cloud. Need at least {max(conversion_period, base_period, lagging_span_period, displacement)} points, but got {len(df)}.")
            return df

        result_df = df.copy()

        try:
            # Calculate Ichimoku Cloud components manually if needed
            # This is a fallback in case pandas_ta has issues
            if True:  # Always use manual calculation for consistency
                # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
                high_9 = result_df['high'].rolling(
                    window=conversion_period).max()
                low_9 = result_df['low'].rolling(
                    window=conversion_period).min()
                result_df['tenkan_sen'] = (high_9 + low_9) / 2

                # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
                high_26 = result_df['high'].rolling(window=base_period).max()
                low_26 = result_df['low'].rolling(window=base_period).min()
                result_df['kijun_sen'] = (high_26 + low_26) / 2

                # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 displaced forward 26 periods
                result_df['senkou_span_a'] = (
                    (result_df['tenkan_sen'] + result_df['kijun_sen']) / 2).shift(displacement)

                # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, displaced forward 26 periods
                high_52 = result_df['high'].rolling(
                    window=lagging_span_period).max()
                low_52 = result_df['low'].rolling(
                    window=lagging_span_period).min()
                result_df['senkou_span_b'] = (
                    (high_52 + low_52) / 2).shift(displacement)

                # Chikou Span (Lagging Span): Close price shifted backwards 26 periods
                result_df['chikou_span'] = result_df['close'].shift(
                    -displacement)

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
                        # Conversion line
                        result_df['tenkan_sen'] = ichimoku[0]
                        result_df['kijun_sen'] = ichimoku[1]   # Base line
                        # Leading span A
                        result_df['senkou_span_a'] = ichimoku[2]
                        # Leading span B
                        result_df['senkou_span_b'] = ichimoku[3]
                        result_df['chikou_span'] = ichimoku[4]   # Lagging span
                    else:
                        logger.warning(
                            "Ichimoku Cloud tuple has unexpected length")
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
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(
                f"Missing required columns for some indicators: {missing_columns}")

        # Add all indicators
        result_df = df.copy()

        # Moving averages
        result_df = TechnicalIndicators.add_moving_averages(result_df)
        result_df = TechnicalIndicators.add_exponential_moving_averages(
            result_df)

        # Bollinger Bands
        result_df = TechnicalIndicators.add_bollinger_bands(result_df)

        # RSI
        result_df = TechnicalIndicators.add_rsi(result_df)

        # MACD
        result_df = TechnicalIndicators.add_macd(result_df)

        # Other indicators
        if all(col in df.columns for col in ['high', 'low']):
            result_df = TechnicalIndicators.add_stochastic_oscillator(
                result_df)
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
            logger.warning(
                "Cannot recognize patterns: Empty DataFrame or missing required columns")
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
        result_df['pattern_morning_star'] = ((result_df['close'].shift(2) < result_df['open'].shift(2)) &  # First day bearish
                                             (abs(result_df['close'].shift(1) - result_df['open'].shift(1)) /
                                              (result_df['high'].shift(1) - result_df['low'].shift(1)) < 0.1) &  # Second day doji
                                             # Third day bullish
                                             (result_df['close'] > result_df['open']) &
                                             (result_df['close'] > (result_df['close'].shift(
                                                 2) + result_df['open'].shift(2)) / 2)
                                             ).astype(int)

        # Evening Star pattern (bearish)
        result_df['pattern_evening_star'] = ((result_df['close'].shift(2) > result_df['open'].shift(2)) &  # First day bullish
                                             (abs(result_df['close'].shift(1) - result_df['open'].shift(1)) /
                                              (result_df['high'].shift(1) - result_df['low'].shift(1)) < 0.1) &  # Second day doji
                                             # Third day bearish
                                             (result_df['close'] < result_df['open']) &
                                             (result_df['close'] < (result_df['close'].shift(
                                                 2) + result_df['open'].shift(2)) / 2)
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
            logger.warning(
                f"Cannot detect support/resistance: Empty DataFrame or missing '{price_column}' column")
            return df

        result_df = df.copy()

        # Find local minima and maxima
        result_df['local_min'] = result_df[price_column].rolling(
            window=window, center=True).min() == result_df[price_column]
        result_df['local_max'] = result_df[price_column].rolling(
            window=window, center=True).max() == result_df[price_column]

        # Initialize support and resistance columns
        result_df['support_level'] = np.nan
        result_df['resistance_level'] = np.nan

        # Extract support and resistance levels
        support_levels = result_df.loc[result_df['local_min'], price_column].tolist(
        )
        resistance_levels = result_df.loc[result_df['local_max'], price_column].tolist(
        )

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
        result_df['at_support'] = (
            ~result_df['support_level'].isna()).astype(int)
        result_df['at_resistance'] = (
            ~result_df['resistance_level'].isna()).astype(int)

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
            logger.warning(
                f"Cannot detect trend: Empty DataFrame or missing '{price_column}' column")
            return df

        result_df = df.copy()

        # Check if we have enough data for the required moving averages
        if len(df) < max(short_period, long_period):
            logger.warning(
                f"Not enough data for trend detection. Need at least {max(short_period, long_period)} points, but got {len(df)}.")
            # Use shorter periods if we don't have enough data
            adjusted_short_period = min(short_period, len(df) // 3)
            adjusted_long_period = min(long_period, len(df) // 2)

            if adjusted_short_period >= 5 and adjusted_long_period > adjusted_short_period:
                logger.info(
                    f"Using adjusted periods: short_period={adjusted_short_period}, long_period={adjusted_long_period}")
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
                result_df[f'sma_{short_period}'] = ta.sma(
                    result_df[price_column], length=short_period)
            except Exception as e:
                logger.warning(
                    f"Error calculating SMA-{short_period}: {e}. Using pandas implementation.")
                result_df[f'sma_{short_period}'] = result_df[price_column].rolling(
                    window=short_period).mean()

        if f'sma_{long_period}' not in result_df.columns:
            try:
                result_df[f'sma_{long_period}'] = ta.sma(
                    result_df[price_column], length=long_period)
            except Exception as e:
                logger.warning(
                    f"Error calculating SMA-{long_period}: {e}. Using pandas implementation.")
                result_df[f'sma_{long_period}'] = result_df[price_column].rolling(
                    window=long_period).mean()

        # Handle NaN values in moving averages
        result_df[f'sma_{short_period}'] = result_df[f'sma_{short_period}'].fillna(
            result_df[price_column])
        result_df[f'sma_{long_period}'] = result_df[f'sma_{long_period}'].fillna(
            result_df[price_column])

        # Determine trend based on moving average relationship
        result_df['uptrend'] = (
            result_df[f'sma_{short_period}'] > result_df[f'sma_{long_period}']).astype(int)
        result_df['downtrend'] = (
            result_df[f'sma_{short_period}'] < result_df[f'sma_{long_period}']).astype(int)

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


class SignalGeneration:
    """Class for generating trading signals by combining multiple technical indicators."""

    @staticmethod
    def combine_signals(
        df: pd.DataFrame,
        rsi_thresholds: Dict[str, float] = {'oversold': 30, 'overbought': 70},
        macd_threshold: float = 0,
        volume_factor: float = 1.5,
        trend_strength_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Combine multiple technical indicators to generate trading signals.

        Args:
            df: DataFrame with technical indicators
            rsi_thresholds: Dict with oversold and overbought RSI levels
            macd_threshold: MACD threshold for signal strength
            volume_factor: Factor above average volume to consider significant
            trend_strength_threshold: Minimum trend strength to consider valid

        Returns:
            DataFrame with combined signal columns
        """
        if df.empty:
            logger.warning("Cannot generate signals: Empty DataFrame")
            return df

        result_df = df.copy()

        # Initialize signal columns
        result_df['buy_signal'] = 0
        result_df['sell_signal'] = 0
        result_df['signal_strength'] = 0

        try:
            # Volume significance
            result_df['volume_significant'] = (
                result_df['volume'] > result_df['volume'].rolling(
                    window=20).mean() * volume_factor
            ).astype(int)

            # Trend confirmation
            trend_confirmed = (
                (result_df['trend_strength'] > trend_strength_threshold) &
                ((result_df['uptrend'] == 1) | (result_df['downtrend'] == 1))
            )

            # Combined buy signals
            buy_conditions = [
                # RSI oversold
                result_df['rsi'] < rsi_thresholds['oversold'],
                # MACD bullish crossover
                result_df['macd'] > result_df['macd_signal'],
                # Price near support
                result_df['at_support'] == 1,
                # Bullish candlestick patterns
                result_df['pattern_bullish_engulfing'] |
                result_df['pattern_hammer'] |
                result_df['pattern_morning_star'],
                # Bollinger Band conditions
                result_df['close'] < result_df['bb_lower'],
                # Volume confirmation
                result_df['volume_significant'] == 1,
                # Ichimoku bullish
                result_df['ichimoku_bullish'] == 1
            ]

            # Combined sell signals
            sell_conditions = [
                # RSI overbought
                result_df['rsi'] > rsi_thresholds['overbought'],
                # MACD bearish crossover
                result_df['macd'] < result_df['macd_signal'],
                # Price near resistance
                result_df['at_resistance'] == 1,
                # Bearish candlestick patterns
                result_df['pattern_bearish_engulfing'] |
                result_df['pattern_shooting_star'] |
                result_df['pattern_evening_star'],
                # Bollinger Band conditions
                result_df['close'] > result_df['bb_upper'],
                # Volume confirmation
                result_df['volume_significant'] == 1,
                # Ichimoku bearish
                result_df['ichimoku_bearish'] == 1
            ]

            # Calculate signal strength (number of conditions met)
            buy_strength = sum(condition.astype(int)
                               for condition in buy_conditions)
            sell_strength = sum(condition.astype(int)
                                for condition in sell_conditions)

            # Generate final signals with strength threshold
            min_conditions = 3  # Minimum number of conditions that must be met
            result_df.loc[buy_strength >= min_conditions, 'buy_signal'] = 1
            result_df.loc[sell_strength >= min_conditions, 'sell_signal'] = 1

            # Calculate overall signal strength (-100 to 100)
            result_df['signal_strength'] = (
                (buy_strength - sell_strength) * 100 / len(buy_conditions)
            ).clip(-100, 100)

            # Add confidence metrics
            result_df['signal_confidence'] = abs(
                result_df['signal_strength']) / 100
            result_df['trend_alignment'] = (
                (result_df['buy_signal'] & result_df['uptrend']) |
                (result_df['sell_signal'] & result_df['downtrend'])
            ).astype(int)

            # Filter signals by trend confirmation
            result_df.loc[~trend_confirmed, ['buy_signal', 'sell_signal']] = 0

            # Add trade risk metrics
            result_df['risk_level'] = (
                result_df['atr_percent'] *
                # Higher confidence = lower risk
                (2 - result_df['signal_confidence'])
            ).clip(0, 100)

            # Potential profit targets based on ATR and signal strength
            result_df['profit_target_percent'] = (
                result_df['atr_percent'] *
                (1 + result_df['signal_confidence']) *
                2  # 2x ATR for initial target
            )

            # Stop loss levels based on ATR
            result_df['stop_loss_percent'] = result_df['atr_percent']

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            # Keep the initialized zero values for signals if there's an error

        return result_df

    @staticmethod
    def generate_trade_recommendations(
        df: pd.DataFrame,
        min_confidence: float = 0.7,
        max_risk_percent: float = 5.0
    ) -> pd.DataFrame:
        """
        Generate specific trade recommendations based on signals.

        Args:
            df: DataFrame with signal data
            min_confidence: Minimum signal confidence required
            max_risk_percent: Maximum acceptable risk percentage

        Returns:
            DataFrame with trade recommendations
        """
        if df.empty:
            logger.warning("Cannot generate recommendations: Empty DataFrame")
            return df

        result_df = df.copy()

        try:
            # Initialize recommendation columns
            result_df['trade_recommendation'] = 'HOLD'
            result_df['recommendation_reason'] = ''
            result_df['position_size_factor'] = 0.0

            # Generate recommendations for high-confidence signals
            mask_buy = (
                (result_df['buy_signal'] == 1) &
                (result_df['signal_confidence'] >= min_confidence) &
                (result_df['risk_level'] <= max_risk_percent)
            )
            mask_sell = (
                (result_df['sell_signal'] == 1) &
                (result_df['signal_confidence'] >= min_confidence) &
                (result_df['risk_level'] <= max_risk_percent)
            )

            # Set trade recommendations
            result_df.loc[mask_buy, 'trade_recommendation'] = 'BUY'
            result_df.loc[mask_sell, 'trade_recommendation'] = 'SELL'

            # Calculate position size factor (0.0 to 1.0) based on confidence and risk
            result_df['position_size_factor'] = (
                result_df['signal_confidence'] *
                (1 - result_df['risk_level'] / 100)
            ).clip(0, 1)

            # Generate detailed reasons for recommendations
            def generate_reason(row):
                if row['trade_recommendation'] == 'HOLD':
                    return 'No clear signal or risk too high'

                reasons = []
                if row['trend_alignment'] == 1:
                    reasons.append('Trend aligned')
                if row['volume_significant'] == 1:
                    reasons.append('Volume confirmed')
                if row['trade_recommendation'] == 'BUY':
                    if row['rsi'] < 30:
                        reasons.append('Oversold')
                    if row['at_support'] == 1:
                        reasons.append('At support')
                elif row['trade_recommendation'] == 'SELL':
                    if row['rsi'] > 70:
                        reasons.append('Overbought')
                    if row['at_resistance'] == 1:
                        reasons.append('At resistance')

                return '; '.join(reasons) if reasons else 'Multiple indicators aligned'

            result_df['recommendation_reason'] = result_df.apply(
                generate_reason, axis=1)

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Keep the initialized values if there's an error

        return result_df


class DatasetPreparation:
    """
    Class for preparing datasets for machine learning models.
    Includes advanced feature engineering capabilities.
    """

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw market data.

        Args:
            df: DataFrame with market data (OHLCV and indicators)

        Returns:
            DataFrame with additional engineered features
        """
        if df.empty:
            logger.warning("Cannot engineer features: Empty DataFrame")
            return df

        result_df = df.copy()

        try:
            # Price action features
            result_df['body_size'] = abs(
                result_df['close'] - result_df['open']) / result_df['open']
            result_df['upper_shadow'] = (
                result_df['high'] - result_df[['open', 'close']].max(axis=1)) / result_df['open']
            result_df['lower_shadow'] = (result_df[['open', 'close']].min(
                axis=1) - result_df['low']) / result_df['open']
            result_df['price_range'] = (
                result_df['high'] - result_df['low']) / result_df['open']

            # Volume features
            result_df['volume_price_trend'] = result_df['volume'] * \
                (result_df['close'] - result_df['open'])
            result_df['volume_ma_ratio'] = result_df['volume'] / \
                result_df['volume'].rolling(20).mean()

            # Technical indicator interaction features
            if 'rsi' in result_df.columns and 'volume' in result_df.columns:
                result_df['rsi_volume'] = result_df['rsi'] * \
                    result_df['volume_ma_ratio']

            if 'macd' in result_df.columns and 'macd_signal' in result_df.columns:
                result_df['macd_cross'] = result_df['macd'] - \
                    result_df['macd_signal']
                result_df['macd_cross_slope'] = result_df['macd_cross'].diff()

            # Trend features
            if all(col in result_df.columns for col in ['sma_20', 'sma_50']):
                result_df['trend_strength'] = (
                    result_df['sma_20'] - result_df['sma_50']) / result_df['sma_50']
                result_df['price_to_sma_ratio'] = result_df['close'] / \
                    result_df['sma_20']

            # Volatility features
            if 'bb_upper' in result_df.columns and 'bb_lower' in result_df.columns:
                result_df['bb_squeeze'] = (
                    result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
                result_df['bb_position'] = (
                    result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])

            # Time series features
            result_df['return'] = result_df['close'].pct_change()
            result_df['log_return'] = np.log1p(result_df['return'])
            for window in [5, 10, 20]:
                result_df[f'volatility_{window}d'] = result_df['return'].rolling(
                    window).std()
                result_df[f'momentum_{window}d'] = result_df['close'].pct_change(
                    window)

            # Combine multiple indicators
            if all(col in result_df.columns for col in ['rsi', 'macd', 'bb_position']):
                result_df['combined_signal'] = (
                    (result_df['rsi'] - 50) / 50 +  # Normalize RSI around 0
                    # Normalize MACD
                    result_df['macd_cross'] / result_df['close'] +
                    (result_df['bb_position'] - 0.5) *
                    2  # Normalize BB position
                ) / 3

        except Exception as e:
            logger.error(f"Error engineering features: {e}")

        return result_df

    @staticmethod
    def create_sequence_features(
        df: pd.DataFrame,
        sequence_length: int = 10,
        target_column: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence features for time series models (e.g., LSTM).

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps to include in each sequence
            target_column: Column to use as prediction target

        Returns:
            Tuple of (X, y) where X is 3D array of sequences and y is target values
        """
        if df.empty:
            logger.warning("Cannot create sequences: Empty DataFrame")
            return np.array([]), np.array([])

        try:
            # Create sequences
            sequences = []
            targets = []

            for i in range(len(df) - sequence_length):
                sequence = df.iloc[i:(i + sequence_length)]
                target = df[target_column].iloc[i + sequence_length]
                sequences.append(sequence.values)
                targets.append(target)

            return np.array(sequences), np.array(targets)

        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    @staticmethod
    def create_target_labels(
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create classification labels based on future returns.

        Args:
            df: DataFrame with market data
            horizon: Number of periods to look ahead
            threshold: Return threshold for buy/sell signals

        Returns:
            Series with categorical labels (1: Buy, 0: Hold, -1: Sell)
        """
        if df.empty or 'close' not in df.columns:
            logger.warning(
                "Cannot create labels: Empty DataFrame or missing close prices")
            return pd.Series()

        try:
            # Calculate future returns
            future_returns = df['close'].shift(-horizon) / df['close'] - 1

            # Create labels based on threshold
            labels = pd.Series(0, index=df.index)  # Default to hold
            labels[future_returns > threshold] = 1  # Buy signal
            labels[future_returns < -threshold] = -1  # Sell signal

            return labels

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
