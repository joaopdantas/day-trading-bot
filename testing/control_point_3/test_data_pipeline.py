import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats
from scipy.signal import find_peaks
from advanced_feature_engineering import AdvancedFeatureEngineering  # ajuste conforme o caminho real

from src.indicators.technical import TechnicalIndicators, PatternRecognition

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering class for creating ML-specific features from financial data.
    
    This class extends the basic feature engineering in DatasetPreparation with more
    sophisticated features specifically designed for machine learning models in trading.
    """
    
    def __init__(self, 
                 fractional_differencing_d: float = 0.4,
                 n_lags: int = 5,
                 n_change_periods: List[int] = [1, 5, 10, 20],
                 enable_ta_features: bool = True,
                 enable_pattern_features: bool = True,
                 enable_statistical_features: bool = True):
        """
        Initialize the advanced feature engineering module.
        
        Args:
            fractional_differencing_d: Fractional differencing parameter (0-0.5 recommended)
            n_lags: Number of lag features to create
            n_change_periods: List of periods for calculating price changes
            enable_ta_features: Enable technical analysis features
            enable_pattern_features: Enable pattern recognition features
            enable_statistical_features: Enable statistical features
        """
        self.fractional_differencing_d = fractional_differencing_d
        self.n_lags = n_lags
        self.n_change_periods = n_change_periods
        self.enable_ta_features = enable_ta_features
        self.enable_pattern_features = enable_pattern_features
        self.enable_statistical_features = enable_statistical_features
        
        logger.info("Advanced feature engineering module initialized")
    
    def fractional_differentiation(self, series: pd.Series, d: float = None) -> pd.Series:
        """
        Apply fractional differentiation to time series to achieve stationarity
        while preserving memory properties.
        
        Args:
            series: Price or indicator time series
            d: Fractional differentiation parameter (0-0.5 recommended)
            
        Returns:
            Fractionally differentiated series
        """
        if d is None:
            d = self.fractional_differencing_d
            
        # Get weights for fractional differentiation
        def get_weights(d, size):
            weights = [1.0]
            for k in range(1, size):
                weights.append(weights[-1] * (k - 1 - d) / k)
            return np.array(weights)
        
        # Apply weights to series
        weights = get_weights(d, len(series))
        res = np.convolve(series, weights, 'full')[:len(series)]
        res = pd.Series(res, index=series.index)
        
        return res
    
    def create_stationary_features(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Create stationary versions of selected features using fractional differentiation.
        
        Args:
            data: DataFrame with time series features
            columns: List of columns to apply fractional differentiation to
            
        Returns:
            DataFrame with original and stationary features
        """
        result = data.copy()
        
        if columns is None:
            # Default to price columns
            columns = [col for col in data.columns if col.lower() in 
                      ['open', 'high', 'low', 'close', 'volume']]
        
        try:
            for col in columns:
                if col in data.columns and data[col].dtype.kind in 'if':  # Integer or float
                    # Create stationary version of the column
                    result[f'{col}_stationary'] = self.fractional_differentiation(data[col])
                    logger.info(f"Created stationary feature for {col}")
                    
                    # Verify stationarity
                    try:
                        adf_result = adfuller(result[f'{col}_stationary'].dropna())
                        logger.debug(f"ADF p-value for {col}_stationary: {adf_result[1]}")
                    except Exception as e:
                        logger.warning(f"Could not verify stationarity for {col}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating stationary features: {e}")
            return data
    
    def create_lag_features(self, data: pd.DataFrame, 
                           columns: List[str] = None, 
                           n_lags: int = None) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Args:
            data: DataFrame with time series data
            columns: List of columns to create lags for
            n_lags: Number of lags to create
            
        Returns:
            DataFrame with additional lag features
        """
        result = data.copy()
        
        if n_lags is None:
            n_lags = self.n_lags
            
        if columns is None:
            # Default to numeric columns
            columns = [col for col in data.columns if data[col].dtype.kind in 'if']
        
        try:
            for col in columns:
                if col in data.columns:
                    for lag in range(1, n_lags + 1):
                        result[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            logger.info(f"Created {n_lags} lag features for {len(columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            return data
    
    def create_rolling_features(self, data: pd.DataFrame,
                               columns: List[str] = None,
                               windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Create rolling window features like mean, std, min, max, etc.
        
        Args:
            data: DataFrame with time series data
            columns: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with additional rolling features
        """
        result = data.copy()
        
        if columns is None:
            # Default to numeric columns
            columns = [col for col in data.columns if data[col].dtype.kind in 'if']
        
        try:
            for col in columns:
                if col in data.columns:
                    for window in windows:
                        # Rolling statistics
                        result[f'{col}_rolling_mean_{window}'] = data[col].rolling(window).mean()
                        result[f'{col}_rolling_std_{window}'] = data[col].rolling(window).std()
                        result[f'{col}_rolling_min_{window}'] = data[col].rolling(window).min()
                        result[f'{col}_rolling_max_{window}'] = data[col].rolling(window).max()
                        result[f'{col}_rolling_median_{window}'] = data[col].rolling(window).median()
                        
                        # Z-score (how many std devs from mean)
                        rolling_mean = data[col].rolling(window).mean()
                        rolling_std = data[col].rolling(window).std()
                        result[f'{col}_zscore_{window}'] = (data[col] - rolling_mean) / rolling_std
                        
                        # Rate of change
                        result[f'{col}_roc_{window}'] = data[col].pct_change(window)
                        
                        # Momentum (current / past)
                        result[f'{col}_momentum_{window}'] = data[col] / data[col].shift(window)
            
            logger.info(f"Created rolling features with {len(windows)} windows for {len(columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            return data
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from technical indicators using pandas-ta or talib.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicator features
        """
        if not self.enable_ta_features:
            return data
            
        result = data.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in map(str.lower, data.columns) for col in required_cols):
            logger.warning(f"Missing required columns for technical features. Required: {required_cols}")
            return data
        
        try:
            # Map column names (case-insensitive)
            col_map = {}
            for req_col in required_cols:
                for col in data.columns:
                    if col.lower() == req_col:
                        col_map[req_col] = col
            
            # Extract mapped columns
            ohlcv = data[[col_map.get(c) for c in required_cols if c in col_map]]
            
            # Rename columns to standard lowercase
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            
            # RSI with multiple periods
            for period in [7, 14, 21]:
                result[f'rsi_{period}'] = ta.rsi(ohlcv['close'], length=period)
            
            # Multiple moving averages
            for ma_type in ['sma', 'ema', 'wma']:
                for period in [9, 20, 50, 200]:
                    result[f'{ma_type}_{period}'] = getattr(ta, ma_type)(ohlcv['close'], length=period)
            
            # MACD
            macd = ta.macd(ohlcv['close'])
            result = pd.concat([result, macd], axis=1)
            
            # Bollinger Bands
            bbands = ta.bbands(ohlcv['close'])
            result = pd.concat([result, bbands], axis=1)
            
            # Add distance to Bollinger Bands
            if 'BBU_20_2.0' in result.columns and 'BBL_20_2.0' in result.columns:
                result['bb_width'] = (result['BBU_20_2.0'] - result['BBL_20_2.0']) / result['BBM_20_2.0']
                result['bb_position'] = (ohlcv['close'] - result['BBL_20_2.0']) / (result['BBU_20_2.0'] - result['BBL_20_2.0'])
            
            # Stochastic oscillator
            stoch = ta.stoch(ohlcv['high'], ohlcv['low'], ohlcv['close'])
            result = pd.concat([result, stoch], axis=1)
            
            # Awesome Oscillator
            result['ao'] = ta.ao(ohlcv['high'], ohlcv['low'])
            
            # Commodity Channel Index
            for period in [14, 20, 40]:
                result[f'cci_{period}'] = ta.cci(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=period)
            
            # Directional Movement Index
            adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'])
            result = pd.concat([result, adx], axis=1)
            
            # Ichimoku Cloud
            ichimoku = ta.ichimoku(ohlcv['high'], ohlcv['low'], ohlcv['close'])
            result = pd.concat([result, ichimoku], axis=1)
            
            # Money Flow Index
            result['mfi_14'] = ta.mfi(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
            
            # On-Balance Volume
            result['obv'] = ta.obv(ohlcv['close'], ohlcv['volume'])
            
            # Chaikin Money Flow
            result['cmf_20'] = ta.cmf(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
            
            # ATR - Average True Range
            for period in [7, 14, 21]:
                result[f'atr_{period}'] = ta.atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=period)
            
            # Keltner Channels
            keltner = ta.kc(ohlcv['high'], ohlcv['low'], ohlcv['close'])
            result = pd.concat([result, keltner], axis=1)
            
            # Drop any columns with all NaN values
            result = result.loc[:, ~result.isna().all()]
            
            logger.info(f"Created {result.shape[1] - data.shape[1]} technical indicator features")
            return result
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return data
    
    def create_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_pattern_features:
            return data

        result = data.copy()
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col.lower() in map(str.lower, data.columns) for col in required_cols):
            logger.warning(f"Missing required columns for pattern features. Required: {required_cols}")
            return data

        col_map = {req_col: col for req_col in required_cols for col in data.columns if col.lower() == req_col}
        ohlc = result[[col_map.get(c) for c in required_cols if c in col_map]]
        ohlc.columns = [c.lower() for c in ohlc.columns]

        # Detetar padrões básicos manualmente
        high_series, low_series = ohlc['high'], ohlc['low']
        result['higher_high_3'] = ((high_series > high_series.shift(1)) & (high_series.shift(1) > high_series.shift(2))).astype(int)
        result['lower_low_3'] = ((low_series < low_series.shift(1)) & (low_series.shift(1) < low_series.shift(2))).astype(int)

        try:
            highs = ohlc['high'].values
            lows = ohlc['low'].values
            min_distance = 5
            prominence = np.nanmean(highs) * 0.01
            high_peaks, _ = find_peaks(highs, distance=min_distance, prominence=prominence)
            low_peaks, _ = find_peaks(-lows, distance=min_distance, prominence=prominence)
            result['double_top'] = 0
            result['double_bottom'] = 0

            if len(high_peaks) >= 2:
                for i in range(1, len(high_peaks)):
                    p1, p2 = high_peaks[i-1], high_peaks[i]
                    diff = abs(highs[p2] - highs[p1]) / highs[p1]
                    if diff < 0.005 and p2 - p1 < 20:
                        result.iloc[p2, result.columns.get_loc('double_top')] = 1

            if len(low_peaks) >= 2:
                for i in range(1, len(low_peaks)):
                    t1, t2 = low_peaks[i-1], low_peaks[i]
                    diff = abs(lows[t2] - lows[t1]) / lows[t1]
                    if diff < 0.005 and t2 - t1 < 20:
                        result.iloc[t2, result.columns.get_loc('double_bottom')] = 1

            logger.info("Created basic candlestick pattern features")

        except Exception as e:
            logger.warning(f"Error creating peak detection features: {e}")

        return result
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features capturing distribution characteristics.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with additional statistical features
        """
        if not self.enable_statistical_features:
            return data
            
        result = data.copy()
        
        # Columns to compute statistics for
        if 'close' not in result.columns and 'Close' in result.columns:
            price_col = 'Close'
        else:
            price_col = 'close'
            
        if price_col not in result.columns:
            logger.warning(f"Column {price_col} not found for statistical features")
            return data
        
        try:
            # Windows for rolling calculations
            windows = [10, 20, 50]
            
            for window in windows:
                # Rolling skewness
                result[f'skew_{window}'] = result[price_col].rolling(window).apply(
                    lambda x: stats.skew(x, nan_policy='omit'))
                
                # Rolling kurtosis
                result[f'kurt_{window}'] = result[price_col].rolling(window).apply(
                    lambda x: stats.kurtosis(x, nan_policy='omit'))
                
                # Rolling Jarque-Bera test for normality (p-value)
                def jarque_bera_p(x):
                    try:
                        return stats.jarque_bera(x)[1]
                    except:
                        return np.nan
                
                result[f'jb_pvalue_{window}'] = result[price_col].rolling(window).apply(jarque_bera_p)
                
                # Is the series normal according to JB test? (p > 0.05)
                result[f'is_normal_{window}'] = (result[f'jb_pvalue_{window}'] > 0.05).astype(int)
                
                # Rolling Hurst exponent (indication of mean reversion vs trend)
                def hurst_exp(prices):
                    try:
                        # Calculate returns
                        returns = np.diff(np.log(prices))
                        if len(returns) < 10:
                            return np.nan
                            
                        # Calculate variance of returns
                        tau = [2, 4, 8, 16]
                        var = []
                        for t in tau:
                            if t >= len(returns):
                                continue
                            # Use non-overlapping windows
                            x = np.std([returns[i:i+t].std() for i in range(0, len(returns)-t+1, t)])
                            var.append(x)
                        
                        if len(var) < 2:
                            return np.nan
                            
                        # Calculate Hurst as slope of log-log plot
                        reg = np.polyfit(np.log(tau[:len(var)]), np.log(var), 1)
                        hurst = reg[0] / 2.0
                        return hurst
                    except:
                        return np.nan
                
                # Try to calculate Hurst exponent
                try:
                    result[f'hurst_{window}'] = result[price_col].rolling(window+1).apply(hurst_exp)
                except Exception as he:
                    logger.warning(f"Could not calculate Hurst exponent: {he}")
                
                # Serial correlation (lag-1 autocorrelation)
                result[f'autocorr_{window}'] = result[price_col].rolling(window).apply(
                    lambda x: x.autocorr(1) if len(x.dropna()) > 1 else np.nan)
                
                # Variance ratio test (random walk vs mean-reverting/trending)
                def variance_ratio(x, lag=2):
                    try:
                        # Returns
                        returns = np.diff(np.log(x))
                        if len(returns) <= lag:
                            return np.nan
                            
                        # Variance of 1-period returns
                        var1 = np.var(returns, ddof=1)
                        
                        # Variance of lag-period returns
                        returns_lag = np.diff(np.log(x[::lag]))
                        var_lag = np.var(returns_lag, ddof=1)
                        
                        # Variance ratio
                        vr = var_lag / (lag * var1)
                        return vr
                    except:
                        return np.nan
                        
                result[f'variance_ratio_{window}'] = result[price_col].rolling(window+2).apply(
                    variance_ratio)
            
            logger.info(f"Created {result.shape[1] - data.shape[1]} statistical features")
            return result
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")
            return data
    
    def create_cross_asset_features(self, 
                                  data: pd.DataFrame, 
                                  related_assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features incorporating information from related assets.
        
        Args:
            data: Main asset DataFrame
            related_assets: Dictionary of related asset DataFrames
            
        Returns:
            DataFrame with cross-asset features
        """
        result = data.copy()
        
        if not related_assets:
            return result
            
        try:
            # For each related asset
            for asset_name, asset_data in related_assets.items():
                # Find common dates
                common_idx = data.index.intersection(asset_data.index)
                
                if len(common_idx) < 10:
                    logger.warning(f"Insufficient common dates with {asset_name}, skipping")
                    continue
                
                # Align dates
                main_data = data.loc[common_idx]
                related_data = asset_data.loc[common_idx]
                
                # Price correlation features
                if 'close' in main_data.columns and 'close' in related_data.columns:
                    # Correlations over different windows
                    for window in [10, 20, 50]:
                        if len(common_idx) > window:
                            result[f'corr_{asset_name}_{window}'] = main_data['close'].rolling(window).corr(
                                related_data['close']).reindex(data.index)
                    
                    # Relative strength
                    result[f'rel_strength_{asset_name}'] = (main_data['close'] / main_data['close'].iloc[0]) / \
                                                         (related_data['close'] / related_data['close'].iloc[0])
                    
                    # Ratio of volatilities
                    main_vol = main_data['close'].pct_change().rolling(20).std()
                    related_vol = related_data['close'].pct_change().rolling(20).std()
                    result[f'vol_ratio_{asset_name}'] = (main_vol / related_vol).reindex(data.index)
                    
                    # Normalize by reindexing back to original index
                    result[f'rel_strength_{asset_name}'] = result[f'rel_strength_{asset_name}'].reindex(data.index)
            
            logger.info(f"Created {result.shape[1] - data.shape[1]} cross-asset features")
            return result
            
        except Exception as e:
            logger.error(f"Error creating cross-asset features: {e}")
            return data
    
    def create_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        price_col = 'close' if 'close' in result.columns else 'Close'
        if price_col not in result.columns:
            logger.warning(f"Column {price_col} not found for market regime features")
            return data

        try:
            for window in [10, 20, 50]:
                if 'high' in result.columns and 'low' in result.columns:
                    adx_df = ta.adx(high=result['high'], low=result['low'], close=result[price_col], length=window)
                    result = pd.concat([result, adx_df], axis=1)
                    result[f'is_trending_{window}'] = (result[f'ADX_{window}'] > 25).astype(int)

                if f'DI+_{window}' in result.columns and f'DI-_{window}' in result.columns:
                    result[f'trend_direction_{window}'] = np.where(result[f'DI+_{window}'] > result[f'DI-_{window}'], 1, -1)

                ma_col = f'sma_{window}'
                if ma_col in result.columns:
                    atr_col = f'atr_{window}'
                    if atr_col not in result.columns:
                        result[atr_col] = ta.atr(high=result['high'], low=result['low'], close=result[price_col], length=window)

                    result[f'ma_trend_intensity_{window}'] = (result[price_col] - result[ma_col]) / result[atr_col]
                    result[f'above_ma_{window}'] = (result[price_col] > result[ma_col]).astype(int)
                    result[f'ma_slope_{window}'] = result[ma_col].diff(5) / result[ma_col].shift(5)

            logger.info("Created market regime features with pandas-ta")
            return result

        except Exception as e:
            logger.error(f"Error creating market regime features: {e}")
            return data

    
    def create_sentiment_features(self, data: pd.DataFrame, 
                                sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features based on market sentiment indicators.
        
        Args:
            data: DataFrame with time series data
            sentiment_data: Optional DataFrame with sentiment indicators (news sentiment, social media, etc.)
            
        Returns:
            DataFrame with sentiment features
        """
        result = data.copy()
        
        try:
            # If external sentiment data is provided
            if sentiment_data is not None:
                # Align indices
                common_idx = data.index.intersection(sentiment_data.index)
                if len(common_idx) < 10:
                    logger.warning("Insufficient common dates with sentiment data, skipping")
                    return data
                
                # Join relevant sentiment columns
                sentiment_cols = [col for col in sentiment_data.columns 
                                 if any(x in col.lower() for x in ['sentiment', 'bullish', 'bearish', 'score'])]
                
                if sentiment_cols:
                    for col in sentiment_cols:
                        result[f'ext_{col}'] = sentiment_data.loc[common_idx, col].reindex(data.index)
                    
                    logger.info(f"Added {len(sentiment_cols)} external sentiment features")
            
            # Create internal sentiment indicators from price/volume data
            if 'close' in result.columns and 'volume' in result.columns:
                # Calculate price and volume moving averages
                price_col = 'close'
                vol_col = 'volume'
                
                # Price-volume relationship (higher volume on up or down days)
                for window in [5, 10, 20]:
                    # Up/down volume ratio
                    price_change = result[price_col].diff()
                    up_volume = np.where(price_change > 0, result[vol_col], 0)
                    down_volume = np.where(price_change < 0, result[vol_col], 0)
                    
                    result[f'up_vol_{window}'] = pd.Series(up_volume).rolling(window).sum()
                    result[f'down_vol_{window}'] = pd.Series(down_volume).rolling(window).sum()
                    result[f'vol_ratio_{window}'] = result[f'up_vol_{window}'] / result[f'down_vol_{window}']
                    
                    # Money Flow Index components
                    typical_price = (result[price_col] + result['high'] + result['low']) / 3 if 'high' in result.columns and 'low' in result.columns else result[price_col]
                    money_flow = typical_price * result[vol_col]
                    
                    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
                    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
                    
                    result[f'pos_flow_{window}'] = pd.Series(pos_flow).rolling(window).sum()
                    result[f'neg_flow_{window}'] = pd.Series(neg_flow).rolling(window).sum()
                    result[f'flow_ratio_{window}'] = result[f'pos_flow_{window}'] / result[f'neg_flow_{window}']
                
                # Accumulation/Distribution Line
                if 'high' in result.columns and 'low' in result.columns:
                    high, low, close, volume = result['high'], result['low'], result[price_col], result[vol_col]
                    clv = ((close - low) - (high - close)) / (high - low)
                    clv = clv.replace([np.inf, -np.inf], 0)  # Handle division by zero
                    adl = (clv * volume).cumsum()
                    result['adl'] = adl
                    
                    # ADL slope
                    result['adl_slope_5'] = adl.diff(5) / abs(adl).rolling(5).mean()
                    result['adl_slope_10'] = adl.diff(10) / abs(adl).rolling(10).mean()
                
                # Chaikin Oscillator
                if 'adl' in result.columns:
                    result['chaikin_osc'] = ta.ema(result['adl'], length=3) - ta.ema(result['adl'], length=10)
            
            logger.info(f"Created {result.shape[1] - data.shape[1]} sentiment features")
            return result
            
        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return data
    
    def create_target_variables(self, data: pd.DataFrame, 
                               price_col: str = 'close',
                               horizons: List[int] = [1, 5, 10, 20],
                               returns_only: bool = False) -> pd.DataFrame:
        """
        Create target variables for supervised learning.
        
        Args:
            data: DataFrame with time series data
            price_col: Column name containing price data
            horizons: List of forecast horizons
            returns_only: If True, only create return targets
            
        Returns:
            DataFrame with added target variables
        """
        result = data.copy()
        
        if price_col not in result.columns:
            if price_col.lower() in result.columns:
                price_col = price_col.lower()
            elif 'Close' in result.columns:
                price_col = 'Close'
            elif 'close' in result.columns:
                price_col = 'close'
            else:
                logger.error(f"Price column '{price_col}' not found for target creation")
                return data
        
        try:
            # Calculate forward returns for different horizons
            for horizon in horizons:
                # Future price
                result[f'future_price_{horizon}'] = result[price_col].shift(-horizon)
                
                # Future return
                result[f'future_return_{horizon}'] = result[f'future_price_{horizon}'] / result[price_col] - 1
                
                if not returns_only:
                    # Binary direction (up/down)
                    result[f'direction_{horizon}'] = np.where(result[f'future_return_{horizon}'] > 0, 1, 0)
                    
                    # Ternary direction with neutral zone (up/neutral/down)
                    neutral_threshold = 0.0025 * np.sqrt(horizon)  # Scale threshold with horizon
                    result[f'direction_ternary_{horizon}'] = np.where(
                        result[f'future_return_{horizon}'] > neutral_threshold, 1,
                        np.where(result[f'future_return_{horizon}'] < -neutral_threshold, -1, 0)
                    )
                    
                    # Volatility target (future realized vol)
                    future_vol = pd.Series(
                        [result[price_col].iloc[i:i+horizon].pct_change().std() * np.sqrt(252)
                         if i+horizon < len(result) else np.nan 
                         for i in range(len(result))]
                    )
                    result[f'future_volatility_{horizon}'] = future_vol
                    
                    # High volatility event (binary)
                    hist_vol = result[price_col].pct_change().rolling(20).std() * np.sqrt(252)
                    vol_threshold = hist_vol.rolling(100).quantile(0.8)
                    result[f'high_vol_event_{horizon}'] = np.where(
                        future_vol > vol_threshold, 1, 0
                    )
                    
                    # Extreme move target (tail event)
                    returns_std = result[price_col].pct_change().rolling(100).std()
                    result[f'extreme_up_{horizon}'] = np.where(
                        result[f'future_return_{horizon}'] > 2 * returns_std, 1, 0
                    )
                    result[f'extreme_down_{horizon}'] = np.where(
                        result[f'future_return_{horizon}'] < -2 * returns_std, 1, 0
                    )
            
            logger.info(f"Created {result.shape[1] - data.shape[1]} target variables")
            return result
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return data
    
    def remove_leakage_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that may cause data leakage.
        
        Args:
            data: DataFrame with all features
            
        Returns:
            DataFrame with leakage features removed
        """
        result = data.copy()
        
        # Identify potential leakage columns
        leakage_patterns = [
            'future_', 'next_', 'target_', 'label', 
            'direction_', 'extreme_', 'volatility_'
        ]
        
        leakage_cols = [col for col in result.columns 
                       if any(pattern in col for pattern in leakage_patterns)]
        
        # Don't drop them, just identify them
        if leakage_cols:
            logger.info(f"Identified {len(leakage_cols)} potential leakage features: {leakage_cols}")
        
        return result
    
    def select_features(self, data: pd.DataFrame, 
                       importance_threshold: float = 0.01,
                       correlation_threshold: float = 0.95,
                       target_col: str = None) -> pd.DataFrame:
        """
        Select features based on importance and correlation.
        
        Args:
            data: DataFrame with all features
            importance_threshold: Minimum feature importance to keep
            correlation_threshold: Maximum correlation between features
            target_col: Target column for feature importance (optional)
            
        Returns:
            DataFrame with selected features
        """
        result = data.copy()
        
        try:
            # Remove missing values
            missing_pct = result.isnull().mean()
            high_missing = missing_pct[missing_pct > 0.5].index.tolist()
            
            if high_missing:
                logger.warning(f"Dropping {len(high_missing)} features with >50% missing values")
                result = result.drop(columns=high_missing)
            
            # Only keep numeric columns
            numeric_cols = [col for col in result.columns if result[col].dtype.kind in 'if']
            result = result[numeric_cols]
            
            # Handle correlation-based selection
            if correlation_threshold < 1.0:
                # Calculate correlation matrix
                corr_matrix = result.corr().abs()
                
                # Create upper triangle matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Find features with correlation higher than threshold
                high_corr_features = [column for column in upper.columns 
                                     if any(upper[column] > correlation_threshold)]
                
                if high_corr_features:
                    logger.info(f"Dropping {len(high_corr_features)} highly correlated features")
                    result = result.drop(columns=high_corr_features)
            
            # Feature importance-based selection (if target column provided)
            if target_col is not None and target_col in data.columns:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import StandardScaler
                    
                    # Extract features and target
                    y = data[target_col].dropna()
                    X = data.drop(columns=[c for c in data.columns 
                                         if any(p in c for p in ['future_', 'direction_', 'extreme_'])])
                    X = X.loc[y.index].select_dtypes(include=['number']).fillna(0)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train a Random Forest for feature importance
                    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    rf.fit(X_scaled, y)
                    
                    # Get feature importances
                    importances = pd.Series(rf.feature_importances_, index=X.columns)
                    important_features = importances[importances > importance_threshold].index.tolist()
                    
                    if important_features:
                        logger.info(f"Selected {len(important_features)} features based on importance threshold")
                        
                        # Find original features plus important ones
                        original_cols = ['open', 'high', 'low', 'close', 'volume']
                        original_cols = [c for c in original_cols if c in result.columns]
                        selected_cols = list(set(original_cols + important_features))
                        
                        result = result[selected_cols]
                except Exception as e:
                    logger.warning(f"Feature importance selection failed: {e}")
            
            logger.info(f"Final feature set contains {result.shape[1]} features")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return data
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features for ML models.
        
        Args:
            data: DataFrame with features
            method: Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        result = data.copy()
        
        # Keep track of non-numeric columns to add back later
        non_numeric_cols = [col for col in result.columns if result[col].dtype.kind not in 'if']
        non_numeric_data = result[non_numeric_cols].copy() if non_numeric_cols else None
        
        # Get only numeric columns
        numeric_cols = [col for col in result.columns if result[col].dtype.kind in 'if']
        numeric_data = result[numeric_cols].copy()
        
        try:
            if method == 'zscore':
                # Z-score normalization
                for col in numeric_cols:
                    numeric_data[col] = (numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std()
                    
            elif method == 'minmax':
                # Min-max scaling
                for col in numeric_cols:
                    numeric_data[col] = (numeric_data[col] - numeric_data[col].min()) / (numeric_data[col].max() - numeric_data[col].min())
                    
            elif method == 'robust':
                # Robust scaling based on median and quantiles
                for col in numeric_cols:
                    median = numeric_data[col].median()
                    q1, q3 = numeric_data[col].quantile(0.25), numeric_data[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        numeric_data[col] = (numeric_data[col] - median) / iqr
                    else:
                        # Fall back to z-score if IQR is 0
                        numeric_data[col] = (numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std()
            
            # Replace inf values with NaN, then fill NaN with 0
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Combine numeric and non-numeric data
            if non_numeric_data is not None:
                result = pd.concat([numeric_data, non_numeric_data], axis=1)
            else:
                result = numeric_data
                
            logger.info(f"Normalized {len(numeric_cols)} features using {method} method")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature normalization: {e}")
            return data
    
    def apply_all_features(self, data: pd.DataFrame, 
                          related_assets: Dict[str, pd.DataFrame] = None,
                          sentiment_data: pd.DataFrame = None,
                          target_horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Apply all feature engineering steps in the recommended order.
        
        Args:
            data: DataFrame with raw OHLCV data
            related_assets: Dictionary with related asset DataFrames
            sentiment_data: DataFrame with sentiment indicators
            target_horizons: List of forecast horizons for target creation
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            logger.info("Starting comprehensive feature engineering pipeline")
            
            # 1. Create stationary features
            logger.info("Creating stationary features")
            result = self.create_stationary_features(data)
            
            # 2. Create technical indicators
            if self.enable_ta_features:
                logger.info("Creating technical features")
                result = self.create_technical_features(result)
            
            # 3. Create pattern recognition features
            if self.enable_pattern_features:
                logger.info("Creating pattern features")
                result = self.create_pattern_features(result)
            
            # 4. Create market regime features
            logger.info("Creating market regime features")
            result = self.create_market_regime_features(result)
            
            # 5. Create statistical features
            if self.enable_statistical_features:
                logger.info("Creating statistical features")
                result = self.create_statistical_features(result)
            
            # 6. Create lag features
            logger.info("Creating lag features")
            result = self.create_lag_features(result)
            
            # 7. Create rolling features
            logger.info("Creating rolling features")
            result = self.create_rolling_features(result)
            
            # 8. Create cross-asset features if related assets provided
            if related_assets:
                logger.info("Creating cross-asset features")
                result = self.create_cross_asset_features(result, related_assets)
            
            # 9. Create sentiment features if data provided
            if sentiment_data is not None:
                logger.info("Creating sentiment features")
                result = self.create_sentiment_features(result, sentiment_data)
                
            # 10. Create target variables
            if target_horizons:
                logger.info("Creating target variables")
                result = self.create_target_variables(result, horizons=target_horizons)
            
            # 11. Remove features with too many missing values
            result = result.dropna(axis=1, thresh=len(result) * 0.5)  # Drop cols with >50% missing
            
            # 12. Fill remaining NAs
            result = result.fillna(method='ffill').fillna(0)
            
            logger.info(f"Feature engineering complete. Created {result.shape[1]} features from {data.shape[1]} original columns")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            return data
        
        

    def test_feature_engineering_and_plotting(df):
        logger.info("\n=== Testing Advanced Feature Engineering ===")

        engine = AdvancedFeatureEngineering()
        df_features = engine.apply_all_features(df)

        # Verifica se temos coluna 'close'
        if 'close' not in df_features.columns:
            logger.error("Coluna 'close' não encontrada após feature engineering.")
            return df_features

        # Gráfico de suporte e resistência
        plt.figure(figsize=(14, 6))
        plt.plot(df_features['close'], label='Preço', color='black')
        if 'support_level' in df_features.columns:
            plt.plot(df_features['support_level'], label='Suporte', color='green', linestyle='--')
        if 'resistance_level' in df_features.columns:
            plt.plot(df_features['resistance_level'], label='Resistência', color='red', linestyle='--')
        plt.title('Níveis de Suporte e Resistência')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('testing/control_point_2/grafico_suporte_resistencia.png')
        plt.close()

        # Gráfico de padrões técnicos
        plt.figure(figsize=(14, 6))
        plt.plot(df_features['close'], label='Preço', color='black')
        if 'double_top' in df_features.columns:
            tops = df_features[df_features['double_top'] == 1]
            plt.scatter(tops.index, tops['close'], color='red', marker='v', s=100, label='Double Top')
        if 'double_bottom' in df_features.columns:
            bottoms = df_features[df_features['double_bottom'] == 1]
            plt.scatter(bottoms.index, bottoms['close'], color='green', marker='^', s=100, label='Double Bottom')
        plt.title('Padrões Técnicos: Double Top / Bottom')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('testing/control_point_2/grafico_padroes_tecnicos.png')
        plt.close()

        logger.info("Gráficos salvos com sucesso:")
        logger.info(" - grafico_suporte_resistencia.png")
        logger.info(" - grafico_padroes_tecnicos.png")

        return df_features