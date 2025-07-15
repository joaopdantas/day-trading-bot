"""
Data Preprocessor Module for Day Trading Bot.

This module handles data cleaning, normalization, and preparation for analysis.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing market data for analysis and modeling."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = MinMaxScaler(feature_range=(0, 1))
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Make a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_cleaned.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # If close price is missing but we have others, estimate it
            if 'close' in missing_columns and 'open' in df_cleaned.columns:
                logger.info("Estimating close price using open price")
                df_cleaned['close'] = df_cleaned['open']
            
            # If volume is missing, add a column of zeros
            if 'volume' in missing_columns:
                logger.info("Adding volume column with zeros")
                df_cleaned['volume'] = 0
        
        # Handle missing values
        # For OHLC columns, use forward fill to use the previous value
        numeric_cols = df_cleaned.select_dtypes(include=['float', 'int']).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(method='ffill')
        
        # For any remaining NaNs, use backward fill
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(method='bfill')
        
        # For any still remaining NaNs (if the series starts with NaNs), replace with zeros
        df_cleaned = df_cleaned.fillna(0)
        
        # Handle outliers in price data using Interquartile Range method
        if all(col in df_cleaned.columns for col in ['open', 'high', 'low', 'close']):
            for col in ['open', 'high', 'low', 'close']:
                df_cleaned = self._handle_outliers(df_cleaned, col)
        
        # Handle outliers in volume data
        if 'volume' in df_cleaned.columns:
            df_cleaned = self._handle_outliers(df_cleaned, 'volume')
        
        return df_cleaned
    
    def _handle_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Handle outliers in a specific column using IQR method.
        
        Args:
            df: DataFrame with market data
            column: Column name to process
        
        Returns:
            DataFrame with outliers handled
        """
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        if outliers.sum() > 0:
            logger.info(f"Found {outliers.sum()} outliers in {column}")
            
            # Replace outliers with boundary values
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
        
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize price and volume data for machine learning algorithms.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
        
        df_norm = df.copy()
        
        # Normalize price data (OHLC)
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_norm.columns]
        if price_cols:
            # Fit the scaler on all price data to maintain relationships
            price_data = df_norm[price_cols].values
            self.price_scaler.fit(price_data)
            
            # Transform the data
            normalized_prices = self.price_scaler.transform(price_data)
            
            # Update the DataFrame
            for i, col in enumerate(price_cols):
                df_norm[f'{col}_norm'] = normalized_prices[:, i]
        
        # Normalize volume separately
        if 'volume' in df_norm.columns:
            # Reshape volume data for scaler
            volume_data = df_norm['volume'].values.reshape(-1, 1)
            self.volume_scaler.fit(volume_data)
            
            # Transform and update
            df_norm['volume_norm'] = self.volume_scaler.transform(volume_data)
        
        return df_norm
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features like day of week, hour of day, etc.
        
        Args:
            df: DataFrame with market data with datetime index
            
        Returns:
            DataFrame with additional time features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for adding time features")
            return df
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex. Attempting to convert.")
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.error("Failed to convert index to datetime. Skipping time features.")
                return df
        
        df_time = df.copy()
        
        # Add day of week (0 = Monday, 6 = Sunday)
        df_time['day_of_week'] = df_time.index.dayofweek
        
        # Add hour of day (0-23)
        df_time['hour_of_day'] = df_time.index.hour
        
        # Month (1-12)
        df_time['month'] = df_time.index.month
        
        # Quarter (1-4)
        df_time['quarter'] = df_time.index.quarter
        
        # Is market opening hour (9-10 AM)
        df_time['is_market_open'] = ((df_time.index.hour >= 9) & (df_time.index.hour < 10)).astype(int)
        
        # Is market closing hour (3-4 PM)
        df_time['is_market_close'] = ((df_time.index.hour >= 15) & (df_time.index.hour < 16)).astype(int)
        
        # Is Monday
        df_time['is_monday'] = (df_time.index.dayofweek == 0).astype(int)
        
        # Is Friday
        df_time['is_friday'] = (df_time.index.dayofweek == 4).astype(int)
        
        return df_time
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with additional return metrics
        """
        if df.empty or 'close' not in df.columns:
            logger.warning("Cannot calculate returns: Empty DataFrame or missing 'close' column")
            return df
        
        df_returns = df.copy()
        
        # Calculate daily returns
        df_returns['daily_return'] = df_returns['close'].pct_change()
        
        # Calculate log returns
        df_returns['log_return'] = np.log(df_returns['close'] / df_returns['close'].shift(1))
        
        # Calculate cumulative returns
        df_returns['cum_return'] = (1 + df_returns['daily_return']).cumprod() - 1
        
        # Fill NaN values in the first row
        df_returns[['daily_return', 'log_return', 'cum_return']] = \
            df_returns[['daily_return', 'log_return', 'cum_return']].fillna(0)
        
        return df_returns
    
    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
        """
        Prepare features for model training.
        
        Args:
            df: DataFrame with market data
            feature_columns: List of column names to include as features
            
        Returns:
            DataFrame ready for model training
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature preparation")
            return df
        
        # Apply all preprocessing steps
        df_processed = self.clean_data(df)
        df_processed = self.normalize_data(df_processed)
        df_processed = self.add_time_features(df_processed)
        df_processed = self.calculate_returns(df_processed)
        
        # Select only requested feature columns
        if feature_columns:
            available_columns = [col for col in feature_columns if col in df_processed.columns]
            missing_columns = [col for col in feature_columns if col not in df_processed.columns]
            
            if missing_columns:
                logger.warning(f"Requested feature columns not found: {missing_columns}")
            
            df_processed = df_processed[available_columns]
        
        return df_processed
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 10,
        target_column: str = 'close',
        feature_columns: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with processed market data
            sequence_length: Number of time steps in each sequence
            target_column: Column to predict
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (X, y) where X is the feature sequences and y is the target values
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for sequence creation")
            return np.array([]), np.array([])
        
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame")
            return np.array([]), np.array([])
        
        # Default to all columns if not specified
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Ensure all feature columns exist
        available_feature_cols = [col for col in feature_columns if col in df.columns]
        
        if len(available_feature_cols) == 0:
            logger.error("No valid feature columns found")
            return np.array([]), np.array([])
        
        # Extract feature and target data
        data = df[available_feature_cols].values
        target = df[target_column].values
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y)