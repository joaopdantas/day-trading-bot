"""
End-to-End Test of Data Pipeline for Control Point 2.

This script tests the entire data pipeline from fetching to storage:
1. Fetch historical data from API
2. Preprocess the data
3. Calculate technical indicators
4. Recognize patterns
5. Store in MongoDB
6. Retrieve from MongoDB
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.fetcher import get_data_api
from src.data.preprocessor import DataPreprocessor
from src.data.storage import get_storage
from src.indicators.technical import TechnicalIndicators, PatternRecognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_fetching():
    """Test fetching historical data from APIs."""
    logger.info("=== Testing Data Fetching ===")
    
    # Test with Alpha Vantage
    logger.info("Testing Alpha Vantage API...")
    alpha_vantage = get_data_api("alpha_vantage")
    
    # Define test parameters
    symbol = "MSFT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Fetch daily data
    df_daily = alpha_vantage.fetch_historical_data(
        symbol, 
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if not df_daily.empty:
        logger.info(f"Successfully fetched {len(df_daily)} daily records for {symbol}")
        logger.info(f"Columns: {df_daily.columns.tolist()}")
        logger.info(f"Date range: {df_daily.index.min()} to {df_daily.index.max()}")
    else:
        logger.error(f"Failed to fetch daily data for {symbol}")
    
    # Test with Yahoo Finance
    logger.info("\nTesting Yahoo Finance API...")
    yahoo_finance = get_data_api("yahoo_finance")
    
    # Fetch daily data
    df_daily_yf = yahoo_finance.fetch_historical_data(
        symbol, 
        interval="1d",
        start_date=start_date,
        end_date=end_date
    )
    
    if not df_daily_yf.empty:
        logger.info(f"Successfully fetched {len(df_daily_yf)} daily records for {symbol}")
        logger.info(f"Columns: {df_daily_yf.columns.tolist()}")
        logger.info(f"Date range: {df_daily_yf.index.min()} to {df_daily_yf.index.max()}")
    else:
        logger.error(f"Failed to fetch daily data for {symbol}")
    
    return df_daily if not df_daily.empty else df_daily_yf


def test_data_preprocessing(df):
    """Test preprocessing market data."""
    logger.info("\n=== Testing Data Preprocessing ===")
    
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_cleaned = preprocessor.clean_data(df)
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    
    # Normalize data
    df_normalized = preprocessor.normalize_data(df_cleaned)
    normalized_cols = [col for col in df_normalized.columns if col.endswith('_norm')]
    logger.info(f"Normalized columns: {normalized_cols}")
    
    # Add time features
    df_with_time = preprocessor.add_time_features(df_normalized)
    time_cols = ['day_of_week', 'hour_of_day', 'month', 'quarter']
    logger.info(f"Time features added: {[col for col in time_cols if col in df_with_time.columns]}")
    
    # Calculate returns
    df_with_returns = preprocessor.calculate_returns(df_with_time)
    return_cols = ['daily_return', 'log_return', 'cum_return']
    logger.info(f"Return metrics added: {[col for col in return_cols if col in df_with_returns.columns]}")
    
    # Full preprocessing
    df_processed = preprocessor.prepare_features(df)
    logger.info(f"Fully processed data shape: {df_processed.shape}")
    logger.info(f"Processed columns: {df_processed.columns.tolist()}")
    
    return df_processed


def test_technical_indicators(df):
    """Test calculating technical indicators."""
    logger.info("\n=== Testing Technical Indicators ===")
    
    # Add all indicators
    df_indicators = TechnicalIndicators.add_all_indicators(df)
    
    # Check which indicators were added
    original_cols = set(df.columns)
    indicator_cols = set(df_indicators.columns) - original_cols
    
    logger.info(f"Added {len(indicator_cols)} technical indicator columns")
    
    # Test specific indicators
    rsi_cols = [col for col in df_indicators.columns if 'rsi' in col.lower()]
    logger.info(f"RSI indicators: {rsi_cols}")
    
    macd_cols = [col for col in df_indicators.columns if 'macd' in col.lower()]
    logger.info(f"MACD indicators: {macd_cols}")
    
    bb_cols = [col for col in df_indicators.columns if 'bb_' in col.lower()]
    logger.info(f"Bollinger Bands indicators: {bb_cols}")
    
    # Plot an indicator for visual verification
    plt.figure(figsize=(12, 6))
    
    # Price with Bollinger Bands
    if 'bb_upper' in df_indicators.columns:
        plt.subplot(2, 1, 1)
        plt.plot(df_indicators.index, df_indicators['close'], label='Close')
        plt.plot(df_indicators.index, df_indicators['bb_upper'], 'r--', label='Upper BB')
        plt.plot(df_indicators.index, df_indicators['bb_middle'], 'g--', label='Middle BB')
        plt.plot(df_indicators.index, df_indicators['bb_lower'], 'r--', label='Lower BB')
        plt.title('Price with Bollinger Bands')
        plt.legend()
    
    # RSI
    if 'rsi' in df_indicators.columns:
        plt.subplot(2, 1, 2)
        plt.plot(df_indicators.index, df_indicators['rsi'])
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI')
    
    plt.tight_layout()
    plt.savefig('testing/control_point_2/test_indicators.png')
    logger.info("Saved indicator visualization to testing/control_point_2/test_indicators.png")
    
    return df_indicators


def test_pattern_recognition(df_indicators):
    """Test recognizing patterns."""
    logger.info("\n=== Testing Pattern Recognition ===")
    
    # Recognize candlestick patterns
    df_patterns = PatternRecognition.recognize_candlestick_patterns(df_indicators)
    
    # Check which patterns were detected
    pattern_cols = [col for col in df_patterns.columns if 'pattern_' in col]
    logger.info(f"Detected {len(pattern_cols)} pattern types")
    
    for col in pattern_cols:
        pattern_count = df_patterns[col].sum()
        if pattern_count > 0:
            logger.info(f"Found {pattern_count} instances of {col}")
    
    # Detect support/resistance levels
    df_levels = PatternRecognition.detect_support_resistance(df_patterns)
    level_cols = ['support_level', 'resistance_level', 'at_support', 'at_resistance']
    logger.info(f"Support/resistance columns: {[col for col in level_cols if col in df_levels.columns]}")
    
    # Detect trend
    df_with_trend = PatternRecognition.detect_trend(df_levels)
    trend_cols = ['uptrend', 'downtrend', 'trend_strength', 'golden_cross', 'death_cross']
    logger.info(f"Trend columns: {[col for col in trend_cols if col in df_with_trend.columns]}")
    
    # Plot patterns for visual verification
    plt.figure(figsize=(12, 8))
    
    # Price chart with patterns
    plt.subplot(3, 1, 1)
    plt.plot(df_with_trend.index, df_with_trend['close'], label='Close Price')
    
    # Highlight patterns
    for pattern in ['pattern_bullish_engulfing', 'pattern_bearish_engulfing']:
        if pattern in df_with_trend.columns:
            pattern_dates = df_with_trend.index[df_with_trend[pattern] == 1]
            pattern_prices = df_with_trend.loc[pattern_dates, 'close']
            if len(pattern_dates) > 0:
                plt.scatter(pattern_dates, pattern_prices, 
                           marker='^' if 'bullish' in pattern else 'v',
                           color='g' if 'bullish' in pattern else 'r',
                           s=100, label=pattern.replace('pattern_', ''))
    
    plt.title('Price with Candlestick Patterns')
    plt.legend()
    
    # Trend
    plt.subplot(3, 1, 2)
    plt.plot(df_with_trend.index, df_with_trend['uptrend'], 'g-', label='Uptrend')
    plt.plot(df_with_trend.index, df_with_trend['downtrend'], 'r-', label='Downtrend')
    plt.title('Trend Indicators')
    plt.legend()
    
    # Support/Resistance
    plt.subplot(3, 1, 3)
    plt.plot(df_with_trend.index, df_with_trend['close'], label='Close')
    
    # Plot support and resistance if detected
    if 'support_level' in df_with_trend.columns and df_with_trend['support_level'].notna().any():
        support = df_with_trend['support_level'].dropna().iloc[0]
        plt.axhline(y=support, color='g', linestyle='-', label=f'Support: {support:.2f}')
    
    if 'resistance_level' in df_with_trend.columns and df_with_trend['resistance_level'].notna().any():
        resistance = df_with_trend['resistance_level'].dropna().iloc[0]
        plt.axhline(y=resistance, color='r', linestyle='-', label=f'Resistance: {resistance:.2f}')
    
    plt.title('Support and Resistance Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('testing/control_point_2/test_patterns.png')
    logger.info("Saved pattern visualization to testing/control_point_2/test_patterns.png")
    
    return df_with_trend


def test_data_storage(df_with_indicators_and_patterns, symbol="MSFT"):
    """Test storing and retrieving data from MongoDB."""
    logger.info("\n=== Testing Data Storage ===")
    
    # Initialize storage
    storage = get_storage()
    
    # Check if MongoDB is available
    if storage.db is None:
        logger.warning("MongoDB not available. Skipping storage tests.")
        return False
    
    # Store historical data
    store_success = storage.store_historical_data(
        symbol=symbol,
        interval="1d",
        data=df_with_indicators_and_patterns,
        source="test_pipeline"
    )
    
    if store_success:
        logger.info(f"Successfully stored {len(df_with_indicators_and_patterns)} records for {symbol}")
    else:
        logger.error(f"Failed to store data for {symbol}")
        return False
    
    # Extract pattern columns for separate storage
    pattern_cols = [col for col in df_with_indicators_and_patterns.columns if 'pattern_' in col]
    if pattern_cols:
        patterns_df = df_with_indicators_and_patterns[pattern_cols].copy()
        
        # Store patterns
        patterns_success = storage.store_patterns(symbol, patterns_df)
        if patterns_success:
            logger.info(f"Successfully stored pattern data for {symbol}")
        else:
            logger.error(f"Failed to store pattern data for {symbol}")
    
    # Retrieve historical data
    retrieved_df = storage.retrieve_historical_data(
        symbol=symbol,
        interval="1d",
        limit=100
    )
    
    if not retrieved_df.empty:
        logger.info(f"Successfully retrieved {len(retrieved_df)} records for {symbol}")
        logger.info(f"Retrieved columns: {retrieved_df.columns.tolist()}")
    else:
        logger.error(f"Failed to retrieve data for {symbol}")
        return False
    
    # Get database statistics
    stats = storage.get_data_statistics()
    logger.info(f"Database statistics: {stats}")
    
    # Close connection
    storage.close()
    
    return True


def run_pipeline_test():
    """Run the complete data pipeline test."""
    logger.info("Starting Data Pipeline Test for Control Point 2")
    
    # Step 1: Fetch Data
    df = test_data_fetching()
    if df.empty:
        logger.error("Data fetching failed. Aborting test.")
        return
    
    # Step 2: Preprocess Data
    df_processed = test_data_preprocessing(df)
    
    # Step 3: Calculate Technical Indicators
    df_indicators = test_technical_indicators(df_processed)
    
    # Step 4: Recognize Patterns
    df_with_patterns = test_pattern_recognition(df_indicators)
    
    # Step 5: Test Storage
    storage_success = test_data_storage(df_with_patterns)
    
    logger.info("\n=== Data Pipeline Test Summary ===")
    logger.info(f"Data Fetching: {'Success' if not df.empty else 'Failed'}")
    logger.info(f"Data Preprocessing: {'Success' if not df_processed.empty else 'Failed'}")
    logger.info(f"Technical Indicators: {'Success' if not df_indicators.empty else 'Failed'}")
    logger.info(f"Pattern Recognition: {'Success' if not df_with_patterns.empty else 'Failed'}")
    logger.info(f"Data Storage: {'Success' if storage_success else 'Failed or Skipped'}")
    
    # Overall result
    if not df.empty and not df_processed.empty and not df_indicators.empty and not df_with_patterns.empty:
        logger.info("\nControl Point 2 Data Pipeline Test: SUCCESS")
        return True
    else:
        logger.error("\nControl Point 2 Data Pipeline Test: FAILED")
        return False


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('testing/control_point_2', exist_ok=True)
    
    # Run the test
    run_pipeline_test()