"""
Data Fetcher Module for Day Trading Bot.

This module handles retrieving market data from various APIs.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
from dotenv import load_dotenv  # For loading environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MarketDataFetcher:
    """Base class for fetching market data from various sources."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            api_key: API key for authentication (optional if provided via env vars)
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Some functionality may be limited.")
    
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a given symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            interval: Time interval between data points (e.g., '1d', '1h', '15m')
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)
            
        Returns:
            DataFrame with historical price data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def fetch_latest_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch latest price for a given symbol.
        
        Args:
            symbol: The stock/cryptocurrency symbol
            
        Returns:
            Dictionary with latest price information
        """
        raise NotImplementedError("Subclasses must implement this method")


class AlphaVantageAPI(MarketDataFetcher):
    """Implementation for Alpha Vantage API."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data from Alpha Vantage.
        
        Args:
            symbol: The stock symbol
            interval: Time interval between data points ('1d', '1h', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical price data
        """
        # Map intervals to Alpha Vantage's format
        interval_mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "1d": "daily",
        }
        
        av_interval = interval_mapping.get(interval, "daily")
        
        # Choose the appropriate function based on interval
        if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": av_interval,
                "outputsize": "full",
                "apikey": self.api_key,
            }
        else:
            function = "TIME_SERIES_DAILY"
            params = {
                "function": function,
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            }
        
        logger.info(f"Fetching {interval} data for {symbol} from Alpha Vantage")
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            
            # Extract time series data
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({av_interval})"
            else:
                time_series_key = "Time Series (Daily)"
            
            if time_series_key not in data:
                error_message = data.get("Error Message", "Unknown error")
                logger.error(f"API error: {error_message}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convert strings to numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add date as a column
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Filter by date range if provided
            if start_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def fetch_latest_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch latest price for a stock from Alpha Vantage.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Dictionary with latest price information
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        
        logger.info(f"Fetching latest price for {symbol} from Alpha Vantage")
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "Global Quote" not in data:
                error_message = data.get("Error Message", "Unknown error")
                logger.error(f"API error: {error_message}")
                return {}
            
            quote = data["Global Quote"]
            
            # Extract and format relevant information
            latest_price = {
                "symbol": quote.get("01. symbol", symbol),
                "price": float(quote.get("05. price", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "timestamp": datetime.now().isoformat(),
                "change": float(quote.get("09. change", 0)),
                "change_percent": quote.get("10. change percent", "0%").strip("%"),
            }
            
            return latest_price
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching latest price from Alpha Vantage: {e}")
            return {}


class YahooFinanceAPI(MarketDataFetcher):
    """Implementation for Yahoo Finance API (unofficial)."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.
        
        Args:
            symbol: The stock symbol
            interval: Time interval between data points ('1d', '1h', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with historical price data
        """
        # Default date range: last 30 days to today
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Convert dates to UNIX timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Map intervals to Yahoo Finance's format
        interval_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
        }
        
        yf_interval = interval_mapping.get(interval, "1d")
        
        params = {
            "symbol": symbol,
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": yf_interval,
            "includePrePost": "true",
        }
        
        logger.info(f"Fetching {interval} data for {symbol} from Yahoo Finance")
        
        try:
            url = f"{self.BASE_URL}/{symbol}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            result = data.get("chart", {}).get("result", None)
            if not result:
                error = data.get("chart", {}).get("error", "Unknown error")
                logger.error(f"Yahoo Finance API error: {error}")
                return pd.DataFrame()
            
            # Extract data
            quote_data = result[0]
            timestamps = quote_data.get("timestamp", [])
            quote = quote_data.get("indicators", {}).get("quote", [{}])[0]
            
            # Create DataFrame
            df = pd.DataFrame({
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })
            
            # Add timestamp as index
            df.index = pd.to_datetime([datetime.fromtimestamp(ts) for ts in timestamps])
            df.index.name = "timestamp"
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def fetch_latest_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch latest price for a stock from Yahoo Finance.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Dictionary with latest price information
        """
        try:
            # Use the same API but only request the latest data point
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            df = self.fetch_historical_data(symbol, interval="1d", start_date=start_date, end_date=end_date)
            
            if df.empty:
                return {}
            
            # Get the latest row
            latest_data = df.iloc[-1]
            
            latest_price = {
                "symbol": symbol,
                "price": float(latest_data["close"]),
                "open": float(latest_data["open"]),
                "high": float(latest_data["high"]),
                "low": float(latest_data["low"]),
                "volume": int(latest_data["volume"]),
                "timestamp": latest_data.name.isoformat(),
            }
            
            return latest_price
        
        except Exception as e:
            logger.error(f"Error fetching latest price from Yahoo Finance: {e}")
            return {}


# Factory function to get the appropriate API client
def get_data_api(api_name: str = "alpha_vantage", api_key: Optional[str] = None) -> MarketDataFetcher:
    """
    Factory function to return the appropriate API client.
    
    Args:
        api_name: Name of the API to use (alpha_vantage, yahoo_finance)
        api_key: API key for authentication
        
    Returns:
        Instance of a MarketDataFetcher subclass
    """
    # If no API key is provided, get the appropriate key from environment variables
    if not api_key:
        if api_name.lower() == "alpha_vantage":
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        elif api_name.lower() == "yahoo_finance":
            api_key = os.getenv("YAHOO_FINANCE_API_KEY")
    
    api_mapping = {
        "alpha_vantage": AlphaVantageAPI,
        "yahoo_finance": YahooFinanceAPI,
    }
    
    api_class = api_mapping.get(api_name.lower())
    
    if not api_class:
        logger.warning(f"Unknown API: {api_name}. Using Alpha Vantage as default.")
        api_class = AlphaVantageAPI
    
    return api_class(api_key=api_key)