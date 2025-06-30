"""
Data service for fetching and caching market data
"""
import pandas as pd
import yfinance as yf
from typing import Optional, Dict
from datetime import datetime, timedelta
import asyncio
import logging
from functools import lru_cache

from app.core.config import settings
from app.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=5)

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        days: int = 100
    ) -> pd.DataFrame:
        """
        Get historical market data with caching
        """
        cache_key = f"{symbol}_{timeframe}_{days}"

        # Check cache
        if (
            cache_key in self._cache
            and cache_key in self._cache_timestamp
            and datetime.now() - self._cache_timestamp[cache_key] < self._cache_duration
        ):
            logger.info(f"Cache hit for {cache_key}")
            return self._cache[cache_key]

        try:
            # Fetch new data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period=f"{days}d",
                interval=timeframe
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Prepare data
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={'adj close': 'adj_close'})

            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)

            # Update cache
            self._cache[cache_key] = df
            self._cache_timestamp[cache_key] = datetime.now()

            logger.info(f"Successfully fetched data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get basic information about a symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "exchange": info.get("exchange", ""),
            }

        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "name": symbol,
                "error": str(e)
            }
