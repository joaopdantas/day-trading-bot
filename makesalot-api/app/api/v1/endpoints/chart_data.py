"""
Chart Data Endpoints for MakesALot Trading API
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..endpoints.fetcher import get_data_api

logger = logging.getLogger(__name__)

router = APIRouter()

# Response Models
class ChartDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class ChartDataResponse(BaseModel):
    symbol: str
    interval: str
    data: List[ChartDataPoint]
    total_points: int
    start_date: str
    end_date: str
    data_source: str

@router.get("/data/{symbol}", response_model=ChartDataResponse)
async def get_chart_data(
    symbol: str,
    interval: str = Query("1d", description="Time interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)"),
    period: str = Query("3m", description="Time period (1d, 5d, 1mo, 3m, 6m, 1y, 2y, 5y, 10y, ytd, max)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get historical chart data for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Fetching chart data for {symbol}, interval: {interval}, period: {period}")
        
        # Calculate date range
        end_dt = datetime.now()
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        # Calculate start date based on period
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        else:
            start_dt = calculate_start_date(end_dt, period)
        
        # Validate date range
        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        # Fetch data from multiple sources
        historical_data = pd.DataFrame()
        data_source = "unknown"
        
        # Try different APIs
        api_sources = ['polygon', 'yahoo_finance', 'alpha_vantage']
        
        for api_name in api_sources:
            try:
                data_fetcher = get_data_api(api_name)
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_dt,
                    end_date=end_dt
                )
                
                if not historical_data.empty:
                    data_source = api_name
                    logger.info(f"Successfully fetched data from {api_name}")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to fetch from {api_name}: {e}")
                continue
        
        if historical_data.empty:
            # Generate mock data for demo purposes
            logger.warning(f"No real data available for {symbol}, generating mock data")
            historical_data = generate_mock_chart_data(symbol, start_dt, end_dt, interval)
            data_source = "mock_data"
        
        # Convert to response format
        chart_points = []
        for timestamp, row in historical_data.iterrows():
            chart_points.append(ChartDataPoint(
                timestamp=timestamp.isoformat(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row.get('volume', 1000000))
            ))
        
        response = ChartDataResponse(
            symbol=symbol,
            interval=interval,
            data=chart_points,
            total_points=len(chart_points),
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            data_source=data_source
        )
        
        logger.info(f"Chart data for {symbol}: {len(chart_points)} points from {data_source}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chart data: {str(e)}"
        )

@router.get("/indicators/{symbol}")
async def get_chart_with_indicators(
    symbol: str,
    interval: str = Query("1d", description="Time interval"),
    period: str = Query("3m", description="Time period"),
    indicators: str = Query("sma20,sma50,rsi", description="Comma-separated list of indicators")
):
    """
    Get chart data with technical indicators overlay
    """
    try:
        symbol = symbol.upper().strip()
        indicator_list = [ind.strip().lower() for ind in indicators.split(',')]
        
        logger.info(f"Fetching chart with indicators for {symbol}: {indicator_list}")
        
        # Get base chart data
        end_dt = datetime.now()
        start_dt = calculate_start_date(end_dt, period)
        
        # Fetch historical data
        historical_data = pd.DataFrame()
        data_source = "unknown"
        
        for api_name in ['polygon', 'yahoo_finance']:
            try:
                data_fetcher = get_data_api(api_name)
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_dt,
                    end_date=end_dt
                )
                
                if not historical_data.empty:
                    data_source = api_name
                    break
                    
            except Exception as e:
                logger.warning(f"API {api_name} failed: {e}")
                continue
        
        if historical_data.empty:
            historical_data = generate_mock_chart_data(symbol, start_dt, end_dt, interval)
            data_source = "mock_data"
        
        # Calculate indicators
        from ..indicators.technical import TechnicalIndicators
        
        close_prices = historical_data['close']
        indicator_data = {}
        
        for indicator in indicator_list:
            try:
                if indicator == 'sma20':
                    indicator_data['sma20'] = TechnicalIndicators.calculate_sma(close_prices, 20).tolist()
                elif indicator == 'sma50':
                    indicator_data['sma50'] = TechnicalIndicators.calculate_sma(close_prices, 50).tolist()
                elif indicator == 'ema12':
                    indicator_data['ema12'] = TechnicalIndicators.calculate_ema(close_prices, 12).tolist()
                elif indicator == 'ema26':
                    indicator_data['ema26'] = TechnicalIndicators.calculate_ema(close_prices, 26).tolist()
                elif indicator == 'rsi':
                    indicator_data['rsi'] = TechnicalIndicators.calculate_rsi(close_prices).tolist()
                elif indicator == 'macd':
                    macd_data = TechnicalIndicators.calculate_macd(close_prices)
                    indicator_data['macd'] = macd_data['macd'].tolist()
                    indicator_data['macd_signal'] = macd_data['signal'].tolist()
                elif indicator.startswith('bb'):
                    bb_data = TechnicalIndicators.calculate_bollinger_bands(close_prices)
                    indicator_data['bb_upper'] = bb_data['upper'].tolist()
                    indicator_data['bb_middle'] = bb_data['middle'].tolist()
                    indicator_data['bb_lower'] = bb_data['lower'].tolist()
                    
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")
                continue
        
        # Build response
        chart_data = []
        timestamps = historical_data.index.tolist()
        
        for i, (timestamp, row) in enumerate(historical_data.iterrows()):
            point_data = {
                "timestamp": timestamp.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row.get('volume', 1000000))
            }
            
            # Add indicator values for this point
            for indicator_name, values in indicator_data.items():
                if i < len(values):
                    value = values[i]
                    if pd.notna(value):
                        point_data[indicator_name] = float(value)
            
            chart_data.append(point_data)
        
        response = {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "data": chart_data,
            "indicators": list(indicator_data.keys()),
            "total_points": len(chart_data),
            "data_source": data_source,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Chart with indicators for {symbol}: {len(chart_data)} points, {len(indicator_data)} indicators")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart with indicators error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chart with indicators: {str(e)}"
        )

@router.get("/volume-profile/{symbol}")
async def get_volume_profile(
    symbol: str,
    period: str = Query("1m", description="Time period for volume profile analysis")
):
    """
    Get volume profile analysis for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Generating volume profile for {symbol}")
        
        # Get chart data
        end_dt = datetime.now()
        start_dt = calculate_start_date(end_dt, period)
        
        # Fetch data
        historical_data = pd.DataFrame()
        for api_name in ['polygon', 'yahoo_finance']:
            try:
                data_fetcher = get_data_api(api_name)
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval="1d",
                    start_date=start_dt,
                    end_date=end_dt
                )
                if not historical_data.empty:
                    break
            except Exception:
                continue
        
        if historical_data.empty:
            historical_data = generate_mock_chart_data(symbol, start_dt, end_dt, "1d")
        
        # Calculate volume profile
        volume_profile = calculate_volume_profile(historical_data)
        
        response = {
            "symbol": symbol,
            "period": period,
            "volume_profile": volume_profile,
            "analysis": {
                "poc_price": volume_profile['poc_price'] if volume_profile else None,
                "value_area_high": volume_profile['value_area_high'] if volume_profile else None,
                "value_area_low": volume_profile['value_area_low'] if volume_profile else None,
                "total_volume": sum(historical_data['volume']) if not historical_data.empty else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Volume profile error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate volume profile: {str(e)}"
        )

def calculate_start_date(end_date: datetime, period: str) -> datetime:
    """Calculate start date based on period string"""
    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "ytd": end_date - datetime(end_date.year, 1, 1),
        "max": timedelta(days=3650)  # Default to 10 years for max
    }
    
    delta = period_map.get(period, timedelta(days=90))  # Default to 3 months
    
    if period == "ytd":
        return datetime(end_date.year, 1, 1)
    else:
        return end_date - delta

def generate_mock_chart_data(symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
    """Generate realistic mock chart data"""
    import numpy as np
    
    # Base prices for known symbols
    base_prices = {
        'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
        'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400
    }
    
    base_price = base_prices.get(symbol, 100 + np.random.random() * 200)
    
    # Calculate frequency based on interval
    freq_map = {
        '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': 'H', '1d': 'D', '1wk': 'W', '1mo': 'M'
    }
    
    freq = freq_map.get(interval, 'D')
    
    # Generate date range
    if interval in ['1m', '5m', '15m', '30m', '1h']:
        # For intraday data, limit to business hours
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        # Filter to business hours (9:30 AM to 4:00 PM ET, simplified)
        date_range = date_range[(date_range.hour >= 9) & (date_range.hour <= 16)]
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate price movements
    n_periods = len(date_range)
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily return ~0.05%, volatility ~2%
    
    # For intraday data, reduce volatility
    if interval in ['1m', '5m', '15m', '30m', '1h']:
        returns = returns * 0.1  # Reduce intraday volatility
    
    # Generate prices
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(date_range, prices)):
        open_price = prices[i-1] if i > 0 else close
        
        # Generate realistic high/low
        daily_range = abs(close - open_price) + (close * 0.01 * np.random.random())
        high = max(open_price, close) + (daily_range * np.random.random())
        low = min(open_price, close) - (daily_range * np.random.random())
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        base_volume = 1000000 if interval == '1d' else 50000
        volume = int(base_volume * (0.5 + np.random.random()))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=date_range)
    logger.info(f"Generated {len(df)} periods of mock data for {symbol}")
    return df

def calculate_volume_profile(data: pd.DataFrame) -> dict:
    """Calculate volume profile for price data"""
    try:
        if data.empty:
            return {}
        
        # Get price range
        price_min = data['low'].min()
        price_max = data['high'].max()
        
        # Create price bins
        n_bins = 50
        price_bins = np.linspace(price_min, price_max, n_bins)
        volume_at_price = {}
        
        # Distribute volume across price levels
        for _, row in data.iterrows():
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            volume = row['volume']
            
            # Find closest bin
            bin_idx = np.digitize(typical_price, price_bins) - 1
            bin_idx = max(0, min(bin_idx, len(price_bins) - 1))
            
            price_level = price_bins[bin_idx]
            volume_at_price[price_level] = volume_at_price.get(price_level, 0) + volume
        
        if not volume_at_price:
            return {}
        
        # Find Point of Control (POC) - price with highest volume
        poc_price = max(volume_at_price.keys(), key=lambda x: volume_at_price[x])
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * 0.7
        
        # Sort by volume and find value area
        sorted_prices = sorted(volume_at_price.keys(), key=lambda x: volume_at_price[x], reverse=True)
        value_area_volume = 0
        value_area_prices = []
        
        for price in sorted_prices:
            value_area_volume += volume_at_price[price]
            value_area_prices.append(price)
            if value_area_volume >= target_volume:
                break
        
        value_area_high = max(value_area_prices) if value_area_prices else poc_price
        value_area_low = min(value_area_prices) if value_area_prices else poc_price
        
        return {
            'price_levels': list(volume_at_price.keys()),
            'volumes': list(volume_at_price.values()),
            'poc_price': float(poc_price),
            'value_area_high': float(value_area_high),
            'value_area_low': float(value_area_low),
            'total_volume': int(total_volume)
        }
        
    except Exception as e:
        logger.error(f"Volume profile calculation error: {e}")
        return {}