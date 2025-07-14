"""
Chart Data by Months Endpoint - Addition to chart_data.py
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import Query
from pydantic import BaseModel

# Adicionar estas classes e endpoint ao arquivo chart_data.py existente

class MonthlyChartDataPoint(BaseModel):
    date: str  # YYYY-MM-DD format
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_percent: Optional[float] = None

class MonthlyChartResponse(BaseModel):
    symbol: str
    months: int
    data: List[MonthlyChartDataPoint]
    total_points: int
    start_date: str
    end_date: str
    current_price: float
    period_change: float
    period_change_percent: float
    highest_price: float
    lowest_price: float
    average_volume: int
    data_source: str
    timestamp: str

@router.get("/monthly-data/{symbol}", response_model=MonthlyChartResponse)
async def get_monthly_chart_data(
    symbol: str,
    months: int = Query(3, ge=1, le=12, description="Number of months (1, 3, or 12)"),
    include_volume: bool = Query(True, description="Include volume data")
):
    """
    Get chart data for specified number of months (1, 3, or 12)
    
    Args:
        symbol: Stock symbol (e.g., MSFT, AAPL)
        months: Number of months to retrieve (1, 3, or 12)
        include_volume: Whether to include volume data
    
    Returns:
        Chart data with OHLCV information for the specified period
    """
    try:
        symbol = symbol.upper().strip()
        
        # Validate months parameter
        if months not in [1, 3, 12]:
            raise HTTPException(
                status_code=400, 
                detail="Months parameter must be 1, 3, or 12"
            )
        
        logger.info(f"Fetching {months} months of chart data for {symbol}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)  # Approximate months to days
        
        # Adjust for weekends and get more data to ensure we have enough trading days
        start_date = start_date - timedelta(days=10)  # Buffer for weekends/holidays
        
        # Determine appropriate interval based on months
        if months == 1:
            interval = "1d"  # Daily data for 1 month
        elif months == 3:
            interval = "1d"  # Daily data for 3 months
        else:  # 12 months
            interval = "1d"  # Daily data for 12 months (could use weekly but keeping daily)
        
        # Fetch data from multiple sources
        historical_data = pd.DataFrame()
        data_source = "unknown"
        
        # Try different APIs in order of preference
        api_sources = ['polygon', 'yahoo_finance', 'alpha_vantage']
        
        for api_name in api_sources:
            try:
                data_fetcher = get_data_api(api_name)
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
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
            historical_data = generate_mock_monthly_data(symbol, months)
            data_source = "mock_data"
        
        # Filter to exact number of months from end_date
        actual_start_date = end_date - timedelta(days=months * 30)
        historical_data = historical_data[historical_data.index >= actual_start_date]
        
        # Ensure we have data
        if historical_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol} in the last {months} months"
            )
        
        # Calculate additional metrics
        current_price = float(historical_data['close'].iloc[-1])
        start_price = float(historical_data['close'].iloc[0])
        period_change = current_price - start_price
        period_change_percent = (period_change / start_price) * 100 if start_price != 0 else 0
        
        highest_price = float(historical_data['high'].max())
        lowest_price = float(historical_data['low'].min())
        average_volume = int(historical_data['volume'].mean()) if 'volume' in historical_data.columns else 1000000
        
        # Convert to response format with daily change percentages
        chart_points = []
        prev_close = None
        
        for timestamp, row in historical_data.iterrows():
            current_close = float(row['close'])
            
            # Calculate daily change percentage
            daily_change_percent = None
            if prev_close is not None and prev_close != 0:
                daily_change_percent = ((current_close - prev_close) / prev_close) * 100
            
            chart_point = MonthlyChartDataPoint(
                date=timestamp.strftime("%Y-%m-%d"),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=current_close,
                volume=int(row.get('volume', average_volume)),
                change_percent=daily_change_percent
            )
            
            chart_points.append(chart_point)
            prev_close = current_close
        
        # Build response
        response = MonthlyChartResponse(
            symbol=symbol,
            months=months,
            data=chart_points,
            total_points=len(chart_points),
            start_date=historical_data.index[0].strftime("%Y-%m-%d"),
            end_date=historical_data.index[-1].strftime("%Y-%m-%d"),
            current_price=current_price,
            period_change=period_change,
            period_change_percent=period_change_percent,
            highest_price=highest_price,
            lowest_price=lowest_price,
            average_volume=average_volume,
            data_source=data_source,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Monthly chart data for {symbol}: {len(chart_points)} points, "
                   f"{months} months, {period_change_percent:+.2f}% change")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monthly chart data error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch monthly chart data: {str(e)}"
        )

def generate_mock_monthly_data(symbol: str, months: int) -> pd.DataFrame:
    """Generate realistic mock data for specified months"""
    import numpy as np
    
    # Base prices for known symbols
    base_prices = {
        'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
        'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400,
        'SPY': 450, 'QQQ': 380, 'VOO': 400
    }
    
    base_price = base_prices.get(symbol, 100 + np.random.random() * 200)
    
    # Generate dates (business days only)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    # Create business day range
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    # Generate realistic price movements
    n_days = len(date_range)
    
    # Adjust volatility based on months (longer periods = more potential for larger moves)
    daily_volatility = 0.015 if months == 1 else 0.018 if months == 3 else 0.020
    
    # Generate returns with some trend
    trend = np.random.normal(0.0003, 0.0001)  # Small positive bias
    returns = np.random.normal(trend, daily_volatility, n_days)
    
    # Add some momentum and mean reversion
    for i in range(1, len(returns)):
        momentum = returns[i-1] * 0.1  # 10% momentum
        returns[i] += momentum
    
    # Generate prices
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(date_range, prices)):
        open_price = prices[i-1] if i > 0 else close
        
        # Generate realistic intraday range
        daily_range = abs(close - open_price) + (close * 0.01 * np.random.random())
        
        # High and low with some randomness
        if close > open_price:  # Up day
            high = close + (daily_range * np.random.random() * 0.5)
            low = open_price - (daily_range * np.random.random() * 0.3)
        else:  # Down day
            high = open_price + (daily_range * np.random.random() * 0.3)
            low = close - (daily_range * np.random.random() * 0.5)
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume on big moves)
        price_change_pct = abs((close - open_price) / open_price) if open_price != 0 else 0
        base_volume = 1000000
        volume_multiplier = 1 + (price_change_pct * 2)  # More volume on big moves
        volume = int(base_volume * volume_multiplier * (0.7 + np.random.random() * 0.6))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=date_range)
    logger.info(f"Generated {len(df)} days of mock data for {symbol} ({months} months)")
    return df

# Endpoint para estatísticas rápidas do período
@router.get("/monthly-stats/{symbol}")
async def get_monthly_stats(
    symbol: str,
    months: int = Query(3, ge=1, le=12, description="Number of months (1, 3, or 12)")
):
    """
    Get quick statistics for the specified period
    """
    try:
        symbol = symbol.upper().strip()
        
        if months not in [1, 3, 12]:
            raise HTTPException(status_code=400, detail="Months must be 1, 3, or 12")
        
        logger.info(f"Fetching {months} months stats for {symbol}")
        
        # Get data (reuse the monthly data logic)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 10)
        
        # Try to fetch real data
        historical_data = pd.DataFrame()
        for api_name in ['polygon', 'yahoo_finance']:
            try:
                data_fetcher = get_data_api(api_name)
                historical_data = data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    interval="1d",
                    start_date=start_date,
                    end_date=end_date
                )
                if not historical_data.empty:
                    break
            except Exception:
                continue
        
        if historical_data.empty:
            historical_data = generate_mock_monthly_data(symbol, months)
        
        # Filter to exact months
        actual_start_date = end_date - timedelta(days=months * 30)
        historical_data = historical_data[historical_data.index >= actual_start_date]
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Calculate statistics
        current_price = float(historical_data['close'].iloc[-1])
        start_price = float(historical_data['close'].iloc[0])
        
        period_return = ((current_price - start_price) / start_price) * 100
        
        # Volatility (standard deviation of daily returns)
        daily_returns = historical_data['close'].pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100)  # Annualized
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)
        
        # Best and worst days
        best_day = float(daily_returns.max() * 100) if not daily_returns.empty else 0
        worst_day = float(daily_returns.min() * 100) if not daily_returns.empty else 0
        
        # Trading days with gains vs losses
        positive_days = (daily_returns > 0).sum()
        negative_days = (daily_returns < 0).sum()
        win_rate = (positive_days / len(daily_returns) * 100) if len(daily_returns) > 0 else 0
        
        stats = {
            "symbol": symbol,
            "period_months": months,
            "current_price": current_price,
            "start_price": start_price,
            "period_return_percent": period_return,
            "highest_price": float(historical_data['high'].max()),
            "lowest_price": float(historical_data['low'].min()),
            "volatility_percent": volatility,
            "max_drawdown_percent": max_drawdown,
            "best_day_percent": best_day,
            "worst_day_percent": worst_day,
            "win_rate_percent": win_rate,
            "positive_days": int(positive_days),
            "negative_days": int(negative_days),
            "total_trading_days": len(historical_data),
            "average_daily_volume": int(historical_data['volume'].mean()) if 'volume' in historical_data.columns else 0,
            "period_label": f"{months} month{'s' if months > 1 else ''}",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Stats for {symbol} ({months}m): {period_return:+.2f}% return")
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monthly stats error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monthly stats: {str(e)}"
        )