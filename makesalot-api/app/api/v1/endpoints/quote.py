"""
Quote and Market Data Endpoints for MakesALot Trading API
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..endpoints.fetcher import get_data_api

logger = logging.getLogger(__name__)

router = APIRouter()

# Response Models
class QuoteResponse(BaseModel):
    symbol: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    change: float
    change_percent: float
    timestamp: str
    market_cap: Optional[str] = None
    pe_ratio: Optional[float] = None
    data_source: str

class QuickQuoteResponse(BaseModel):
    symbol: str
    price: float
    change_percent: float
    volume: int
    timestamp: str

@router.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_enhanced_quote(symbol: str):
    """
    Get enhanced quote with additional market data
    """
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Fetching enhanced quote for {symbol}")
        
        # Try multiple data sources
        data_fetcher = None
        quote_data = {}
        
        # Try Polygon first
        try:
            data_fetcher = get_data_api('polygon')
            quote_data = data_fetcher.fetch_latest_price(symbol)
            data_source = "polygon"
        except Exception as e:
            logger.warning(f"Polygon failed for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        if not quote_data:
            try:
                data_fetcher = get_data_api('yahoo_finance')
                quote_data = data_fetcher.fetch_latest_price(symbol)
                data_source = "yahoo_finance"
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
        
        # Fallback to Alpha Vantage
        if not quote_data:
            try:
                data_fetcher = get_data_api('alpha_vantage')
                quote_data = data_fetcher.fetch_latest_price(symbol)
                data_source = "alpha_vantage"
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        if not quote_data:
            # Generate mock quote for demo
            quote_data = generate_mock_quote(symbol)
            data_source = "mock_data"
        
        # Calculate change percentage if not provided
        price = quote_data.get('price', 0)
        open_price = quote_data.get('open', price)
        change = price - open_price
        change_percent = (change / open_price * 100) if open_price != 0 else 0
        
        response = QuoteResponse(
            symbol=symbol,
            price=price,
            open=open_price,
            high=quote_data.get('high', price),
            low=quote_data.get('low', price),
            volume=quote_data.get('volume', 1000000),
            change=change,
            change_percent=change_percent,
            timestamp=quote_data.get('timestamp', datetime.now().isoformat()),
            data_source=data_source
        )
        
        logger.info(f"Enhanced quote for {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
        return response
        
    except Exception as e:
        logger.error(f"Enhanced quote error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch quote for {symbol}: {str(e)}"
        )

@router.get("/quick-quote/{symbol}", response_model=QuickQuoteResponse)
async def get_quick_quote(symbol: str):
    """
    Get quick quote with basic information
    """
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Fetching quick quote for {symbol}")
        
        # Try to get recent historical data for quick quote
        data_fetcher = get_data_api('polygon')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        try:
            historical_data = data_fetcher.fetch_historical_data(
                symbol=symbol,
                interval="1d",
                start_date=start_date,
                end_date=end_date
            )
            
            if not historical_data.empty:
                latest = historical_data.iloc[-1]
                prev = historical_data.iloc[-2] if len(historical_data) > 1 else latest
                
                price = float(latest['close'])
                prev_price = float(prev['close'])
                change_percent = ((price - prev_price) / prev_price) * 100
                volume = int(latest['volume'])
                
                response = QuickQuoteResponse(
                    symbol=symbol,
                    price=price,
                    change_percent=change_percent,
                    volume=volume,
                    timestamp=datetime.now().isoformat()
                )
                
                logger.info(f"Quick quote for {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
                return response
        except Exception as e:
            logger.warning(f"Historical data failed for quick quote: {e}")
        
        # Fallback to latest price API
        try:
            quote_data = data_fetcher.fetch_latest_price(symbol)
            if quote_data:
                price = quote_data.get('price', 0)
                change_percent = quote_data.get('change_percent', 0)
                
                # Parse change_percent if it's a string
                if isinstance(change_percent, str):
                    change_percent = float(change_percent.replace('%', ''))
                
                response = QuickQuoteResponse(
                    symbol=symbol,
                    price=price,
                    change_percent=change_percent,
                    volume=quote_data.get('volume', 1000000),
                    timestamp=quote_data.get('timestamp', datetime.now().isoformat())
                )
                
                logger.info(f"Quick quote for {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
                return response
        except Exception as e:
            logger.warning(f"Latest price API failed: {e}")
        
        # Generate mock quote
        mock_quote = generate_mock_quote(symbol)
        price = mock_quote['price']
        open_price = mock_quote['open']
        change_percent = ((price - open_price) / open_price) * 100
        
        response = QuickQuoteResponse(
            symbol=symbol,
            price=price,
            change_percent=change_percent,
            volume=mock_quote['volume'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Mock quick quote for {symbol}: ${price:.2f} ({change_percent:+.2f}%)")
        return response
        
    except Exception as e:
        logger.error(f"Quick quote error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch quick quote for {symbol}: {str(e)}"
        )

@router.get("/market-status")
async def get_market_status():
    """
    Get current market status
    """
    try:
        now = datetime.now()
        
        # Simple market hours check (US Eastern Time approximation)
        # This is a simplified version - real implementation would need timezone handling
        hour = now.hour
        weekday = now.weekday()
        
        # Monday = 0, Sunday = 6
        is_weekend = weekday >= 5
        
        # Market hours: 9:30 AM - 4:00 PM ET (simplified as 9-16 UTC)
        is_market_hours = 14 <= hour <= 21 and not is_weekend  # Approximate UTC conversion
        
        market_status = "open" if is_market_hours else "closed"
        
        # Next market open/close
        if is_market_hours:
            next_event = "Market closes at 4:00 PM ET"
        else:
            next_event = "Market opens at 9:30 AM ET"
        
        return {
            "status": market_status,
            "is_open": is_market_hours,
            "next_event": next_event,
            "timestamp": now.isoformat(),
            "timezone": "UTC",
            "note": "Market hours are approximate and for demonstration"
        }
        
    except Exception as e:
        logger.error(f"Market status error: {e}")
        return {
            "status": "unknown",
            "is_open": False,
            "next_event": "Unable to determine market status",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/batch-quotes")
async def get_batch_quotes(
    symbols: str = Query(..., description="Comma-separated list of symbols")
):
    """
    Get quotes for multiple symbols
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if len(symbol_list) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 symbols allowed per request"
            )
        
        logger.info(f"Fetching batch quotes for {len(symbol_list)} symbols")
        
        quotes = {}
        errors = {}
        
        for symbol in symbol_list:
            try:
                # Try quick quote for each symbol
                data_fetcher = get_data_api('polygon')
                quote_data = data_fetcher.fetch_latest_price(symbol)
                
                if quote_data:
                    price = quote_data.get('price', 0)
                    change_percent = quote_data.get('change_percent', 0)
                    
                    quotes[symbol] = {
                        "price": price,
                        "change_percent": change_percent,
                        "volume": quote_data.get('volume', 0),
                        "timestamp": quote_data.get('timestamp', datetime.now().isoformat())
                    }
                else:
                    # Mock quote
                    mock_quote = generate_mock_quote(symbol)
                    price = mock_quote['price']
                    open_price = mock_quote['open']
                    change_percent = ((price - open_price) / open_price) * 100
                    
                    quotes[symbol] = {
                        "price": price,
                        "change_percent": change_percent,
                        "volume": mock_quote['volume'],
                        "timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                errors[symbol] = str(e)
                logger.warning(f"Failed to fetch quote for {symbol}: {e}")
        
        return {
            "quotes": quotes,
            "errors": errors,
            "total_requested": len(symbol_list),
            "successful": len(quotes),
            "failed": len(errors),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch quotes error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch quotes failed: {str(e)}"
        )

def generate_mock_quote(symbol: str) -> Dict:
    """Generate realistic mock quote data"""
    import random
    
    # Base prices for known symbols
    base_prices = {
        'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
        'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400,
        'SPY': 450, 'QQQ': 380, 'VOO': 400
    }
    
    base_price = base_prices.get(symbol, 100 + random.random() * 200)
    
    # Add some random movement
    change_percent = (random.random() - 0.5) * 6  # -3% to +3%
    price = base_price * (1 + change_percent / 100)
    
    # Generate OHLC
    open_price = base_price * (1 + (random.random() - 0.5) * 0.02)
    high = max(price, open_price) * (1 + random.random() * 0.02)
    low = min(price, open_price) * (1 - random.random() * 0.02)
    volume = int(500000 + random.random() * 2000000)
    
    return {
        'symbol': symbol,
        'price': price,
        'open': open_price,
        'high': high,
        'low': low,
        'volume': volume,
        'timestamp': datetime.now().isoformat()
    }