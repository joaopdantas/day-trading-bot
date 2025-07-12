"""
Utility Endpoints for MakesALot Trading API
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..endpoints.fetcher import get_data_api

logger = logging.getLogger(__name__)

router = APIRouter()

# Response Models
class SymbolValidationResponse(BaseModel):
    symbol: str
    is_valid: bool
    exists: bool
    symbol_type: str
    exchange: Optional[str] = None
    company_name: Optional[str] = None
    error: Optional[str] = None

class SymbolSearchResponse(BaseModel):
    symbol: str
    name: str
    exchange: str
    type: str
    score: float

class MarketDataResponse(BaseModel):
    available_endpoints: List[str]
    supported_symbols: List[str]
    api_status: Dict[str, str]
    rate_limits: Dict[str, str]

@router.get("/validate-symbol/{symbol}", response_model=SymbolValidationResponse)
async def validate_symbol(symbol: str):
    """
    Validate if a symbol exists and is tradeable
    """
    try:
        symbol = symbol.upper().strip()
        logger.info(f"Validating symbol: {symbol}")
        
        # Basic format validation
        if not re.match(r'^[A-Z0-9.-]{1,10}$', symbol):
            return SymbolValidationResponse(
                symbol=symbol,
                is_valid=False,
                exists=False,
                symbol_type="invalid",
                error="Invalid symbol format"
            )
        
        # Determine symbol type
        symbol_type = determine_symbol_type(symbol)
        
        # Try to fetch data to validate existence
        exists = False
        company_name = None
        exchange = None
        error = None
        
        # Try multiple data sources
        data_sources = ['polygon', 'yahoo_finance', 'alpha_vantage']
        
        for api_name in data_sources:
            try:
                data_fetcher = get_data_api(api_name)
                quote_data = data_fetcher.fetch_latest_price(symbol)
                
                if quote_data and quote_data.get('price', 0) > 0:
                    exists = True
                    company_name = quote_data.get('name', None)
                    exchange = guess_exchange(symbol, symbol_type)
                    break
                    
            except Exception as e:
                logger.warning(f"Validation failed with {api_name}: {e}")
                continue
        
        # If no data found, check if it's a known symbol
        if not exists:
            exists = is_known_symbol(symbol)
            if exists:
                exchange = guess_exchange(symbol, symbol_type)
        
        response = SymbolValidationResponse(
            symbol=symbol,
            is_valid=True,  # Format is valid even if symbol doesn't exist
            exists=exists,
            symbol_type=symbol_type,
            exchange=exchange,
            company_name=company_name,
            error=error
        )
        
        logger.info(f"Symbol validation for {symbol}: exists={exists}, type={symbol_type}")
        return response
        
    except Exception as e:
        logger.error(f"Symbol validation error for {symbol}: {e}")
        return SymbolValidationResponse(
            symbol=symbol,
            is_valid=False,
            exists=False,
            symbol_type="unknown",
            error=str(e)
        )

@router.get("/search-symbols")
async def search_symbols(
    query: str = Query(..., min_length=1, max_length=50, description="Search query")
):
    """
    Search for symbols by name or ticker
    """
    try:
        query = query.strip().upper()
        logger.info(f"Searching symbols for: {query}")
        
        # This is a simplified search - in production, you'd use a proper financial data API
        results = []
        
        # Search in known symbols
        known_symbols = get_known_symbols()
        
        for symbol_info in known_symbols:
            symbol = symbol_info['symbol']
            name = symbol_info['name']
            
            # Calculate search score
            score = 0.0
            
            # Exact symbol match
            if symbol == query:
                score = 1.0
            # Symbol starts with query
            elif symbol.startswith(query):
                score = 0.8
            # Symbol contains query
            elif query in symbol:
                score = 0.6
            # Name contains query
            elif query.lower() in name.lower():
                score = 0.4
            
            if score > 0:
                results.append(SymbolSearchResponse(
                    symbol=symbol,
                    name=name,
                    exchange=symbol_info.get('exchange', 'NASDAQ'),
                    type=symbol_info.get('type', 'stock'),
                    score=score
                ))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:20]  # Limit to top 20 results
        
        logger.info(f"Found {len(results)} symbols for query: {query}")
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Symbol search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Symbol search failed: {str(e)}"
        )

@router.get("/market-data-info", response_model=MarketDataResponse)
async def get_market_data_info():
    """
    Get information about available market data and API status
    """
    try:
        # Check API status for different providers
        api_status = {}
        
        # Test Polygon API
        try:
            polygon_api = get_data_api('polygon')
            test_data = polygon_api.fetch_latest_price('AAPL')
            api_status['polygon'] = 'operational' if test_data else 'limited'
        except Exception:
            api_status['polygon'] = 'unavailable'
        
        # Test Yahoo Finance API
        try:
            yahoo_api = get_data_api('yahoo_finance')
            test_data = yahoo_api.fetch_latest_price('AAPL')
            api_status['yahoo_finance'] = 'operational' if test_data else 'limited'
        except Exception:
            api_status['yahoo_finance'] = 'unavailable'
        
        # Test Alpha Vantage API
        try:
            alpha_api = get_data_api('alpha_vantage')
            test_data = alpha_api.fetch_latest_price('AAPL')
            api_status['alpha_vantage'] = 'operational' if test_data else 'limited'
        except Exception:
            api_status['alpha_vantage'] = 'unavailable'
        
        response = MarketDataResponse(
            available_endpoints=[
                "/api/v1/analyze",
                "/api/v1/simple-analyze",
                "/api/v1/quote/{symbol}",
                "/api/v1/quick-quote/{symbol}",
                "/api/v1/chart/data/{symbol}",
                "/api/v1/utils/validate-symbol/{symbol}",
                "/api/v1/utils/search-symbols"
            ],
            supported_symbols=[
                "US Stocks (NYSE, NASDAQ)",
                "Major Indices (^GSPC, ^IXIC, ^DJI)",
                "ETFs (SPY, QQQ, VOO)",
                "Crypto (BTC-USD, ETH-USD)",
                "International symbols (limited)"
            ],
            api_status=api_status,
            rate_limits={
                "polygon": "5 requests/minute (free tier)",
                "yahoo_finance": "Unlimited (unofficial)",
                "alpha_vantage": "5 requests/minute (free tier)",
                "makesalot_api": "No limits (internal)"
            }
        )
        
        logger.info("Market data info retrieved successfully")
        return response
        
    except Exception as e:
        logger.error(f"Market data info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market data info: {str(e)}"
        )

@router.get("/health-check")
async def health_check():
    """
    Comprehensive health check for all services
    """
    try:
        health_status = {
            "service": "MakesALot Trading API",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {}
        }
        
        # Check data sources
        data_sources = ['polygon', 'yahoo_finance', 'alpha_vantage']
        
        for source in data_sources:
            try:
                api = get_data_api(source)
                # Quick test with a known symbol
                test_result = api.fetch_latest_price('AAPL')
                
                if test_result and test_result.get('price', 0) > 0:
                    health_status['components'][source] = {
                        "status": "healthy",
                        "response_time": "< 1s",
                        "last_check": datetime.now().isoformat()
                    }
                else:
                    health_status['components'][source] = {
                        "status": "degraded",
                        "message": "API responding but no data",
                        "last_check": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                health_status['components'][source] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        # Check strategies
        try:
            from ..endpoints.strategies import MLTradingStrategy, TechnicalAnalysisStrategy
            
            # Test strategy initialization
            ml_strategy = MLTradingStrategy()
            tech_strategy = TechnicalAnalysisStrategy()
            
            health_status['components']['strategies'] = {
                "status": "healthy",
                "available": ["ml_trading", "technical", "rsi_divergence"],
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            health_status['components']['strategies'] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Check technical indicators
        try:
            from ..indicators.technical import TechnicalIndicators
            
            # Test indicator calculation
            import pandas as pd
            import numpy as np
            
            test_data = pd.Series(np.random.randn(50).cumsum() + 100)
            rsi = TechnicalIndicators.calculate_rsi(test_data)
            
            if not rsi.empty:
                health_status['components']['technical_indicators'] = {
                    "status": "healthy",
                    "last_check": datetime.now().isoformat()
                }
            else:
                health_status['components']['technical_indicators'] = {
                    "status": "degraded",
                    "message": "Indicators calculating but empty results",
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            health_status['components']['technical_indicators'] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Determine overall status
        component_statuses = [comp.get('status', 'unknown') for comp in health_status['components'].values()]
        
        if all(status == 'healthy' for status in component_statuses):
            health_status['status'] = 'healthy'
        elif any(status == 'unhealthy' for status in component_statuses):
            health_status['status'] = 'degraded'
        else:
            health_status['status'] = 'healthy'
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "service": "MakesALot Trading API",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/supported-symbols")
async def get_supported_symbols():
    """
    Get list of well-supported symbols
    """
    try:
        symbols = get_known_symbols()
        
        return {
            "total_symbols": len(symbols),
            "categories": {
                "large_cap_stocks": [s for s in symbols if s.get('type') == 'large_cap'],
                "tech_stocks": [s for s in symbols if s.get('sector') == 'technology'],
                "indices": [s for s in symbols if s.get('type') == 'index'],
                "etfs": [s for s in symbols if s.get('type') == 'etf'],
                "crypto": [s for s in symbols if s.get('type') == 'crypto']
            },
            "recommended_for_testing": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                "SPY", "QQQ", "BTC-USD", "ETH-USD"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Supported symbols error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported symbols: {str(e)}"
        )

def determine_symbol_type(symbol: str) -> str:
    """Determine the type of financial symbol"""
    
    # Index symbols (usually start with ^)
    if symbol.startswith('^'):
        return "index"
    
    # Crypto symbols (usually end with -USD)
    if symbol.endswith('-USD') or symbol.endswith('-BTC') or symbol.endswith('-ETH'):
        return "crypto"
    
    # ETF patterns
    etf_patterns = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG']
    if symbol in etf_patterns or symbol.endswith('ETF'):
        return "etf"
    
    # Large cap stocks (simplified)
    large_cap_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
        'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'TMO', 'WMT'
    ]
    if symbol in large_cap_symbols:
        return "large_cap"
    
    # Default to stock
    if len(symbol) <= 5 and symbol.isalpha():
        return "stock"
    
    return "unknown"

def guess_exchange(symbol: str, symbol_type: str) -> str:
    """Guess the exchange based on symbol and type"""
    
    if symbol_type == "crypto":
        return "CRYPTO"
    
    if symbol_type == "index":
        return "INDEX"
    
    # Common NASDAQ symbols
    nasdaq_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'QQQ', 'NFLX', 'ADBE', 'INTC', 'CMCSA', 'CSCO', 'GOOG'
    ]
    
    if symbol in nasdaq_symbols:
        return "NASDAQ"
    
    # Default to NYSE for other stocks
    return "NYSE"

def is_known_symbol(symbol: str) -> bool:
    """Check if symbol is in our known symbols list"""
    known_symbols = get_known_symbols()
    return any(s['symbol'] == symbol for s in known_symbols)

def get_known_symbols() -> List[Dict]:
    """Get list of well-known symbols with metadata"""
    return [
        # Large Cap Tech Stocks
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "automotive"},
        {"symbol": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ", "type": "large_cap", "sector": "entertainment"},
        
        # Traditional Large Caps
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "type": "large_cap", "sector": "financial"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE", "type": "large_cap", "sector": "healthcare"},
        {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "financial"},
        {"symbol": "PG", "name": "Procter & Gamble Co.", "exchange": "NYSE", "type": "large_cap", "sector": "consumer"},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "healthcare"},
        {"symbol": "HD", "name": "Home Depot Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "retail"},
        {"symbol": "MA", "name": "Mastercard Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "financial"},
        {"symbol": "BAC", "name": "Bank of America Corp.", "exchange": "NYSE", "type": "large_cap", "sector": "financial"},
        
        # Popular ETFs
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE", "type": "etf", "sector": "index"},
        {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "type": "etf", "sector": "technology"},
        {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE", "type": "etf", "sector": "small_cap"},
        {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSE", "type": "etf", "sector": "broad_market"},
        {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "exchange": "NYSE", "type": "etf", "sector": "large_cap"},
        
        # Major Indices
        {"symbol": "^GSPC", "name": "S&P 500 Index", "exchange": "INDEX", "type": "index", "sector": "broad_market"},
        {"symbol": "^IXIC", "name": "NASDAQ Composite", "exchange": "INDEX", "type": "index", "sector": "technology"},
        {"symbol": "^DJI", "name": "Dow Jones Industrial Average", "exchange": "INDEX", "type": "index", "sector": "industrial"},
        {"symbol": "^RUT", "name": "Russell 2000 Index", "exchange": "INDEX", "type": "index", "sector": "small_cap"},
        
        # Cryptocurrencies
        {"symbol": "BTC-USD", "name": "Bitcoin", "exchange": "CRYPTO", "type": "crypto", "sector": "cryptocurrency"},
        {"symbol": "ETH-USD", "name": "Ethereum", "exchange": "CRYPTO", "type": "crypto", "sector": "cryptocurrency"},
        {"symbol": "ADA-USD", "name": "Cardano", "exchange": "CRYPTO", "type": "crypto", "sector": "cryptocurrency"},
        {"symbol": "SOL-USD", "name": "Solana", "exchange": "CRYPTO", "type": "crypto", "sector": "cryptocurrency"},
        
        # Additional Popular Stocks
        {"symbol": "WMT", "name": "Walmart Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "retail"},
        {"symbol": "DIS", "name": "Walt Disney Co.", "exchange": "NYSE", "type": "large_cap", "sector": "entertainment"},
        {"symbol": "KO", "name": "Coca-Cola Co.", "exchange": "NYSE", "type": "large_cap", "sector": "beverage"},
        {"symbol": "PFE", "name": "Pfizer Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "pharmaceutical"},
        {"symbol": "INTC", "name": "Intel Corporation", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "AMD", "name": "Advanced Micro Devices", "exchange": "NASDAQ", "type": "large_cap", "sector": "technology"},
        {"symbol": "CRM", "name": "Salesforce Inc.", "exchange": "NYSE", "type": "large_cap", "sector": "software"},
        {"symbol": "ORCL", "name": "Oracle Corporation", "exchange": "NYSE", "type": "large_cap", "sector": "software"}
    ]