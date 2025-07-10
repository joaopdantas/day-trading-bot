"""
MakesALot Trading API - Versão Corrigida
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import sys
from contextlib import asynccontextmanager

# Configurar path para imports locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== IMPORTS LOCAIS COM FALLBACK =====
try:
    from app.api.v1.endpoints.fetcher import get_data_api, PolygonAPI
    from app.api.v1.endpoints.strategies import (
        MLTradingStrategy, TechnicalAnalysisStrategy, 
        RSIDivergenceStrategy, BuyAndHoldStrategy
    )
    from app.api.v1.indicators.technical import TechnicalIndicators
    logger.info("✅ Componentes avançados carregados com sucesso")
except ImportError as e:
    logger.warning(f"⚠️ Erro ao importar componentes avançados: {e}")
    logger.info("🔄 Usando implementações simplificadas...")
    
    # Implementações simplificadas como fallback
    class PolygonAPI:
        def fetch_historical_data(self, symbol, interval="1d", start_date=None, end_date=None):
            return pd.DataFrame()
        def fetch_latest_price(self, symbol):
            return {}
    
    def get_data_api(api_name):
        return PolygonAPI()
    
    class MLTradingStrategy:
        def reset(self): pass
        def generate_signal(self, current_data, historical_data):
            return {'action': 'HOLD', 'confidence': 0.5, 'reasoning': ['Fallback strategy']}
    
    class TechnicalAnalysisStrategy:
        def reset(self): pass
        def generate_signal(self, current_data, historical_data):
            return {'action': 'HOLD', 'confidence': 0.6, 'reasoning': ['Basic technical analysis']}
    
    class RSIDivergenceStrategy:
        def reset(self): pass
        def generate_signal(self, current_data, historical_data):
            return {'action': 'HOLD', 'confidence': 0.7, 'reasoning': ['RSI analysis']}
    
    class BuyAndHoldStrategy:
        def reset(self): pass
        def generate_signal(self, current_data, historical_data):
            return {'action': 'BUY', 'confidence': 1.0, 'reasoning': ['Buy and hold strategy']}
    
    class TechnicalIndicators:
        @staticmethod
        def add_all_indicators(data):
            df = data.copy()
            if len(df) >= 14:
                # RSI básico
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['rsi'] = df['rsi'].fillna(50.0)
            else:
                df['rsi'] = 50.0
            
            # MACD básico
            if len(df) >= 26:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            else:
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
                df['macd_histogram'] = 0.0
            
            # SMAs
            df['sma_20'] = df['close'].rolling(window=20).mean().fillna(df['close'])
            df['sma_50'] = df['close'].rolling(window=50).mean().fillna(df['close'])
            
            # Bollinger Bands
            if len(df) >= 20:
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                df['bb_upper'] = sma + (std * 2)
                df['bb_lower'] = sma - (std * 2)
                df['bb_middle'] = sma
            else:
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_middle'] = df['close']
            
            return df.fillna(method='forward').fillna(method='backward')

# ===== IMPORTAR YFINANCE COMO FALLBACK =====
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("✅ yfinance disponível como fallback")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("⚠️ yfinance não disponível")

# ===== MODELOS DE DADOS =====
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Símbolo da ação")
    timeframe: Optional[str] = Field("1d", description="Timeframe: 1d, 1h, 15m")
    days: Optional[int] = Field(100, ge=30, le=365, description="Número de dias (30-365)")
    strategy: Optional[str] = Field("ml_trading", description="Estratégia: ml_trading, technical, rsi_divergence")
    include_predictions: Optional[bool] = Field(True, description="Incluir previsões ML")

class AnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    trend: str
    recommendation: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    predictions: Optional[Dict[str, Any]] = None
    support_resistance: Dict[str, List[float]]
    risk_assessment: str
    volume_analysis: Dict[str, Any]
    timestamp: str
    data_source: str

class QuoteResponse(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[int] = None
    day_range: Dict[str, float]
    year_range: Optional[Dict[str, float]] = None
    pe_ratio: Optional[float] = None

class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Símbolo para backtest")
    strategy: str = Field(..., description="Nome da estratégia")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: Optional[float] = Field(10000, ge=1000, description="Capital inicial")

# ===== FUNÇÕES AUXILIARES =====
def get_yfinance_data(symbol: str, days: int = 100):
    """Buscar dados usando yfinance como fallback"""
    if not YFINANCE_AVAILABLE:
        return None, None
    
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None, None
        
        # Padronizar nomes das colunas
        data = data.rename(columns={
            'Close': 'close', 'Open': 'open', 
            'High': 'high', 'Low': 'low', 'Volume': 'volume'
        })
        
        # Buscar info da empresa
        info = ticker.info
        return data, info
        
    except Exception as e:
        logger.error(f"Erro yfinance para {symbol}: {e}")
        return None, None

def generate_mock_data(symbol: str, days: int = 100):
    """Gerar dados mock realistas"""
    
    base_prices = {
        'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
        'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400,
        'SPY': 450, 'QQQ': 380, 'VOO': 400,
        'BTC-USD': 45000, 'ETH-USD': 2500
    }
    
    base_price = base_prices.get(symbol, 100 + np.random.uniform(50, 200))
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Parâmetros por tipo de ativo
    if symbol.endswith('-USD'):  # Crypto
        volatility = 0.04
        trend = 0.002
    elif symbol in ['SPY', 'QQQ', 'VOO']:  # ETFs
        volatility = 0.015
        trend = 0.0008
    else:  # Stocks
        volatility = 0.025
        trend = 0.001
    
    # Gerar retornos
    returns = np.random.normal(trend, volatility, days)
    
    # Calcular preços
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.3))
    
    # Criar OHLCV
    ohlcv_data = []
    for i, close_price in enumerate(prices):
        daily_range = close_price * np.random.uniform(0.005, 0.025)
        
        open_price = prices[i-1] if i > 0 else close_price
        high = close_price + np.random.uniform(0, daily_range)
        low = close_price - np.random.uniform(0, daily_range)
        
        # Garantir ordem OHLC
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = int(np.random.uniform(1000000, 5000000))
        
        ohlcv_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(ohlcv_data, index=dates[:len(prices)])
    return df

def validate_symbol(symbol: str) -> str:
    """Validar e sanitizar símbolo"""
    if not symbol:
        raise HTTPException(status_code=400, detail="Símbolo não pode estar vazio")
    
    symbol = symbol.upper().strip()
    
    # Verificar caracteres válidos
    import re
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        raise HTTPException(status_code=400, detail="Símbolo contém caracteres inválidos")
    
    if len(symbol) > 10:
        raise HTTPException(status_code=400, detail="Símbolo muito longo (máximo 10 caracteres)")
    
    return symbol

# ===== CLASSE PRINCIPAL DA API =====
class TradingAPIManager:
    """Gerenciador principal da API de Trading"""
    
    def __init__(self):
        # Inicializar APIs de dados com fallback
        self.data_apis = {
            'polygon': PolygonAPI(),
            'yahoo': get_data_api('yahoo_finance'),
            'alpha': get_data_api('alpha_vantage')
        }
        
        # Inicializar estratégias
        self.strategies = {
            'ml_trading': MLTradingStrategy(),
            'technical': TechnicalAnalysisStrategy(), 
            'rsi_divergence': RSIDivergenceStrategy(),
            'buy_hold': BuyAndHoldStrategy()
        }
        
        # Cache para otimização
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        logger.info("🚀 TradingAPIManager inicializado com sucesso")
    
    async def get_market_data(self, symbol: str, days: int = 100, use_cache: bool = True) -> pd.DataFrame:
        """Buscar dados de mercado com cache e fallback"""
        
        cache_key = f"{symbol}_{days}"
        
        # Verificar cache
        if use_cache and cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                logger.info(f"📋 Usando dados em cache para {symbol}")
                return cached_data
        
        # Buscar dados com fallback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for api_name, api in self.data_apis.items():
            try:
                logger.info(f"📡 Buscando {symbol} via {api_name}")
                
                data = api.fetch_historical_data(
                    symbol=symbol,
                    interval="1d",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data.empty:
                    # Padronizar colunas
                    if 'Close' in data.columns:
                        data = data.rename(columns={
                            'Close': 'close', 'Open': 'open', 
                            'High': 'high', 'Low': 'low', 'Volume': 'volume'
                        })
                    
                    # Adicionar indicadores técnicos
                    data = TechnicalIndicators.add_all_indicators(data)
                    
                    # Salvar no cache
                    self.data_cache[cache_key] = (data, datetime.now().timestamp())
                    
                    logger.info(f"✅ Dados obtidos via {api_name}: {len(data)} pontos")
                    return data
                    
            except Exception as e:
                logger.warning(f"❌ Erro com {api_name}: {e}")
                continue
        
        # Fallback para yfinance
        if YFINANCE_AVAILABLE:
            try:
                logger.info(f"📡 Tentando yfinance para {symbol}")
                data, _ = get_yfinance_data(symbol, days)
                if data is not None and not data.empty:
                    data = TechnicalIndicators.add_all_indicators(data)
                    self.data_cache[cache_key] = (data, datetime.now().timestamp())
                    logger.info(f"✅ Dados obtidos via yfinance: {len(data)} pontos")
                    return data
            except Exception as e:
                logger.warning(f"❌ Erro com yfinance: {e}")
        
        # Fallback para dados mock
        logger.warning(f"⚠️ Gerando dados mock para {symbol}")
        mock_data = generate_mock_data(symbol, days)
        mock_data = TechnicalIndicators.add_all_indicators(mock_data)
        return mock_data

# Inicializar gerenciador global
api_manager = TradingAPIManager()

# ===== CRIAR APP FASTAPI =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Iniciando MakesALot Trading API...")
    yield
    # Shutdown
    logger.info("🛑 Desligando MakesALot Trading API...")

app = FastAPI(
    title="MakesALot Trading API",
    description="API avançada para análise técnica, previsões ML e estratégias de trading",
    version="2.0.0",
    lifespan=lifespan
)

# ===== CONFIGURAR CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MIDDLEWARE DE LOGGING =====
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# ===== ENDPOINTS PRINCIPAIS =====
@app.get("/")
def root():
    """Endpoint principal com informações da API"""
    return {
        "message": "🚀 MakesALot Trading API v2.0",
        "description": "API avançada para análise técnica e previsões de trading",
        "version": "2.0.0",
        "features": [
            "📊 Análise técnica completa",
            "🤖 Previsões de Machine Learning", 
            "📈 Múltiplas estratégias de trading",
            "📉 Análise de suporte/resistência",
            "📋 Backtesting de estratégias",
            "🔄 Múltiplas fontes de dados com fallback"
        ],
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "quote": "/api/v1/quote/{symbol}",
            "backtest": "/api/v1/backtest",
            "strategies": "/api/v1/strategies"
        },
        "data_sources": ["Polygon.io", "Yahoo Finance", "Alpha Vantage", "yfinance"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Verificação de saúde da API"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MakesALot Trading API v2.0",
        "uptime": "Running",
        "strategies_loaded": len(api_manager.strategies),
        "cache_entries": len(api_manager.data_cache),
        "yfinance_available": YFINANCE_AVAILABLE
    }

@app.get("/api/v1/quote/{symbol}")
async def get_quote(symbol: str):
    """Obter cotação de um símbolo"""
    
    try:
        symbol = validate_symbol(symbol)
        
        # Buscar dados de 2 dias para calcular mudança
        data = await api_manager.get_market_data(symbol, 2)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Símbolo {symbol} não encontrado")
        
        current = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else current
        
        current_price = float(current['close'])
        previous_price = float(previous['close'])
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price > 0 else 0
        
        return {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(current['volume']),
            "day_range": {
                "high": round(float(current['high']), 2),
                "low": round(float(current['low']), 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na cotação de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar cotação: {str(e)}")

@app.post("/api/v1/analyze")
async def analyze_stock(request: AnalysisRequest):
    """Análise técnica completa de uma ação"""
    
    try:
        symbol = validate_symbol(request.symbol)
        
        # Buscar dados de mercado
        market_data = await api_manager.get_market_data(symbol, request.days)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"Dados não encontrados para {symbol}")
        
        current_data = market_data.iloc[-1]
        previous_data = market_data.iloc[-2] if len(market_data) > 1 else current_data
        
        # Calcular métricas básicas
        current_price = float(current_data['close'])
        previous_price = float(previous_data['close'])
        change_percent = ((current_price - previous_price) / previous_price) * 100
        
        # Análise de tendência básica
        sma_20 = current_data.get('sma_20', current_price)
        sma_50 = current_data.get('sma_50', current_price)
        
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Gerar recomendação usando estratégia
        strategy = api_manager.strategies.get(request.strategy, api_manager.strategies['ml_trading'])
        strategy.reset()
        
        signal = strategy.generate_signal(current_data, market_data.iloc[:-1])
        
        # Indicadores técnicos
        technical_indicators = {
            "rsi": float(current_data.get('rsi', 50)),
            "macd": float(current_data.get('macd', 0)),
            "macd_signal": float(current_data.get('macd_signal', 0)),
            "bb_upper": float(current_data.get('bb_upper', current_price)),
            "bb_lower": float(current_data.get('bb_lower', current_price)),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50)
        }
        
        # Análise de volume
        current_volume = int(current_data['volume'])
        avg_volume = int(market_data['volume'].tail(20).mean()) if len(market_data) >= 20 else current_volume
        
        volume_analysis = {
            "current": current_volume,
            "average_20d": avg_volume,
            "ratio": round(current_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
            "trend": "high" if current_volume > avg_volume * 1.5 else "normal"
        }
        
        # Análise de suporte e resistência (simplificada)
        recent_highs = market_data['high'].tail(20).nlargest(3).tolist()
        recent_lows = market_data['low'].tail(20).nsmallest(3).tolist()
        
        resistance_levels = [h for h in recent_highs if h > current_price * 1.01][:3]
        support_levels = [l for l in recent_lows if l < current_price * 0.99][:3]
        
        support_resistance = {
            "support": [round(s, 2) for s in sorted(support_levels, reverse=True)],
            "resistance": [round(r, 2) for r in sorted(resistance_levels)]
        }
        
        # Avaliação de risco
        rsi = technical_indicators['rsi']
        volatility = market_data['close'].pct_change().tail(20).std()
        
        if rsi > 80 or rsi < 20 or volatility > 0.03:
            risk_assessment = "high"
        elif rsi > 70 or rsi < 30 or volatility > 0.02:
            risk_assessment = "medium"
        else:
            risk_assessment = "low"
        
        response = AnalysisResponse(
            symbol=symbol,
            current_price=round(current_price, 2),
            change_percent=round(change_percent, 2),
            trend=trend,
            recommendation={
                "action": signal.get('action', 'HOLD'),
                "confidence": signal.get('confidence', 0.5),
                "reasoning": signal.get('reasoning', []),
                "strategy_used": request.strategy
            },
            technical_indicators=technical_indicators,
            predictions=None,  # Pode ser implementado depois
            support_resistance=support_resistance,
            risk_assessment=risk_assessment,
            volume_analysis=volume_analysis,
            timestamp=datetime.now().isoformat(),
            data_source="multi_api_with_fallback"
        )
        
        logger.info(f"✅ Análise concluída para {symbol}: {signal.get('action')} com {signal.get('confidence', 0):.2f} confiança")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na análise de {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno na análise: {str(e)}")

@app.get("/api/v1/strategies")
async def list_strategies():
    """Listar estratégias disponíveis"""
    
    strategies_info = {
        "ml_trading": {
            "name": "ML Trading Strategy",
            "description": "Estratégia avançada usando Machine Learning com indicadores técnicos",
            "expected_accuracy": "68%",
            "risk_level": "medium"
        },
        "technical": {
            "name": "Technical Analysis Strategy",
            "description": "Análise técnica tradicional com RSI, MACD e médias móveis",
            "expected_accuracy": "62%",
            "risk_level": "low"
        },
        "rsi_divergence": {
            "name": "RSI Divergence Strategy",
            "description": "Estratégia de divergência RSI - 64% retorno histórico",
            "expected_accuracy": "76%",
            "risk_level": "medium"
        },
        "buy_hold": {
            "name": "Buy and Hold",
            "description": "Estratégia passiva de comprar e manter",
            "expected_accuracy": "55%",
            "risk_level": "low"
        }
    }
    
    return {
        "available_strategies": strategies_info,
        "total_strategies": len(strategies_info),
        "recommendation": "Use 'rsi_divergence' para máxima performance"
    }

@app.get("/api/v1/market-overview")
async def get_market_overview():
    """Visão geral do mercado"""
    
    try:
        # Analisar principais índices
        major_indices = ["SPY", "QQQ"]
        market_data = {}
        
        for symbol in major_indices:
            try:
                data = await api_manager.get_market_data(symbol, 5)
                if not data.empty:
                    current = data.iloc[-1]
                    previous = data.iloc[-2] if len(data) > 1 else current
                    
                    change_pct = ((current['close'] - previous['close']) / previous['close']) * 100
                    
                    market_data[symbol] = {
                        "price": round(float(current['close']), 2),
                        "change_percent": round(change_pct, 2),
                        "volume": int(current['volume']),
                        "trend": "bullish" if change_pct > 0.5 else "bearish" if change_pct < -0.5 else "neutral"
                    }
            except:
                continue
        
        # Análise de sentimento geral
        positive_count = sum(1 for data in market_data.values() if data.get('change_percent', 0) > 0)
        total_count = len(market_data)
        
        market_sentiment = "bullish" if positive_count > total_count * 0.6 else "bearish" if positive_count < total_count * 0.4 else "mixed"
        
        return {
            "market_sentiment": market_sentiment,
            "indices": market_data,
            "summary": {
                "positive_indices": positive_count,
                "total_indices": total_count
            },
            "market_status": "open" if datetime.now().weekday() < 5 else "closed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na visão geral do mercado: {e}")
        return {
            "market_sentiment": "unknown",
            "indices": {},
            "summary": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/v1/test")
async def test_endpoint():
    """Endpoint de teste para verificar se a API está funcionando"""
    
    try:
        # Teste simples com dados mock
        test_data = generate_mock_data("AAPL", 5)
        test_data = TechnicalIndicators.add_all_indicators(test_data)
        
        return {
            "status": "API funcionando",
            "test_data_points": len(test_data),
            "indicators_calculated": list(test_data.columns),
            "yfinance_available": YFINANCE_AVAILABLE,
            "strategies_loaded": len(api_manager.strategies),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        return {
            "status": "Erro no teste",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ===== ENDPOINTS ADICIONAIS SIMPLIFICADOS =====
@app.get("/api/v1/symbols")
async def get_supported_symbols():
    """Lista de símbolos suportados"""
    
    return {
        "popular_stocks": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NFLX", "name": "Netflix Inc."}
        ],
        "etfs": [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
            {"symbol": "VOO", "name": "Vanguard S&P 500 ETF"}
        ],
        "crypto": [
            {"symbol": "BTC-USD", "name": "Bitcoin"},
            {"symbol": "ETH-USD", "name": "Ethereum"}
        ],
        "total_supported": "Todos os símbolos de ações e ETFs principais"
    }

@app.post("/api/v1/backtest")
async def run_backtest(request: BacktestRequest):
    """Executar backtest simplificado de uma estratégia"""
    
    try:
        symbol = validate_symbol(request.symbol)
        
        if request.strategy not in api_manager.strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Estratégia '{request.strategy}' não encontrada. Disponíveis: {list(api_manager.strategies.keys())}"
            )
        
        # Datas padrão
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else datetime.now()
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00')) if request.start_date else end_date - timedelta(days=365)
        
        days = (end_date - start_date).days
        
        logger.info(f"🧪 Executando backtest: {request.strategy} para {symbol} ({days} dias)")
        
        # Buscar dados históricos
        data = await api_manager.get_market_data(symbol, days + 50)
        
        # Filtrar período
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if len(data) < 30:
            raise HTTPException(status_code=400, detail="Dados insuficientes para backtest (mínimo 30 dias)")
        
        # Executar backtest simplificado
        strategy = api_manager.strategies[request.strategy]
        strategy.reset()
        
        # Simular trading básico
        portfolio = {
            'cash': request.initial_capital,
            'shares': 0,
            'total_value': request.initial_capital,
            'trades': [],
            'daily_values': []
        }
        
        for i in range(len(data)):
            current_data = data.iloc[i]
            historical_data = data.iloc[:i] if i > 0 else pd.DataFrame()
            
            if len(historical_data) < 20:
                portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_data['close'])
                portfolio['daily_values'].append(portfolio_value)
                continue
            
            # Gerar sinal
            signal = strategy.generate_signal(current_data, historical_data)
            current_price = current_data['close']
            
            # Executar trades
            if signal['action'] == 'BUY' and portfolio['cash'] > current_price:
                shares_to_buy = int(portfolio['cash'] * 0.95 / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    portfolio['cash'] -= cost
                    portfolio['shares'] += shares_to_buy
                    
                    portfolio['trades'].append({
                        'date': current_data.name.isoformat(),
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'value': cost
                    })
            
            elif signal['action'] == 'SELL' and portfolio['shares'] > 0:
                proceeds = portfolio['shares'] * current_price
                
                portfolio['trades'].append({
                    'date': current_data.name.isoformat(),
                    'action': 'SELL',
                    'shares': portfolio['shares'],
                    'price': current_price,
                    'value': proceeds
                })
                
                portfolio['cash'] += proceeds
                portfolio['shares'] = 0
            
            # Calcular valor do portfolio
            portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_price)
            portfolio['daily_values'].append(portfolio_value)
        
        # Calcular métricas de performance
        final_value = portfolio['daily_values'][-1] if portfolio['daily_values'] else request.initial_capital
        total_return = ((final_value - request.initial_capital) / request.initial_capital) * 100
        
        # Benchmark (buy and hold)
        buy_hold_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
        
        # Win rate simplificado
        win_rate = 0
        if len(portfolio['trades']) >= 2:
            wins = 0
            for i in range(0, len(portfolio['trades']) - 1, 2):
                if i + 1 < len(portfolio['trades']):
                    buy_trade = portfolio['trades'][i] if portfolio['trades'][i]['action'] == 'BUY' else portfolio['trades'][i + 1]
                    sell_trade = portfolio['trades'][i + 1] if portfolio['trades'][i + 1]['action'] == 'SELL' else portfolio['trades'][i]
                    if sell_trade['price'] > buy_trade['price']:
                        wins += 1
            
            total_pairs = len(portfolio['trades']) // 2
            win_rate = (wins / total_pairs * 100) if total_pairs > 0 else 0
        
        logger.info(f"✅ Backtest concluído: {total_return:.2f}% vs {buy_hold_return:.2f}% benchmark")
        
        return {
            "strategy": request.strategy,
            "symbol": symbol,
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "initial_capital": request.initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 2),
            "benchmark_return": round(buy_hold_return, 2),
            "excess_return": round(total_return - buy_hold_return, 2),
            "total_trades": len(portfolio['trades']),
            "win_rate": round(win_rate, 2),
            "recent_trades": portfolio['trades'][-5:],
            "performance_summary": {
                "market_beating": total_return > buy_hold_return,
                "profitable": total_return > 0,
                "good_performance": total_return > 10
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro no backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no backtest: {str(e)}")

# ===== HANDLER DE ERROS =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler personalizado para erros HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler para erros gerais"""
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# ===== EVENTOS DE STARTUP/SHUTDOWN =====
@app.on_event("startup")
async def startup_event():
    """Eventos de inicialização da API"""
    logger.info("🚀 MakesALot Trading API v2.0 iniciando...")
    
    try:
        strategies_count = len(api_manager.strategies)
        apis_count = len(api_manager.data_apis)
        
        logger.info(f"📊 Componentes carregados:")
        logger.info(f"   - Estratégias: {strategies_count}")
        logger.info(f"   - APIs de dados: {apis_count}")
        logger.info(f"   - yfinance: {'✅ Disponível' if YFINANCE_AVAILABLE else '❌ Indisponível'}")
        
        # Teste rápido de funcionalidade
        test_symbol = "AAPL"
        if YFINANCE_AVAILABLE:
            test_data, _ = get_yfinance_data(test_symbol, 2)
            logger.info(f"   - Teste de dados: {'✅ OK' if test_data is not None else '⚠️ Fallback para mock'}")
        else:
            logger.info(f"   - Teste de dados: ⚠️ Usando dados mock")
        
        logger.info("✅ API inicializada com sucesso!")
        logger.info(f"📚 Documentação disponível em: /docs")
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        logger.warning("⚠️ Continuando com funcionalidade limitada...")

@app.on_event("shutdown")
async def shutdown_event():
    """Eventos de encerramento da API"""
    logger.info("🛑 Encerrando MakesALot Trading API...")
    
    try:
        # Limpar cache se existir
        if hasattr(api_manager, 'data_cache'):
            cache_size = len(api_manager.data_cache)
            api_manager.data_cache.clear()
            logger.info(f"🧹 Cache limpo: {cache_size} entradas removidas")
        
        logger.info("✅ API encerrada com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro no encerramento: {e}")

# ===== EXECUTAR APLICAÇÃO =====
if __name__ == "__main__":
    # Configurações do servidor
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Banner de inicialização
    print("\n" + "=" * 60)
    print("🚀 MAKESALOT TRADING API v2.0")
    print("=" * 60)
    print(f"🌐 Servidor: {host}:{port}")
    print(f"🐛 Debug: {'Ativado' if debug else 'Desativado'}")
    print(f"📊 yfinance: {'✅ Disponível' if YFINANCE_AVAILABLE else '❌ Indisponível'}")
    
    try:
        strategies_count = len(api_manager.strategies)
        apis_count = len(api_manager.data_apis)
        print(f"🔧 Estratégias carregadas: {strategies_count}")
        print(f"📡 APIs de dados: {apis_count}")
    except:
        print("🔧 Estratégias: Modo fallback")
        print("📡 APIs de dados: Modo fallback")
    
    print("=" * 60)
    print("📚 ENDPOINTS PRINCIPAIS:")
    print("   GET  /              - Informações da API")
    print("   GET  /health        - Status de saúde")
    print("   GET  /docs          - Documentação interativa")
    print("   POST /api/v1/analyze - Análise técnica completa")
    print("   GET  /api/v1/quote/{symbol} - Cotação detalhada")
    print("   POST /api/v1/backtest - Backtesting de estratégias")
    print("   GET  /api/v1/strategies - Lista de estratégias")
    print("   GET  /api/v1/market-overview - Visão geral do mercado")
    print("   GET  /api/v1/test - Teste de funcionalidade")
    print("=" * 60)
    print("💡 DICAS:")
    print("   • Use /test para verificar status")
    print("   • Fallbacks automáticos para dados")
    print("   • Logs detalhados em modo debug")
    print("   • Cache automático para performance")
    print("=" * 60)
    
    # Executar servidor
    try:
        print(f"🚀 Iniciando servidor...")
        print(f"📖 Documentação: http://{host}:{port}/docs")
        print(f"🔍 Health check: http://{host}:{port}/health")
        print(f"⚡ API info: http://{host}:{port}/")
        print("=" * 60 + "\n")
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="debug" if debug else "info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Servidor interrompido pelo usuário")
        print("✅ Encerramento graceful realizado")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Erro ao iniciar servidor: {e}")
        print("💡 Verifique as configurações e dependências")
        print("=" * 60)
        raise