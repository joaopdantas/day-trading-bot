"""
MakesALot Trading API - Versão Simples para Render
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ===== MODELOS DE DADOS =====
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1d"
    days: Optional[int] = 100

class AnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    trend: str
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    recommendation: str
    volume: int
    timestamp: str

class QuoteResponse(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[int] = None

# ===== CRIAR APP FASTAPI =====
app = FastAPI(
    title="MakesALot Trading API",
    description="API para análise técnica e previsões de trading",
    version="1.0.0"
)

# ===== CONFIGURAR CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifica o teu Chrome extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== FUNÇÕES DE ANÁLISE TÉCNICA =====
def calculate_rsi(prices, period=14):
    """Calcula o indicador RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 2) if not rsi.empty else None
    except:
        return None

def get_stock_data(symbol: str, days: int = 100):
    """Busca dados da ação usando yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None, None
            
        # Também buscar info da empresa
        info = ticker.info
        return data, info
        
    except Exception as e:
        print(f"Erro ao buscar dados para {symbol}: {e}")
        return None, None

def analyze_trend(sma_20, sma_50, current_price, prev_price):
    """Determina a tendência baseada em médias móveis e preço"""
    if sma_20 and sma_50 and prev_price:
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        if sma_20 > sma_50 and price_change > 0:
            return "bullish"
        elif sma_20 < sma_50 and price_change < 0:
            return "bearish"
        else:
            return "neutral"
    return "neutral"

def generate_recommendation(rsi, trend, change_percent):
    """Gera recomendação baseada em indicadores"""
    if not rsi:
        return "hold"
    
    # Recomendação baseada em RSI e tendência
    if rsi < 30 and trend != "bearish":
        return "buy"
    elif rsi > 70 and trend != "bullish":
        return "sell"
    elif trend == "bullish" and change_percent > 2:
        return "buy"
    elif trend == "bearish" and change_percent < -2:
        return "sell"
    else:
        return "hold"

# ===== ENDPOINTS DA API =====
@app.get("/")
def root():
    """Endpoint principal"""
    return {
        "message": "🚀 MakesALot Trading API está online!",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "quote": "/api/v1/quote/{symbol}"
        }
    }

@app.get("/health")
def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MakesALot Trading API"
    }

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def technical_analysis(request: AnalysisRequest):
    """
    Análise técnica completa de uma ação
    """
    try:
        # Buscar dados da ação
        data, info = get_stock_data(request.symbol, request.days)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Não foi possível encontrar dados para: {request.symbol}"
            )
        
        # Calcular indicadores
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        change_percent = round(((current_price - prev_price) / prev_price) * 100, 2)
        
        # Médias móveis
        sma_20 = round(float(data['Close'].rolling(20).mean().iloc[-1]), 2) if len(data) >= 20 else None
        sma_50 = round(float(data['Close'].rolling(50).mean().iloc[-1]), 2) if len(data) >= 50 else None
        
        # RSI
        rsi = calculate_rsi(data['Close'])
        
        # Análise de tendência
        trend = analyze_trend(sma_20, sma_50, current_price, prev_price)
        
        # Recomendação
        recommendation = generate_recommendation(rsi, trend, change_percent)
        
        # Volume
        volume = int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
        
        return AnalysisResponse(
            symbol=request.symbol.upper(),
            current_price=round(current_price, 2),
            change_percent=change_percent,
            trend=trend,
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
            recommendation=recommendation,
            volume=volume,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na análise de {request.symbol}: {str(e)}"
        )

@app.get("/api/v1/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str):
    """
    Cotação atual de uma ação
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="2d")  # Últimos 2 dias para calcular mudança
        
        if not info or hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Símbolo {symbol} não encontrado"
            )
        
        current_price = info.get("currentPrice") or float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = round(current_price - prev_close, 2)
        change_percent = round((change / prev_close) * 100, 2)
        
        return QuoteResponse(
            symbol=symbol.upper(),
            name=info.get("longName", "N/A"),
            price=round(current_price, 2),
            change=change,
            change_percent=change_percent,
            volume=info.get("volume", 0),
            market_cap=info.get("marketCap")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar cotação de {symbol}: {str(e)}"
        )

@app.get("/api/v1/symbols")
async def get_popular_symbols():
    """Lista de símbolos populares para trading"""
    return {
        "popular": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NFLX", "name": "Netflix Inc."}
        ],
        "crypto": [
            {"symbol": "BTC-USD", "name": "Bitcoin"},
            {"symbol": "ETH-USD", "name": "Ethereum"},
            {"symbol": "ADA-USD", "name": "Cardano"}
        ]
    }

# ===== EXECUTAR APLICAÇÃO =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Iniciando MakesALot Trading API em {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False  # False para produção
    )