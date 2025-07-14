from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware  # 👈 IMPORTANTE
from pydantic import BaseModel
from typing import Optional, Literal
from fetcher import get_data_api
from strategies import (
    TechnicalAnalysisStrategy,
    MLTradingStrategy,
    BuyAndHoldStrategy,
    RSIDivergenceStrategy,
    HybridRSIDivergenceStrategy,
)

import pandas as pd

app = FastAPI(title="MakesALot Trading API")

# 👇 Adicionar CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← Usa "*" só em desenvolvimento. Para produção: substitui pelo ID da tua extensão
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos para o POST
class SignalRequest(BaseModel):
    symbol: str
    interval: Optional[str] = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    api: Optional[Literal["alpha_vantage", "yahoo_finance", "polygon"]] = "alpha_vantage"
    strategy: Optional[Literal[
        "technical", "ml", "buy_and_hold", "rsi_divergence", "hybrid"
    ]] = "technical"

@app.get("/price/latest")
def get_latest_price(
    symbol: str,
    api: Literal["alpha_vantage", "yahoo_finance", "polygon"] = "alpha_vantage",
):
    client = get_data_api(api_name=api)
    return client.fetch_latest_price(symbol)

@app.get("/price/historical")
def get_historical_data(
    symbol: str,
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api: Literal["alpha_vantage", "yahoo_finance", "polygon"] = "alpha_vantage",
):
    client = get_data_api(api_name=api)
    df = client.fetch_historical_data(symbol, interval, start_date, end_date)
    return df.reset_index().to_dict(orient="records")

@app.post("/strategy/signal")
def generate_strategy_signal(request: SignalRequest):
    client = get_data_api(api_name=request.api)
    df = client.fetch_historical_data(
        symbol=request.symbol,
        interval=request.interval,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    if df.empty or len(df) < 5:
        return {"error": "Dados insuficientes para análise."}

    current = df.iloc[-1]
    historical = df.iloc[:-1]

    # Inicializar a estratégia
    if request.strategy == "ml":
        strategy = MLTradingStrategy()
    elif request.strategy == "buy_and_hold":
        strategy = BuyAndHoldStrategy()
    elif request.strategy == "rsi_divergence":
        strategy = RSIDivergenceStrategy()
    elif request.strategy == "hybrid":
        strategy = HybridRSIDivergenceStrategy()
    else:
        strategy = TechnicalAnalysisStrategy()

    signal = strategy.generate_signal(current, historical)
    return signal
