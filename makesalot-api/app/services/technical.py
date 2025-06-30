"""
Technical analysis service for market analysis
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import talib
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class TechnicalAnalysisResult:
    trend: str
    signals: Dict[str, str]
    indicators: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    risk_reward: Optional[float] = None


class TechnicalAnalysisService:
    def __init__(self):
        self.TREND_WINDOW = 20
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30

    def analyze(
        self,
        df: pd.DataFrame,
        indicators: List[str] = ["RSI", "MACD", "BB"]
    ) -> TechnicalAnalysisResult:
        """
        Perform technical analysis on price data
        """
        try:
            # Calculate trend
            trend = self._calculate_trend(df)

            # Get indicator values
            indicator_values = {}
            signals = {}

            for indicator in indicators:
                if indicator == "RSI":
                    value = self._calculate_rsi(df)
                    signal = self._get_rsi_signal(value)
                elif indicator == "MACD":
                    value, signal = self._calculate_macd(df)
                elif indicator == "BB":
                    value = self._calculate_bollinger_bands(df)
                    signal = self._get_bb_signal(df, value)
                else:
                    continue

                indicator_values[indicator] = value
                signals[indicator] = signal

            # Calculate support/resistance
            support, resistance = self._find_support_resistance(df)

            # Calculate risk/reward
            risk_reward = self._calculate_risk_reward(
                df, support[-1] if support.any() else None,
                resistance[-1] if resistance.any() else None
            )

            return TechnicalAnalysisResult(
                trend=trend,
                signals=signals,
                indicators=indicator_values,
                support_levels=support.tolist(),
                resistance_levels=resistance.tolist(),
                risk_reward=risk_reward
            )

        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            raise

    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """Calculate overall trend using moving averages"""
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()

        current_ma20 = ma20.iloc[-1]
        current_ma50 = ma50.iloc[-1]

        if current_ma20 > current_ma50:
            return "bullish"
        elif current_ma20 < current_ma50:
            return "bearish"
        else:
            return "neutral"

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate RSI"""
        rsi = talib.RSI(df['close'].values)
        return rsi[-1]

    def _get_rsi_signal(self, rsi: float) -> str:
        """Get trading signal based on RSI"""
        if rsi > self.RSI_OVERBOUGHT:
            return "sell"
        elif rsi < self.RSI_OVERSOLD:
            return "buy"
        return "neutral"

    def _calculate_macd(self, df: pd.DataFrame) -> tuple:
        """Calculate MACD"""
        macd, signal, _ = talib.MACD(df['close'].values)
        return macd[-1], "buy" if macd[-1] > signal[-1] else "sell"

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(df['close'].values)
        return {
            "upper": upper[-1],
            "middle": middle[-1],
            "lower": lower[-1]
        }

    def _get_bb_signal(self, df: pd.DataFrame, bb: Dict) -> str:
        """Get trading signal based on Bollinger Bands"""
        current_price = df['close'].iloc[-1]
        if current_price > bb['upper']:
            return "sell"
        elif current_price < bb['lower']:
            return "buy"
        return "neutral"

    def _find_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        prominence: float = 1.0
    ) -> tuple:
        """Find support and resistance levels"""
        prices = df['close'].values

        # Find peaks (resistance) and troughs (support)
        peaks, _ = find_peaks(prices, prominence=prominence)
        troughs, _ = find_peaks(-prices, prominence=prominence)

        resistance_levels = prices[peaks]
        support_levels = prices[troughs]

        return support_levels, resistance_levels

    def _calculate_risk_reward(
        self,
        df: pd.DataFrame,
        support: Optional[float],
        resistance: Optional[float]
    ) -> Optional[float]:
        """Calculate risk/reward ratio"""
        if not support or not resistance:
            return None

        current_price = df['close'].iloc[-1]
        risk = abs(current_price - support)
        reward = abs(resistance - current_price)

        if risk == 0:
            return None

        return reward / risk
