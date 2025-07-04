"""
Enhanced Technical Analysis Endpoints
Integrates with data fetcher, preprocessor and storage for professional analysis
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Import our enhanced components (simulated - would be actual imports in production)
from .main import data_fetcher, data_preprocessor, storage

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== ENHANCED REQUEST MODELS =====
class TechnicalAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., MSFT, AAPL)")
    timeframe: Optional[str] = Field("1d", description="Data timeframe: 1m, 5m, 15m, 30m, 1h, 1d")
    days: Optional[int] = Field(100, description="Number of days of historical data")
    indicators: Optional[List[str]] = Field(
        ["RSI", "MACD", "BB", "Stochastic", "Williams_R"], 
        description="Technical indicators to calculate"
    )
    include_patterns: Optional[bool] = Field(True, description="Include candlestick pattern analysis")
    include_volume_analysis: Optional[bool] = Field(True, description="Include volume analysis")

class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    trend: str
    confidence_score: float
    signals: Dict[str, str]
    indicators: List[Dict[str, Any]]
    support_levels: List[float]
    resistance_levels: List[float]
    volume_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    market_regime: str
    data_quality: Dict[str, Any]
    timestamp: str

# ===== ENHANCED ANALYSIS FUNCTIONS =====
class AdvancedTechnicalAnalyzer:
    """Professional-grade technical analysis engine"""
    
    def __init__(self):
        self.signal_weights = {
            'RSI': 0.20,
            'MACD': 0.25,
            'BB': 0.15,
            'Stochastic': 0.15,
            'Williams_R': 0.10,
            'Volume': 0.15
        }
    
    def analyze_symbol(self, df: pd.DataFrame, symbol: str, request: TechnicalAnalysisRequest) -> Dict[str, Any]:
        """Comprehensive technical analysis of a symbol"""
        try:
            # Add all technical indicators
            df_analyzed = data_preprocessor.add_enhanced_indicators(df)
            
            # Current market data
            current_data = df_analyzed.iloc[-1]
            previous_data = df_analyzed.iloc[-2] if len(df_analyzed) > 1 else current_data
            
            # Calculate basic metrics
            current_price = float(current_data['close'])
            change_percent = ((current_price - previous_data['close']) / previous_data['close']) * 100
            
            # Generate signals for each indicator
            signals = self._generate_signals(df_analyzed, request.indicators)
            
            # Calculate overall trend and confidence
            trend, confidence = self._calculate_trend_confidence(df_analyzed, signals)
            
            # Support and resistance levels
            support_levels, resistance_levels = data_preprocessor.calculate_support_resistance(df_analyzed)
            
            # Volume analysis
            volume_analysis = self._analyze_volume(df_analyzed) if request.include_volume_analysis else {}
            
            # Pattern analysis
            pattern_analysis = self._analyze_patterns(df_analyzed) if request.include_patterns else {}
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(df_analyzed)
            
            # Market regime detection
            market_regime = self._detect_market_regime(df_analyzed)
            
            # Data quality assessment
            data_quality = self._assess_data_quality(df_analyzed)
            
            # Format indicators for response
            indicators_formatted = self._format_indicators(df_analyzed, request.indicators)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'change_percent': round(change_percent, 2),
                'trend': trend,
                'confidence_score': round(confidence, 2),
                'signals': signals,
                'indicators': indicators_formatted,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'volume_analysis': volume_analysis,
                'pattern_analysis': pattern_analysis,
                'risk_metrics': risk_metrics,
                'market_regime': market_regime,
                'data_quality': data_quality,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _generate_signals(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, str]:
        """Generate trading signals for each indicator"""
        signals = {}
        current = df.iloc[-1]
        
        if 'RSI' in indicators and 'RSI' in df.columns:
            rsi = current['RSI']
            if pd.notna(rsi):
                if rsi < 30:
                    signals['RSI'] = 'BUY'
                elif rsi > 70:
                    signals['RSI'] = 'SELL'
                else:
                    signals['RSI'] = 'HOLD'
            else:
                signals['RSI'] = 'HOLD'
        
        if 'MACD' in indicators and 'MACD' in df.columns:
            macd = current['MACD']
            macd_signal = current.get('MACD_signal', 0)
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal and macd > 0:
                    signals['MACD'] = 'BUY'
                elif macd < macd_signal and macd < 0:
                    signals['MACD'] = 'SELL'
                else:
                    signals['MACD'] = 'HOLD'
            else:
                signals['MACD'] = 'HOLD'
        
        if 'BB' in indicators and 'BB_position' in df.columns:
            bb_pos = current['BB_position']
            if pd.notna(bb_pos):
                if bb_pos < 0.2:
                    signals['BB'] = 'BUY'
                elif bb_pos > 0.8:
                    signals['BB'] = 'SELL'
                else:
                    signals['BB'] = 'HOLD'
            else:
                signals['BB'] = 'HOLD'
        
        if 'Stochastic' in indicators and 'Stoch_K' in df.columns:
            stoch_k = current['Stoch_K']
            if pd.notna(stoch_k):
                if stoch_k < 20:
                    signals['Stochastic'] = 'BUY'
                elif stoch_k > 80:
                    signals['Stochastic'] = 'SELL'
                else:
                    signals['Stochastic'] = 'HOLD'
            else:
                signals['Stochastic'] = 'HOLD'
        
        if 'Williams_R' in indicators and 'Williams_R' in df.columns:
            williams = current['Williams_R']
            if pd.notna(williams):
                if williams < -80:
                    signals['Williams_R'] = 'BUY'
                elif williams > -20:
                    signals['Williams_R'] = 'SELL'
                else:
                    signals['Williams_R'] = 'HOLD'
            else:
                signals['Williams_R'] = 'HOLD'
        
        return signals
    
    def _calculate_trend_confidence(self, df: pd.DataFrame, signals: Dict[str, str]) -> tuple:
        """Calculate overall trend and confidence score"""
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        weighted_score = 0
        
        for indicator, signal in signals.items():
            weight = self.signal_weights.get(indicator, 0.1)
            signal_counts[signal] += 1
            total_weight += weight
            
            if signal == 'BUY':
                weighted_score += weight
            elif signal == 'SELL':
                weighted_score -= weight
        
        # Price trend analysis
        if len(df) >= 20:
            recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            if recent_trend > 0.05:
                weighted_score += 0.2
            elif recent_trend < -0.05:
                weighted_score -= 0.2
        
        # Determine overall trend
        if weighted_score > 0.3:
            trend = 'BULLISH'
        elif weighted_score < -0.3:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Calculate confidence (0-100)
        confidence = min(abs(weighted_score) * 100, 100)
        
        return trend, confidence
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive volume analysis"""
        if 'volume' not in df.columns or len(df) < 20:
            return {'status': 'insufficient_data'}
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        volume_trend = 'increasing' if volume_ratio > 1.5 else 'decreasing' if volume_ratio < 0.5 else 'normal'
        
        # Price-Volume correlation
        if len(df) >= 10:
            price_changes = df['close'].pct_change().dropna()
            volume_changes = df['volume'].pct_change().dropna()
            correlation = price_changes.corr(volume_changes) if len(price_changes) > 5 else 0
        else:
            correlation = 0
        
        # On-Balance Volume analysis
        obv_signal = 'neutral'
        if 'OBV' in df.columns and len(df) >= 10:
            obv_trend = (df['OBV'].iloc[-1] - df['OBV'].iloc[-10]) / abs(df['OBV'].iloc[-10]) if df['OBV'].iloc[-10] != 0 else 0
            obv_signal = 'bullish' if obv_trend > 0.1 else 'bearish' if obv_trend < -0.1 else 'neutral'
        
        return {
            'current_volume': int(current_volume),
            'average_volume': int(avg_volume),
            'volume_ratio': round(volume_ratio, 2),
            'volume_trend': volume_trend,
            'price_volume_correlation': round(correlation, 3),
            'obv_signal': obv_signal,
            'interpretation': self._interpret_volume_analysis(volume_trend, correlation, obv_signal)
        }
    
    def _interpret_volume_analysis(self, volume_trend: str, correlation: float, obv_signal: str) -> str:
        """Interpret volume analysis results"""
        if volume_trend == 'increasing' and correlation > 0.3 and obv_signal == 'bullish':
            return 'Strong bullish volume confirmation'
        elif volume_trend == 'increasing' and correlation < -0.3 and obv_signal == 'bearish':
            return 'Strong bearish volume confirmation'
        elif volume_trend == 'decreasing':
            return 'Low volume suggests weak conviction'
        else:
            return 'Mixed volume signals require caution'
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Candlestick and chart pattern analysis"""
        if len(df) < 10:
            return {'status': 'insufficient_data'}
        
        patterns_found = []
        current = df.iloc[-1]
        
        # Doji pattern
        if 'doji' in df.columns and current['doji'] == 1:
            patterns_found.append({
                'name': 'Doji',
                'type': 'reversal',
                'significance': 'medium',
                'description': 'Indecision in market, potential reversal'
            })
        
        # Hammer pattern
        if 'hammer' in df.columns and current['hammer'] == 1:
            patterns_found.append({
                'name': 'Hammer',
                'type': 'bullish_reversal',
                'significance': 'high',
                'description': 'Potential bullish reversal signal'
            })
        
        # Moving average crossovers
        ma_signals = self._detect_ma_crossovers(df)
        patterns_found.extend(ma_signals)
        
        # Price channel analysis
        channel_analysis = self._analyze_price_channels(df)
        
        return {
            'patterns_detected': len(patterns_found),
            'patterns': patterns_found,
            'channel_analysis': channel_analysis,
            'overall_pattern_signal': self._summarize_pattern_signals(patterns_found)
        }
    
    def _detect_ma_crossovers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect moving average crossovers"""
        crossovers = []
        
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns and len(df) >= 2:
            current_20 = df['SMA_20'].iloc[-1]
            current_50 = df['SMA_50'].iloc[-1]
            prev_20 = df['SMA_20'].iloc[-2]
            prev_50 = df['SMA_50'].iloc[-2]
            
            if pd.notna(current_20) and pd.notna(current_50) and pd.notna(prev_20) and pd.notna(prev_50):
                # Golden cross
                if prev_20 <= prev_50 and current_20 > current_50:
                    crossovers.append({
                        'name': 'Golden Cross',
                        'type': 'bullish',
                        'significance': 'high',
                        'description': 'SMA 20 crossed above SMA 50 - bullish signal'
                    })
                # Death cross
                elif prev_20 >= prev_50 and current_20 < current_50:
                    crossovers.append({
                        'name': 'Death Cross',
                        'type': 'bearish',
                        'significance': 'high',
                        'description': 'SMA 20 crossed below SMA 50 - bearish signal'
                    })
        
        return crossovers
    
    def _analyze_price_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price channels and breakouts"""
        if len(df) < 20:
            return {'status': 'insufficient_data'}
        
        # Calculate recent high and low
        lookback = min(20, len(df))
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]
        
        # Channel position
        channel_range = recent_high - recent_low
        position_in_channel = (current_price - recent_low) / channel_range if channel_range > 0 else 0.5
        
        # Breakout detection
        breakout_signal = 'none'
        if current_price > recent_high * 1.001:  # 0.1% buffer
            breakout_signal = 'bullish_breakout'
        elif current_price < recent_low * 0.999:  # 0.1% buffer
            breakout_signal = 'bearish_breakout'
        
        return {
            'channel_high': round(recent_high, 2),
            'channel_low': round(recent_low, 2),
            'channel_width': round(channel_range, 2),
            'position_in_channel': round(position_in_channel, 3),
            'breakout_signal': breakout_signal,
            'channel_interpretation': self._interpret_channel_position(position_in_channel, breakout_signal)
        }
    
    def _interpret_channel_position(self, position: float, breakout: str) -> str:
        """Interpret channel position"""
        if breakout == 'bullish_breakout':
            return 'Price broke above resistance - bullish momentum'
        elif breakout == 'bearish_breakout':
            return 'Price broke below support - bearish momentum'
        elif position > 0.8:
            return 'Price near resistance - potential selling pressure'
        elif position < 0.2:
            return 'Price near support - potential buying opportunity'
        else:
            return 'Price in middle of channel - no clear directional bias'
    
    def _summarize_pattern_signals(self, patterns: List[Dict[str, Any]]) -> str:
        """Summarize overall pattern signals"""
        if not patterns:
            return 'neutral'
        
        bullish_count = sum(1 for p in patterns if p.get('type') in ['bullish', 'bullish_reversal'])
        bearish_count = sum(1 for p in patterns if p.get('type') in ['bearish', 'bearish_reversal'])
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        if len(df) < 20:
            return {'status': 'insufficient_data'}
        
        returns = df['close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Average True Range as % of price
        if 'ATR' in df.columns:
            atr_percent = (df['ATR'].iloc[-1] / df['close'].iloc[-1]) * 100
        else:
            atr_percent = volatility / np.sqrt(252)  # Daily volatility approximation
        
        # Beta calculation (vs SPY - simplified)
        beta = 1.0  # Would need market data for real calculation
        
        return {
            'volatility_annualized': round(volatility, 2),
            'value_at_risk_95': round(var_95, 2),
            'max_drawdown': round(max_drawdown, 2),
            'atr_percent': round(atr_percent, 2),
            'beta_estimate': beta,
            'risk_level': self._categorize_risk_level(volatility, max_drawdown)
        }
    
    def _categorize_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """Categorize overall risk level"""
        if volatility > 50 or max_drawdown < -30:
            return 'high'
        elif volatility > 25 or max_drawdown < -15:
            return 'medium'
        else:
            return 'low'
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < 50:
            return 'unknown'
        
        # Trend analysis
        short_ma = df['close'].rolling(10).mean().iloc[-1]
        long_ma = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Volatility analysis
        returns = df['close'].pct_change().dropna()
        recent_vol = returns.tail(20).std()
        historical_vol = returns.std()
        
        # Regime classification
        if current_price > short_ma > long_ma and recent_vol < historical_vol * 1.2:
            return 'bull_market'
        elif current_price < short_ma < long_ma and recent_vol < historical_vol * 1.2:
            return 'bear_market'
        elif recent_vol > historical_vol * 1.5:
            return 'high_volatility'
        else:
            return 'consolidation'
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of the data used for analysis"""
        quality_score = 100
        issues = []
        
        # Check data completeness
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            quality_score -= min(missing_data * 2, 20)
            issues.append(f"Missing {missing_data} data points")
        
        # Check data recency
        if len(df) > 0:
            last_date = df.index[-1] if hasattr(df.index, 'date') else datetime.now()
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            days_old = (datetime.now() - last_date).days if hasattr(last_date, 'date') else 0
            
            if days_old > 1:
                quality_score -= min(days_old * 5, 25)
                issues.append(f"Data is {days_old} days old")
        
        # Check sufficient data length
        if len(df) < 50:
            quality_score -= (50 - len(df))
            issues.append(f"Limited data history ({len(df)} days)")
        
        return {
            'quality_score': max(quality_score, 0),
            'data_points': len(df),
            'issues': issues,
            'recommendation': 'reliable' if quality_score > 80 else 'caution' if quality_score > 60 else 'unreliable'
        }
    
    def _format_indicators(self, df: pd.DataFrame, requested_indicators: List[str]) -> List[Dict[str, Any]]:
        """Format indicators for API response"""
        indicators = []
        current = df.iloc[-1]
        
        indicator_configs = {
            'RSI': {
                'column': 'RSI',
                'interpretation': lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral'
            },
            'MACD': {
                'column': 'MACD',
                'interpretation': lambda x: 'Bullish' if x > 0 else 'Bearish'
            },
            'BB': {
                'column': 'BB_position',
                'interpretation': lambda x: 'Near Lower Band' if x < 0.2 else 'Near Upper Band' if x > 0.8 else 'Middle Range'
            },
            'Stochastic': {
                'column': 'Stoch_K',
                'interpretation': lambda x: 'Oversold' if x < 20 else 'Overbought' if x > 80 else 'Neutral'
            },
            'Williams_R': {
                'column': 'Williams_R',
                'interpretation': lambda x: 'Oversold' if x < -80 else 'Overbought' if x > -20 else 'Neutral'
            }
        }
        
        for indicator in requested_indicators:
            if indicator in indicator_configs:
                config = indicator_configs[indicator]
                column = config['column']
                
                if column in df.columns:
                    value = current[column]
                    if pd.notna(value):
                        indicators.append({
                            'name': indicator,
                            'value': round(float(value), 4),
                            'interpretation': config['interpretation'](value),
                            'signal': self._get_indicator_signal(indicator, value)
                        })
        
        return indicators
    
    def _get_indicator_signal(self, indicator: str, value: float) -> str:
        """Get signal for individual indicator"""
        if indicator == 'RSI':
            return 'BUY' if value < 30 else 'SELL' if value > 70 else 'HOLD'
        elif indicator == 'MACD':
            return 'BUY' if value > 0 else 'SELL'
        elif indicator == 'BB':
            return 'BUY' if value < 0.2 else 'SELL' if value > 0.8 else 'HOLD'
        elif indicator == 'Stochastic':
            return 'BUY' if value < 20 else 'SELL' if value > 80 else 'HOLD'
        elif indicator == 'Williams_R':
            return 'BUY' if value < -80 else 'SELL' if value > -20 else 'HOLD'
        else:
            return 'HOLD'

# Initialize analyzer
analyzer = AdvancedTechnicalAnalyzer()

# ===== API ENDPOINTS =====

@router.post("/analyze", response_model=TechnicalAnalysisResponse)
async def enhanced_technical_analysis(request: TechnicalAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Enhanced technical analysis with comprehensive indicators and risk metrics
    """
    try:
        # Check cache first
        cached_data = storage.get_cached_data(request.symbol, request.timeframe)
        
        if cached_data is None:
            # Fetch fresh data with fallback
            logger.info(f"Fetching data for {request.symbol}")
            df, source = await data_fetcher.fetch_with_fallback(
                request.symbol, 
                request.timeframe, 
                request.days
            )
            
            # Cache the data
            storage.cache_data(request.symbol, request.timeframe, (df, source))
            
            logger.info(f"Data fetched from {source} for {request.symbol}")
        else:
            df, source = cached_data
            logger.info(f"Using cached data for {request.symbol}")
        
        # Perform comprehensive analysis
        analysis_result = analyzer.analyze_symbol(df, request.symbol, request)
        
        # Add data source info
        analysis_result['data_quality']['source'] = source
        
        return TechnicalAnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in technical analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Technical analysis failed for {request.symbol}: {str(e)}"
        )

@router.get("/indicators/{symbol}")
async def get_available_indicators(symbol: str):
    """
    Get list of available technical indicators for a symbol
    """
    available_indicators = [
        {
            'name': 'RSI',
            'description': 'Relative Strength Index - Momentum oscillator',
            'range': '0-100',
            'signals': 'Oversold < 30, Overbought > 70'
        },
        {
            'name': 'MACD',
            'description': 'Moving Average Convergence Divergence',
            'range': 'Unbounded',
            'signals': 'Crossovers and divergences'
        },
        {
            'name': 'BB',
            'description': 'Bollinger Bands position',
            'range': '0-1',
            'signals': 'Near bands indicate potential reversals'
        },
        {
            'name': 'Stochastic',
            'description': 'Stochastic Oscillator',
            'range': '0-100',
            'signals': 'Oversold < 20, Overbought > 80'
        },
        {
            'name': 'Williams_R',
            'description': 'Williams %R momentum indicator',
            'range': '-100 to 0',
            'signals': 'Oversold < -80, Overbought > -20'
        },
        {
            'name': 'ATR',
            'description': 'Average True Range - Volatility measure',
            'range': 'Positive values',
            'signals': 'Higher values indicate higher volatility'
        },
        {
            'name': 'OBV',
            'description': 'On-Balance Volume',
            'range': 'Cumulative',
            'signals': 'Trend confirmation through volume'
        }
    ]
    
    return {
        'symbol': symbol,
        'available_indicators': available_indicators,
        'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '1d'],
        'max_historical_days': 365
    }

@router.get("/market-regime/{symbol}")
async def get_market_regime(symbol: str, days: int = Query(100, description="Days of data for analysis")):
    """
    Analyze current market regime for a symbol
    """
    try:
        # Fetch data
        df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", days)
        df_analyzed = data_preprocessor.add_enhanced_indicators(df)
        
        # Detect regime
        regime = analyzer._detect_market_regime(df_analyzed)
        risk_metrics = analyzer._calculate_risk_metrics(df_analyzed)
        
        # Additional regime analysis
        volatility_regime = 'high' if risk_metrics.get('volatility_annualized', 0) > 30 else 'normal'
        trend_strength = abs(df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        return {
            'symbol': symbol,
            'market_regime': regime,
            'volatility_regime': volatility_regime,
            'trend_strength_percent': round(trend_strength, 2),
            'risk_metrics': risk_metrics,
            'analysis_period_days': len(df),
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market regime analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Market regime analysis failed: {str(e)}")

@router.post("/bulk-analyze")
async def bulk_technical_analysis(symbols: List[str], timeframe: str = "1d", days: int = 50):
    """
    Perform technical analysis on multiple symbols simultaneously
    """
    if len(symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed per request")
    
    results = {}
    errors = {}
    
    for symbol in symbols:
        try:
            request = TechnicalAnalysisRequest(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                indicators=["RSI", "MACD", "BB"]
            )
            
            # Simplified analysis for bulk requests
            df, source = await data_fetcher.fetch_with_fallback(symbol, timeframe, days)
            analysis_result = analyzer.analyze_symbol(df, symbol, request)
            
            # Simplified response for bulk
            results[symbol] = {
                'current_price': analysis_result['current_price'],
                'change_percent': analysis_result['change_percent'],
                'trend': analysis_result['trend'],
                'confidence_score': analysis_result['confidence_score'],
                'signals': analysis_result['signals'],
                'risk_level': analysis_result['risk_metrics'].get('risk_level', 'unknown')
            }
            
        except Exception as e:
            errors[symbol] = str(e)
    
    return {
        'successful_analyses': len(results),
        'failed_analyses': len(errors),
        'results': results,
        'errors': errors,
        'timestamp': datetime.now().isoformat()
    }