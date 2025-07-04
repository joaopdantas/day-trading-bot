"""
Enhanced Chart Data Endpoints
Provides professional-grade chart data with technical overlays and analytics
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Import our enhanced components
from .main import data_fetcher, data_preprocessor, storage

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== ENHANCED CHART MODELS =====
class ChartRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: Optional[str] = Field("3m", description="Time period: 1w, 1m, 3m, 6m, 1y, 2y, 5y")
    interval: Optional[str] = Field("1d", description="Data interval: 1m, 5m, 15m, 30m, 1h, 1d, 1w")
    indicators: Optional[List[str]] = Field([], description="Technical indicators to include")
    include_volume: Optional[bool] = Field(True, description="Include volume data")
    include_patterns: Optional[bool] = Field(False, description="Include pattern recognition")
    normalize: Optional[bool] = Field(False, description="Normalize prices to percentage change")

class ChartDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: Optional[Dict[str, float]] = {}
    patterns: Optional[Dict[str, Any]] = {}

class ChartResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    current_price: float
    price_change: float
    price_change_percent: float
    data_points: int
    data: List[ChartDataPoint]
    indicators_metadata: Dict[str, Any]
    volume_profile: Dict[str, Any]
    support_resistance: Dict[str, List[float]]
    market_hours: Dict[str, Any]
    data_quality: Dict[str, Any]
    timestamp: str

class EnhancedChartEngine:
    """Professional chart data processing engine"""
    
    def __init__(self):
        self.period_mapping = {
            '1w': 7, '1m': 30, '3m': 90, '6m': 180, 
            '1y': 365, '2y': 730, '5y': 1825
        }
        
        self.interval_mapping = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '1d': '1D', '1w': '1W'
        }
    
    def process_chart_data(self, df: pd.DataFrame, request: ChartRequest) -> Dict[str, Any]:
        """Process raw data into comprehensive chart data"""
        try:
            # Add technical indicators if requested
            if request.indicators:
                df = data_preprocessor.add_enhanced_indicators(df)
            
            # Calculate current metrics
            current_price = float(df['close'].iloc[-1])
            previous_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
            price_change = current_price - previous_price
            price_change_percent = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            # Create chart data points
            chart_data = []
            for idx, row in df.iterrows():
                # Basic OHLCV data
                data_point = {
                    'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
                    'indicators': {},
                    'patterns': {}
                }
                
                # Add requested indicators
                for indicator in request.indicators:
                    if indicator in df.columns and not pd.isna(row[indicator]):
                        data_point['indicators'][indicator] = float(row[indicator])
                
                # Add pattern data if requested
                if request.include_patterns:
                    pattern_data = self._extract_pattern_data(row, df.columns)
                    data_point['patterns'] = pattern_data
                
                chart_data.append(data_point)
            
            # Normalize data if requested
            if request.normalize:
                chart_data = self._normalize_chart_data(chart_data)
            
            # Calculate additional analytics
            indicators_metadata = self._calculate_indicators_metadata(df, request.indicators)
            volume_profile = self._calculate_volume_profile(df) if request.include_volume else {}
            support_resistance = self._calculate_support_resistance_levels(df)
            market_hours = self._get_market_hours_info(df)
            data_quality = self._assess_chart_data_quality(df)
            
            return {
                'symbol': request.symbol,
                'period': request.period,
                'interval': request.interval,
                'current_price': current_price,
                'price_change': round(price_change, 4),
                'price_change_percent': round(price_change_percent, 2),
                'data_points': len(chart_data),
                'data': chart_data,
                'indicators_metadata': indicators_metadata,
                'volume_profile': volume_profile,
                'support_resistance': support_resistance,
                'market_hours': market_hours,
                'data_quality': data_quality,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Chart data processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Chart processing failed: {str(e)}")
    
    def _extract_pattern_data(self, row: pd.Series, columns: List[str]) -> Dict[str, Any]:
        """Extract pattern recognition data from row"""
        patterns = {}
        
        # Candlestick patterns
        pattern_columns = ['doji', 'hammer']
        for pattern in pattern_columns:
            if pattern in columns and not pd.isna(row[pattern]):
                patterns[pattern] = bool(row[pattern])
        
        # Price action patterns
        if 'body_size' in columns and not pd.isna(row['body_size']):
            patterns['body_size'] = float(row['body_size'])
        
        if 'upper_shadow' in columns and not pd.isna(row['upper_shadow']):
            patterns['upper_shadow'] = float(row['upper_shadow'])
        
        if 'lower_shadow' in columns and not pd.isna(row['lower_shadow']):
            patterns['lower_shadow'] = float(row['lower_shadow'])
        
        return patterns
    
    def _normalize_chart_data(self, chart_data: List[Dict]) -> List[Dict]:
        """Normalize chart data to percentage changes from first data point"""
        if not chart_data:
            return chart_data
        
        base_price = chart_data[0]['close']
        
        for point in chart_data:
            for price_field in ['open', 'high', 'low', 'close']:
                if price_field in point:
                    point[price_field] = ((point[price_field] - base_price) / base_price) * 100
        
        return chart_data
    
    def _calculate_indicators_metadata(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Calculate metadata for technical indicators"""
        metadata = {}
        
        for indicator in indicators:
            if indicator not in df.columns:
                continue
            
            values = df[indicator].dropna()
            if len(values) == 0:
                continue
            
            # Basic statistics
            current_value = float(values.iloc[-1]) if len(values) > 0 else None
            min_value = float(values.min())
            max_value = float(values.max())
            mean_value = float(values.mean())
            std_value = float(values.std())
            
            # Signal interpretation
            signal = self._interpret_indicator_signal(indicator, current_value, mean_value, std_value)
            
            metadata[indicator] = {
                'current': current_value,
                'min': min_value,
                'max': max_value,
                'mean': mean_value,
                'std': std_value,
                'signal': signal,
                'interpretation': self._get_indicator_interpretation(indicator, current_value)
            }
        
        return metadata
    
    def _interpret_indicator_signal(self, indicator: str, current: float, mean: float, std: float) -> str:
        """Interpret technical indicator signals"""
        if indicator == 'RSI':
            if current < 30:
                return 'oversold'
            elif current > 70:
                return 'overbought'
            else:
                return 'neutral'
        
        elif indicator == 'MACD':
            return 'bullish' if current > 0 else 'bearish'
        
        elif indicator == 'BB_position':
            if current < 0.2:
                return 'near_lower_band'
            elif current > 0.8:
                return 'near_upper_band'
            else:
                return 'middle_range'
        
        elif indicator in ['Stoch_K', 'Stoch_D']:
            if current < 20:
                return 'oversold'
            elif current > 80:
                return 'overbought'
            else:
                return 'neutral'
        
        elif indicator == 'Williams_R':
            if current < -80:
                return 'oversold'
            elif current > -20:
                return 'overbought'
            else:
                return 'neutral'
        
        else:
            # Generic signal based on standard deviations from mean
            z_score = (current - mean) / std if std > 0 else 0
            if z_score > 1.5:
                return 'high'
            elif z_score < -1.5:
                return 'low'
            else:
                return 'normal'
    
    def _get_indicator_interpretation(self, indicator: str, value: float) -> str:
        """Get human-readable interpretation of indicator values"""
        interpretations = {
            'RSI': {
                'description': 'Relative Strength Index measures momentum',
                'current_meaning': f'Current RSI of {value:.1f} indicates ' + 
                    ('oversold conditions' if value < 30 else 'overbought conditions' if value > 70 else 'neutral momentum')
            },
            'MACD': {
                'description': 'Moving Average Convergence Divergence shows trend changes',
                'current_meaning': f'MACD of {value:.3f} suggests ' + 
                    ('bullish momentum' if value > 0 else 'bearish momentum')
            },
            'BB_position': {
                'description': 'Position within Bollinger Bands (0=lower band, 1=upper band)',
                'current_meaning': f'Price at {value:.1%} of band width indicates ' + 
                    ('potential support' if value < 0.2 else 'potential resistance' if value > 0.8 else 'normal range')
            },
            'Stoch_K': {
                'description': 'Stochastic %K oscillator measures price momentum',
                'current_meaning': f'Stochastic of {value:.1f} shows ' + 
                    ('oversold conditions' if value < 20 else 'overbought conditions' if value > 80 else 'neutral momentum')
            }
        }
        
        return interpretations.get(indicator, {
            'description': f'{indicator} technical indicator',
            'current_meaning': f'Current value: {value:.3f}'
        })
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile analysis"""
        if 'volume' not in df.columns or len(df) < 10:
            return {'status': 'insufficient_data'}
        
        try:
            # Current vs average volume
            current_volume = df['volume'].iloc[-1]
            avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Volume trend
            recent_volumes = df['volume'].tail(5)
            volume_trend = 'increasing' if recent_volumes.iloc[-1] > recent_volumes.mean() * 1.2 else \
                          'decreasing' if recent_volumes.iloc[-1] < recent_volumes.mean() * 0.8 else 'stable'
            
            # Price-Volume correlation
            price_changes = df['close'].pct_change().dropna()
            volume_changes = df['volume'].pct_change().dropna()
            
            if len(price_changes) > 10 and len(volume_changes) > 10:
                # Align lengths
                min_length = min(len(price_changes), len(volume_changes))
                correlation = price_changes.tail(min_length).corr(volume_changes.tail(min_length))
            else:
                correlation = 0
            
            # Volume by price levels (simplified VPOC)
            price_volume_data = []
            price_range = df['high'].max() - df['low'].min()
            num_levels = min(20, len(df))
            
            for i in range(num_levels):
                level_min = df['low'].min() + (i * price_range / num_levels)
                level_max = df['low'].min() + ((i + 1) * price_range / num_levels)
                
                # Find candles that traded in this price level
                level_volume = 0
                for _, row in df.iterrows():
                    if row['low'] <= level_max and row['high'] >= level_min:
                        # Weight by how much of the candle was in this level
                        overlap = min(row['high'], level_max) - max(row['low'], level_min)
                        candle_range = row['high'] - row['low']
                        weight = overlap / candle_range if candle_range > 0 else 0
                        level_volume += row['volume'] * weight
                
                price_volume_data.append({
                    'price_level': (level_min + level_max) / 2,
                    'volume': int(level_volume)
                })
            
            # Find VPOC (Volume Point of Control)
            vpoc_data = max(price_volume_data, key=lambda x: x['volume'])
            
            return {
                'current_volume': int(current_volume),
                'average_volume_20d': int(avg_volume_20),
                'volume_ratio': round(volume_ratio, 2),
                'volume_trend': volume_trend,
                'price_volume_correlation': round(correlation, 3),
                'vpoc_price': round(vpoc_data['price_level'], 2),
                'vpoc_volume': vpoc_data['volume'],
                'volume_distribution': price_volume_data,
                'interpretation': self._interpret_volume_profile(volume_ratio, volume_trend, correlation)
            }
            
        except Exception as e:
            logger.warning(f"Volume profile calculation error: {e}")
            return {'status': 'calculation_error', 'error': str(e)}
    
    def _interpret_volume_profile(self, ratio: float, trend: str, correlation: float) -> str:
        """Interpret volume profile data"""
        if ratio > 2 and trend == 'increasing' and correlation > 0.3:
            return 'Strong bullish volume confirmation with high participation'
        elif ratio > 2 and trend == 'increasing' and correlation < -0.3:
            return 'High volume with bearish price action - potential distribution'
        elif ratio > 1.5 and trend == 'increasing':
            return 'Above average volume suggests increased interest'
        elif ratio < 0.5:
            return 'Below average volume indicates low participation'
        elif trend == 'decreasing':
            return 'Decreasing volume may signal waning momentum'
        else:
            return 'Normal volume patterns with neutral implications'
    
    def _calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate dynamic support and resistance levels"""
        try:
            support_levels, resistance_levels = data_preprocessor.calculate_support_resistance(df)
            
            # Add moving average levels as dynamic S/R
            ma_levels = []
            for period in [20, 50, 100, 200]:
                ma_col = f'SMA_{period}'
                if ma_col in df.columns:
                    current_ma = df[ma_col].iloc[-1]
                    if pd.notna(current_ma):
                        ma_levels.append(float(current_ma))
            
            # Add Bollinger Bands as S/R
            bb_levels = []
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                bb_upper = df['BB_upper'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                if pd.notna(bb_upper) and pd.notna(bb_lower):
                    bb_levels.extend([float(bb_upper), float(bb_lower)])
            
            # Combine and sort levels
            all_support = sorted(support_levels + [level for level in ma_levels + bb_levels 
                                                 if level < df['close'].iloc[-1]])
            all_resistance = sorted([level for level in resistance_levels + ma_levels + bb_levels 
                                   if level > df['close'].iloc[-1]])
            
            return {
                'support': all_support[-3:] if len(all_support) >= 3 else all_support,  # Top 3 closest
                'resistance': all_resistance[:3] if len(all_resistance) >= 3 else all_resistance,  # Top 3 closest
                'pivot_points': self._calculate_pivot_points(df),
                'psychological_levels': self._find_psychological_levels(df['close'].iloc[-1])
            }
            
        except Exception as e:
            logger.warning(f"Support/resistance calculation error: {e}")
            return {'support': [], 'resistance': [], 'pivot_points': {}, 'psychological_levels': []}
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate traditional pivot points"""
        if len(df) < 1:
            return {}
        
        # Use previous day's data for pivot calculation
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Standard pivot point formula
        pivot = (high + low + close) / 3
        
        # Support and resistance levels
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2),
            's1': round(s1, 2),
            's2': round(s2, 2),
            's3': round(s3, 2)
        }
    
    def _find_psychological_levels(self, current_price: float) -> List[float]:
        """Find psychological price levels (round numbers)"""
        levels = []
        
        # Find the appropriate scale
        if current_price > 1000:
            # For high-priced stocks, use $50 and $100 levels
            base = int(current_price / 100) * 100
            levels = [base - 100, base, base + 100]
        elif current_price > 100:
            # For medium-priced stocks, use $10 and $25 levels
            base = int(current_price / 25) * 25
            levels = [base - 25, base, base + 25]
        elif current_price > 10:
            # For lower-priced stocks, use $5 and $10 levels
            base = int(current_price / 10) * 10
            levels = [base - 10, base, base + 10]
        else:
            # For very low-priced stocks, use $1 levels
            base = int(current_price)
            levels = [base - 1, base, base + 1]
        
        # Filter out negative levels and levels too far from current price
        valid_levels = [level for level in levels 
                       if level > 0 and abs(level - current_price) / current_price < 0.5]
        
        return valid_levels
    
    def _get_market_hours_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get market hours and trading session information"""
        try:
            if len(df) == 0:
                return {'status': 'no_data'}
            
            # Detect if this is intraday data
            if hasattr(df.index, 'hour'):
                # Check for market hours patterns
                hours = df.index.hour
                trading_hours = hours[(hours >= 9) & (hours <= 16)]
                is_intraday = len(trading_hours) > 0
                
                if is_intraday:
                    # Analyze volume by time of day
                    hourly_volume = df.groupby(df.index.hour)['volume'].mean()
                    peak_hour = hourly_volume.idxmax() if len(hourly_volume) > 0 else 10
                    
                    return {
                        'is_intraday': True,
                        'trading_hours': '09:30-16:00 EST',
                        'peak_volume_hour': int(peak_hour),
                        'market_open_volume': float(hourly_volume.get(9, 0)),
                        'market_close_volume': float(hourly_volume.get(15, 0)),
                        'session_analysis': self._analyze_trading_sessions(df)
                    }
            
            # For daily data or non-intraday
            return {
                'is_intraday': False,
                'data_frequency': 'daily',
                'market_hours': 'standard',
                'trading_days': len(df)
            }
            
        except Exception as e:
            logger.warning(f"Market hours analysis error: {e}")
            return {'status': 'analysis_error', 'error': str(e)}
    
    def _analyze_trading_sessions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different trading sessions (pre-market, regular, after-hours)"""
        try:
            if not hasattr(df.index, 'hour'):
                return {'status': 'not_intraday'}
            
            # Define sessions
            premarket = df[(df.index.hour >= 4) & (df.index.hour < 9)]
            regular = df[(df.index.hour >= 9) & (df.index.hour < 16)]
            afterhours = df[(df.index.hour >= 16) & (df.index.hour < 20)]
            
            sessions = {}
            
            for session_name, session_data in [('premarket', premarket), 
                                             ('regular', regular), 
                                             ('afterhours', afterhours)]:
                if len(session_data) > 0:
                    avg_volume = session_data['volume'].mean()
                    price_range = (session_data['high'].max() - session_data['low'].min()) / session_data['close'].mean()
                    
                    sessions[session_name] = {
                        'average_volume': int(avg_volume),
                        'price_range_percent': round(price_range * 100, 2),
                        'data_points': len(session_data)
                    }
            
            return sessions
            
        except Exception as e:
            logger.warning(f"Trading sessions analysis error: {e}")
            return {'status': 'analysis_error'}
    
    def _assess_chart_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of chart data"""
        quality_score = 100
        issues = []
        
        # Check for missing data
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            quality_score -= min(missing_data * 2, 20)
            issues.append(f"{missing_data} missing data points")
        
        # Check for data gaps
        if hasattr(df.index, 'date'):
            date_diffs = df.index.to_series().diff()
            expected_diff = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else pd.Timedelta(days=1)
            large_gaps = (date_diffs > expected_diff * 3).sum()
            
            if large_gaps > 0:
                quality_score -= min(large_gaps * 5, 15)
                issues.append(f"{large_gaps} significant time gaps")
        
        # Check for data recency
        if len(df) > 0:
            try:
                last_date = df.index[-1]
                if hasattr(last_date, 'date'):
                    days_old = (datetime.now().date() - last_date.date()).days
                    if days_old > 1:
                        quality_score -= min(days_old * 2, 10)
                        issues.append(f"Data is {days_old} days old")
            except:
                pass
        
        # Check for sufficient data length
        if len(df) < 30:
            quality_score -= (30 - len(df))
            issues.append(f"Limited data history ({len(df)} periods)")
        
        # Check for price anomalies
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # >50% changes
            
            if extreme_changes > 0:
                quality_score -= min(extreme_changes * 5, 10)
                issues.append(f"{extreme_changes} extreme price movements detected")
        
        return {
            'quality_score': max(quality_score, 0),
            'data_points': len(df),
            'issues': issues,
            'recommendation': 'excellent' if quality_score > 90 else 
                           'good' if quality_score > 75 else 
                           'fair' if quality_score > 60 else 'poor',
            'completeness_percent': round((1 - missing_data / df.size) * 100, 1) if df.size > 0 else 0
        }

# Initialize chart engine
chart_engine = EnhancedChartEngine()

# ===== API ENDPOINTS =====

@router.get("/data/{symbol}", response_model=ChartResponse)
async def enhanced_chart_data(
    symbol: str,
    period: str = Query("3m", description="Time period: 1w, 1m, 3m, 6m, 1y, 2y, 5y"),
    interval: str = Query("1d", description="Data interval: 1m, 5m, 15m, 30m, 1h, 1d, 1w"),
    indicators: List[str] = Query([], description="Technical indicators to include"),
    include_volume: bool = Query(True, description="Include volume analysis"),
    include_patterns: bool = Query(False, description="Include pattern recognition"),
    normalize: bool = Query(False, description="Normalize to percentage changes")
):
    """
    Get comprehensive chart data with technical analysis and market insights
    """
    try:
        # Validate parameters
        if period not in chart_engine.period_mapping:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
        
        # Calculate days needed
        days = chart_engine.period_mapping[period]
        
        # Fetch data
        logger.info(f"Fetching chart data for {symbol}, period: {period}, interval: {interval}")
        df, source = await data_fetcher.fetch_with_fallback(symbol, interval, days)
        
        # Create request object
        request = ChartRequest(
            symbol=symbol,
            period=period,
            interval=interval,
            indicators=indicators,
            include_volume=include_volume,
            include_patterns=include_patterns,
            normalize=normalize
        )
        
        # Process chart data
        chart_data = chart_engine.process_chart_data(df, request)
        
        # Add data source info
        chart_data['data_quality']['source'] = source
        
        return ChartResponse(**chart_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Chart data retrieval failed: {str(e)}")

@router.get("/indicators/{symbol}")
async def get_chart_indicators(symbol: str, period: str = "3m"):
    """
    Get available technical indicators with current values and interpretations
    """
    try:
        # Fetch data
        days = chart_engine.period_mapping.get(period, 90)
        df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", days)
        
        # Add all indicators
        df = data_preprocessor.add_enhanced_indicators(df)
        
        # Available indicators
        available_indicators = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_position',
            'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'OBV',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_12', 'EMA_26'
        ]
        
        # Filter to available columns
        existing_indicators = [ind for ind in available_indicators if ind in df.columns]
        
        # Calculate metadata
        indicators_info = chart_engine._calculate_indicators_metadata(df, existing_indicators)
        
        return {
            'symbol': symbol,
            'period': period,
            'available_indicators': existing_indicators,
            'indicators_data': indicators_info,
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Indicators data error: {e}")
        raise HTTPException(status_code=500, detail=f"Indicators retrieval failed: {str(e)}")

@router.get("/volume-profile/{symbol}")
async def get_volume_profile(symbol: str, period: str = "1m", interval: str = "1h"):
    """
    Get detailed volume profile analysis for intraday trading
    """
    try:
        # Fetch intraday data
        days = chart_engine.period_mapping.get(period, 30)
        df, source = await data_fetcher.fetch_with_fallback(symbol, interval, days)
        
        # Calculate comprehensive volume profile
        volume_profile = chart_engine._calculate_volume_profile(df)
        
        # Additional volume metrics
        volume_metrics = {
            'total_volume': int(df['volume'].sum()),
            'average_daily_volume': int(df['volume'].mean()),
            'volume_weighted_average_price': float((df['close'] * df['volume']).sum() / df['volume'].sum()),
            'high_volume_threshold': float(df['volume'].quantile(0.8)),
            'low_volume_threshold': float(df['volume'].quantile(0.2))
        }
        
        return {
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'volume_profile': volume_profile,
            'volume_metrics': volume_metrics,
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Volume profile error: {e}")
        raise HTTPException(status_code=500, detail=f"Volume profile analysis failed: {str(e)}")

@router.get("/support-resistance/{symbol}")
async def get_support_resistance(symbol: str, period: str = "6m"):
    """
    Get comprehensive support and resistance analysis
    """
    try:
        # Fetch data
        days = chart_engine.period_mapping.get(period, 180)
        df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", days)
        
        # Calculate support/resistance
        support_resistance = chart_engine._calculate_support_resistance_levels(df)
        
        # Additional analysis
        current_price = df['close'].iloc[-1]
        
        # Distance to nearest levels
        nearest_support = max([level for level in support_resistance['support'] 
                             if level < current_price], default=None)
        nearest_resistance = min([level for level in support_resistance['resistance'] 
                                if level > current_price], default=None)
        
        analysis = {
            'current_price': float(current_price),
            'nearest_support': float(nearest_support) if nearest_support else None,
            'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
            'support_distance_percent': round((current_price - nearest_support) / current_price * 100, 2) if nearest_support else None,
            'resistance_distance_percent': round((nearest_resistance - current_price) / current_price * 100, 2) if nearest_resistance else None
        }
        
        return {
            'symbol': symbol,
            'period': period,
            'support_resistance': support_resistance,
            'analysis': analysis,
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Support/resistance analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Support/resistance analysis failed: {str(e)}")

@router.get("/compare")
async def compare_symbols(
    symbols: List[str] = Query(..., description="List of symbols to compare"),
    period: str = Query("3m", description="Time period for comparison"),
    normalize: bool = Query(True, description="Normalize prices for comparison")
):
    """
    Compare multiple symbols on the same chart
    """
    if len(symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed for comparison")
    
    try:
        comparison_data = {}
        
        for symbol in symbols:
            try:
                # Fetch data for each symbol
                days = chart_engine.period_mapping.get(period, 90)
                df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", days)
                
                # Basic processing
                current_price = float(df['close'].iloc[-1])
                start_price = float(df['close'].iloc[0])
                total_return = (current_price - start_price) / start_price * 100
                
                # Create simplified chart data
                chart_points = []
                for idx, row in df.iterrows():
                    price = float(row['close'])
                    if normalize:
                        price = (price - start_price) / start_price * 100
                    
                    chart_points.append({
                        'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        'price': price,
                        'volume': int(row['volume'])
                    })
                
                comparison_data[symbol] = {
                    'current_price': current_price,
                    'total_return_percent': round(total_return, 2),
                    'data_points': len(chart_points),
                    'chart_data': chart_points,
                    'data_source': source
                }
                
            except Exception as e:
                comparison_data[symbol] = {'error': str(e)}
        
        return {
            'symbols': symbols,
            'period': period,
            'normalized': normalize,
            'comparison_data': comparison_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Symbol comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Symbol comparison failed: {str(e)}")