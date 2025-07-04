"""
Enhanced ML Prediction Endpoints
Integrates with data fetcher, preprocessor and advanced ML models
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components
from .main import data_fetcher, data_preprocessor, storage

logger = logging.getLogger(__name__)

router = APIRouter()

# ===== ENHANCED PREDICTION MODELS =====
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., MSFT, AAPL)")
    timeframe: Optional[str] = Field("1d", description="Data timeframe")
    prediction_horizon: Optional[int] = Field(5, description="Days to predict ahead (1-30)")
    model_type: Optional[str] = Field("ensemble", description="Model: ml, technical, ensemble")
    confidence_level: Optional[float] = Field(0.95, description="Confidence level for intervals")
    include_scenarios: Optional[bool] = Field(True, description="Include bull/bear scenarios")

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predictions: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    scenarios: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    market_outlook: str
    data_quality: Dict[str, Any]
    timestamp: str

class MLPredictionEngine:
    """Advanced ML prediction engine with multiple models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set for ML models"""
        df_features = df.copy()
        
        # Add all technical indicators
        df_features = data_preprocessor.add_enhanced_indicators(df_features)
        
        # Price-based features
        df_features['price_change_1d'] = df_features['close'].pct_change(1)
        df_features['price_change_3d'] = df_features['close'].pct_change(3)
        df_features['price_change_5d'] = df_features['close'].pct_change(5)
        df_features['price_change_10d'] = df_features['close'].pct_change(10)
        
        # Volatility features
        df_features['volatility_5d'] = df_features['close'].rolling(5).std()
        df_features['volatility_10d'] = df_features['close'].rolling(10).std()
        df_features['volatility_20d'] = df_features['close'].rolling(20).std()
        
        # Volume features
        df_features['volume_change_1d'] = df_features['volume'].pct_change(1)
        df_features['volume_sma_ratio'] = df_features['volume'] / df_features['volume'].rolling(20).mean()
        
        # Price position features
        df_features['price_vs_sma20'] = df_features['close'] / df_features.get('SMA_20', df_features['close'])
        df_features['price_vs_sma50'] = df_features['close'] / df_features.get('SMA_50', df_features['close'])
        
        # High/Low features
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_vs_high'] = df_features['close'] / df_features['high']
        df_features['close_vs_low'] = df_features['close'] / df_features['low']
        
        # Time-based features
        df_features['day_of_week'] = df_features.index.dayofweek if hasattr(df_features.index, 'dayofweek') else 0
        df_features['month'] = df_features.index.month if hasattr(df_features.index, 'month') else 1
        df_features['quarter'] = df_features.index.quarter if hasattr(df_features.index, 'quarter') else 1
        
        # Momentum features
        df_features['momentum_3d'] = df_features['close'] / df_features['close'].shift(3)
        df_features['momentum_5d'] = df_features['close'] / df_features['close'].shift(5)
        df_features['momentum_10d'] = df_features['close'] / df_features['close'].shift(10)
        
        return df_features
    
    def create_ml_dataset(self, df: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create dataset for ML training with target variable"""
        df_features = self.prepare_features(df)
        
        # Select relevant features for ML
        feature_columns = [
            'RSI', 'MACD', 'BB_position', 'Stoch_K', 'Williams_R', 'ATR',
            'price_change_1d', 'price_change_3d', 'price_change_5d', 'price_change_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'volume_change_1d', 'volume_sma_ratio',
            'price_vs_sma20', 'price_vs_sma50',
            'high_low_ratio', 'close_vs_high', 'close_vs_low',
            'day_of_week', 'month', 'quarter',
            'momentum_3d', 'momentum_5d', 'momentum_10d'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df_features.columns]
        self.feature_names = available_features
        
        if not available_features:
            raise ValueError("No features available for ML model")
        
        # Create feature matrix
        X = df_features[available_features].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create target variable (future price change)
        y = df_features['close'].shift(-prediction_horizon) / df_features['close'] - 1
        
        # Remove rows with NaN targets
        valid_indices = ~y.isna()
        X = X[valid_indices].values
        y = y[valid_indices].values
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train multiple ML models and return performance metrics"""
        if len(X) < 50:
            raise ValueError("Insufficient data for ML training (minimum 50 samples required)")
        
        # Split data for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        performance = {}
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                
                # Calculate performance metrics
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                # Direction accuracy
                direction_accuracy = np.mean(np.sign(y_val) == np.sign(y_pred))
                
                performance[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'direction_accuracy': direction_accuracy,
                    'score': model.score(X_val_scaled, y_val)
                }
                
            except Exception as e:
                logger.warning(f"Model {name} training failed: {e}")
                performance[name] = {'error': str(e)}
        
        self.is_trained = True
        return performance
    
    def predict_ensemble(self, X: np.ndarray, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Make ensemble predictions with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        # Get predictions from each model
        model_predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = float(pred[0])
                model_predictions.append(pred[0])
            except Exception as e:
                logger.warning(f"Prediction failed for model {name}: {e}")
        
        if not model_predictions:
            raise ValueError("No models available for prediction")
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(model_predictions)
        pred_std = np.std(model_predictions)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        z_score = 1.96  # 95% confidence
        
        confidence_intervals = {
            'lower': ensemble_pred - z_score * pred_std,
            'upper': ensemble_pred + z_score * pred_std,
            'prediction': ensemble_pred,
            'std': pred_std
        }
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence_intervals': confidence_intervals,
            'prediction_std': pred_std
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        # Random Forest importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            for i, feature in enumerate(self.feature_names):
                importance_dict[f"rf_{feature}"] = float(rf_importance[i])
        
        # Gradient Boosting importance
        if 'gradient_boost' in self.models:
            gb_importance = self.models['gradient_boost'].feature_importances_
            for i, feature in enumerate(self.feature_names):
                importance_dict[f"gb_{feature}"] = float(gb_importance[i])
        
        # Calculate average importance
        avg_importance = {}
        for feature in self.feature_names:
            rf_imp = importance_dict.get(f"rf_{feature}", 0)
            gb_imp = importance_dict.get(f"gb_{feature}", 0)
            avg_importance[feature] = (rf_imp + gb_imp) / 2
        
        return avg_importance

class TechnicalPredictionEngine:
    """Technical analysis based prediction engine"""
    
    def predict_technical(self, df: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
        """Make predictions based on technical indicators"""
        df_tech = data_preprocessor.add_enhanced_indicators(df)
        current = df_tech.iloc[-1]
        current_price = current['close']
        
        # Technical signals
        signals = []
        signal_weights = []
        
        # RSI signal
        if 'RSI' in df_tech.columns and pd.notna(current['RSI']):
            rsi = current['RSI']
            if rsi < 30:
                signals.append(0.8)  # Strong buy
                signal_weights.append(0.25)
            elif rsi > 70:
                signals.append(-0.8)  # Strong sell
                signal_weights.append(0.25)
            else:
                signals.append((50 - rsi) / 50)  # Gradual signal
                signal_weights.append(0.15)
        
        # MACD signal
        if 'MACD' in df_tech.columns and pd.notna(current['MACD']):
            macd = current['MACD']
            macd_signal = current.get('MACD_signal', 0)
            if pd.notna(macd_signal):
                macd_diff = macd - macd_signal
                signals.append(np.tanh(macd_diff * 10))  # Normalized signal
                signal_weights.append(0.3)
        
        # Bollinger Bands signal
        if 'BB_position' in df_tech.columns and pd.notna(current['BB_position']):
            bb_pos = current['BB_position']
            if bb_pos < 0.2:
                signals.append(0.6)  # Buy signal
                signal_weights.append(0.2)
            elif bb_pos > 0.8:
                signals.append(-0.6)  # Sell signal
                signal_weights.append(0.2)
            else:
                signals.append((0.5 - bb_pos) * 2)  # Gradual signal
                signal_weights.append(0.1)
        
        # Moving average trend
        if 'SMA_20' in df_tech.columns and 'SMA_50' in df_tech.columns:
            sma20 = current.get('SMA_20')
            sma50 = current.get('SMA_50')
            if pd.notna(sma20) and pd.notna(sma50):
                ma_signal = (sma20 - sma50) / sma50
                signals.append(np.tanh(ma_signal * 5))
                signal_weights.append(0.25)
        
        # Calculate weighted prediction
        if signals and signal_weights:
            weighted_signal = np.average(signals, weights=signal_weights)
            
            # Convert signal to price change prediction
            # Scale by recent volatility
            recent_volatility = df_tech['close'].pct_change().tail(20).std()
            predicted_change = weighted_signal * recent_volatility * np.sqrt(horizon)
            
            return {
                'predicted_price_change': predicted_change,
                'predicted_price': current_price * (1 + predicted_change),
                'signal_strength': abs(weighted_signal),
                'direction': 'bullish' if weighted_signal > 0.1 else 'bearish' if weighted_signal < -0.1 else 'neutral',
                'confidence': min(abs(weighted_signal) * 100, 100)
            }
        
        return {
            'predicted_price_change': 0,
            'predicted_price': current_price,
            'signal_strength': 0,
            'direction': 'neutral',
            'confidence': 50
        }

class ScenarioAnalysis:
    """Generate bull/bear/base case scenarios"""
    
    def generate_scenarios(self, current_price: float, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate multiple market scenarios"""
        
        # Calculate historical statistics
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std()
        mean_return = returns.mean()
        
        # Scale to prediction horizon
        horizon_vol = daily_vol * np.sqrt(horizon)
        horizon_return = mean_return * horizon
        
        scenarios = {
            'base_case': {
                'price': current_price * (1 + horizon_return),
                'return': horizon_return,
                'probability': 0.6,
                'description': 'Most likely outcome based on historical trends'
            },
            'bull_case': {
                'price': current_price * (1 + horizon_return + 1.5 * horizon_vol),
                'return': horizon_return + 1.5 * horizon_vol,
                'probability': 0.2,
                'description': 'Optimistic scenario with strong positive momentum'
            },
            'bear_case': {
                'price': current_price * (1 + horizon_return - 1.5 * horizon_vol),
                'return': horizon_return - 1.5 * horizon_vol,
                'probability': 0.2,
                'description': 'Pessimistic scenario with significant downside'
            }
        }
        
        # Add extreme scenarios
        scenarios['extreme_bull'] = {
            'price': current_price * (1 + horizon_return + 2.5 * horizon_vol),
            'return': horizon_return + 2.5 * horizon_vol,
            'probability': 0.05,
            'description': 'Highly optimistic scenario (top 5% outcome)'
        }
        
        scenarios['extreme_bear'] = {
            'price': current_price * (1 + horizon_return - 2.5 * horizon_vol),
            'return': horizon_return - 2.5 * horizon_vol,
            'probability': 0.05,
            'description': 'Highly pessimistic scenario (bottom 5% outcome)'
        }
        
        return scenarios

# Initialize prediction engines
ml_engine = MLPredictionEngine()
technical_engine = TechnicalPredictionEngine()
scenario_engine = ScenarioAnalysis()

# ===== API ENDPOINTS =====

@router.post("/predict", response_model=PredictionResponse)
async def enhanced_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Enhanced ML-powered price prediction with multiple models and scenarios
    """
    try:
        # Validate parameters
        if request.prediction_horizon < 1 or request.prediction_horizon > 30:
            raise HTTPException(status_code=400, detail="Prediction horizon must be between 1 and 30 days")
        
        # Fetch data with extended history for ML training
        logger.info(f"Fetching data for prediction: {request.symbol}")
        df, source = await data_fetcher.fetch_with_fallback(
            request.symbol, 
            request.timeframe, 
            max(200, request.prediction_horizon * 20)  # Ensure sufficient training data
        )
        
        current_price = float(df['close'].iloc[-1])
        
        # Initialize prediction results
        predictions = {}
        model_performance = {}
        feature_importance = {}
        
        # ML Prediction
        if request.model_type in ['ml', 'ensemble']:
            try:
                X, y = ml_engine.create_ml_dataset(df, request.prediction_horizon)
                performance = ml_engine.train_models(X, y)
                
                # Make prediction on latest data
                latest_features = ml_engine.prepare_features(df).iloc[-1:].fillna(method='ffill').fillna(0)
                X_latest = latest_features[ml_engine.feature_names].values
                
                ml_prediction = ml_engine.predict_ensemble(X_latest, request.confidence_level)
                
                predictions['ml'] = {
                    'price_change': ml_prediction['ensemble_prediction'],
                    'predicted_price': current_price * (1 + ml_prediction['ensemble_prediction']),
                    'confidence_intervals': ml_prediction['confidence_intervals']
                }
                
                model_performance = performance
                feature_importance = ml_engine.get_feature_importance()
                
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                predictions['ml'] = {'error': str(e)}
        
        # Technical Analysis Prediction
        if request.model_type in ['technical', 'ensemble']:
            try:
                tech_prediction = technical_engine.predict_technical(df, request.prediction_horizon)
                predictions['technical'] = tech_prediction
                
            except Exception as e:
                logger.warning(f"Technical prediction failed: {e}")
                predictions['technical'] = {'error': str(e)}
        
        # Ensemble Prediction
        if request.model_type == 'ensemble' and 'ml' in predictions and 'technical' in predictions:
            if 'error' not in predictions['ml'] and 'error' not in predictions['technical']:
                ml_change = predictions['ml']['price_change']
                tech_change = predictions['technical']['predicted_price_change']
                
                # Weighted ensemble (ML 70%, Technical 30%)
                ensemble_change = 0.7 * ml_change + 0.3 * tech_change
                ensemble_price = current_price * (1 + ensemble_change)
                
                predictions['ensemble'] = {
                    'price_change': ensemble_change,
                    'predicted_price': ensemble_price,
                    'ml_weight': 0.7,
                    'technical_weight': 0.3
                }
        
        # Generate scenarios
        scenarios = {}
        if request.include_scenarios:
            scenarios = scenario_engine.generate_scenarios(current_price, df, request.prediction_horizon)
        
        # Risk Assessment
        risk_assessment = calculate_prediction_risk(df, request.prediction_horizon)
        
        # Market Outlook
        market_outlook = determine_market_outlook(predictions, scenarios)
        
        # Data Quality Assessment
        data_quality = {
            'data_points': len(df),
            'source': source,
            'completeness': 100 - (df.isnull().sum().sum() / df.size * 100),
            'recency': 'current',
            'training_samples': len(X) if 'X' in locals() else 0
        }
        
        # Confidence intervals for ensemble
        confidence_intervals = {}
        if 'ensemble' in predictions:
            # Combine ML and technical confidence
            if 'ml' in predictions and 'confidence_intervals' in predictions['ml']:
                ml_ci = predictions['ml']['confidence_intervals']
                confidence_intervals = {
                    'lower': current_price * (1 + ml_ci['lower']),
                    'upper': current_price * (1 + ml_ci['upper']),
                    'prediction': predictions['ensemble']['predicted_price']
                }
        elif 'ml' in predictions and 'confidence_intervals' in predictions['ml']:
            ml_ci = predictions['ml']['confidence_intervals']
            confidence_intervals = {
                'lower': current_price * (1 + ml_ci['lower']),
                'upper': current_price * (1 + ml_ci['upper']),
                'prediction': predictions['ml']['predicted_price']
            }
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_performance=model_performance,
            feature_importance=feature_importance,
            scenarios=scenarios,
            risk_assessment=risk_assessment,
            market_outlook=market_outlook,
            data_quality=data_quality,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def calculate_prediction_risk(df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """Calculate risk metrics for the prediction"""
    returns = df['close'].pct_change().dropna()
    
    # Volatility risk
    daily_vol = returns.std()
    horizon_vol = daily_vol * np.sqrt(horizon)
    
    # Downside risk
    downside_returns = returns[returns < 0]
    downside_risk = downside_returns.std() if len(downside_returns) > 0 else 0
    
    # Maximum drawdown risk
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # VaR (Value at Risk)
    var_95 = np.percentile(returns, 5)
    
    return {
        'volatility_risk': round(horizon_vol * 100, 2),
        'downside_risk': round(downside_risk * np.sqrt(horizon) * 100, 2),
        'max_historical_drawdown': round(max_drawdown * 100, 2),
        'value_at_risk_95': round(var_95 * 100, 2),
        'risk_level': 'high' if horizon_vol > 0.1 else 'medium' if horizon_vol > 0.05 else 'low'
    }

def determine_market_outlook(predictions: Dict[str, Any], scenarios: Dict[str, Any]) -> str:
    """Determine overall market outlook based on predictions"""
    positive_signals = 0
    total_signals = 0
    
    # Check ML prediction
    if 'ml' in predictions and 'error' not in predictions['ml']:
        if predictions['ml']['price_change'] > 0.02:
            positive_signals += 2
        elif predictions['ml']['price_change'] > 0:
            positive_signals += 1
        total_signals += 2
    
    # Check technical prediction
    if 'technical' in predictions and 'error' not in predictions['technical']:
        direction = predictions['technical'].get('direction', 'neutral')
        if direction == 'bullish':
            positive_signals += 1
        total_signals += 1
    
    # Check scenarios
    if scenarios:
        base_return = scenarios.get('base_case', {}).get('return', 0)
        if base_return > 0.05:
            positive_signals += 1
        total_signals += 1
    
    if total_signals == 0:
        return 'uncertain'
    
    positive_ratio = positive_signals / total_signals
    
    if positive_ratio >= 0.7:
        return 'bullish'
    elif positive_ratio <= 0.3:
        return 'bearish'
    else:
        return 'neutral'

@router.get("/models/performance/{symbol}")
async def get_model_performance(symbol: str, days: int = Query(200, description="Days of data for backtesting")):
    """
    Get historical performance metrics for prediction models
    """
    try:
        # Fetch data
        df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", days)
        
        # Backtest different horizons
        horizons = [1, 3, 5, 10]
        performance_results = {}
        
        for horizon in horizons:
            try:
                X, y = ml_engine.create_ml_dataset(df, horizon)
                if len(X) > 50:
                    performance = ml_engine.train_models(X, y)
                    performance_results[f'{horizon}d'] = performance
            except Exception as e:
                performance_results[f'{horizon}d'] = {'error': str(e)}
        
        return {
            'symbol': symbol,
            'data_points': len(df),
            'model_performance': performance_results,
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

@router.post("/batch-predict")
async def batch_prediction(symbols: List[str], horizon: int = 5, model_type: str = "ensemble"):
    """
    Run predictions for multiple symbols simultaneously
    """
    if len(symbols) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed per batch request")
    
    results = {}
    errors = {}
    
    for symbol in symbols:
        try:
            request = PredictionRequest(
                symbol=symbol,
                prediction_horizon=horizon,
                model_type=model_type,
                include_scenarios=False  # Simplified for batch
            )
            
            # Simplified prediction for batch processing
            df, source = await data_fetcher.fetch_with_fallback(symbol, "1d", 100)
            current_price = float(df['close'].iloc[-1])
            
            # Quick technical prediction
            tech_prediction = technical_engine.predict_technical(df, horizon)
            
            results[symbol] = {
                'current_price': current_price,
                'predicted_price': tech_prediction['predicted_price'],
                'price_change_percent': tech_prediction['predicted_price_change'] * 100,
                'direction': tech_prediction['direction'],
                'confidence': tech_prediction['confidence']
            }
            
        except Exception as e:
            errors[symbol] = str(e)
    
    return {
        'successful_predictions': len(results),
        'failed_predictions': len(errors),
        'results': results,
        'errors': errors,
        'timestamp': datetime.now().isoformat()
    }