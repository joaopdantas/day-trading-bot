"""
Módulo de Indicadores Técnicos para MakesALot Trading API

Este módulo fornece cálculos de indicadores técnicos robustos e otimizados
para análise de dados financeiros.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Classe principal para cálculo de indicadores técnicos
    
    Suporta todos os principais indicadores:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Médias Móveis (SMA, EMA)
    - Stochastic Oscillator
    - Williams %R
    - ATR (Average True Range)
    - Volume indicators
    """
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Adicionar todos os indicadores técnicos ao DataFrame
        
        Args:
            data: DataFrame com colunas OHLCV
            
        Returns:
            DataFrame com todos os indicadores adicionados
        """
        
        if data.empty or len(data) < 2:
            logger.warning("Dados insuficientes para calcular indicadores")
            return data
        
        df = data.copy()
        
        try:
            # Indicadores de momentum
            df = TechnicalIndicators.add_rsi(df)
            df = TechnicalIndicators.add_macd(df)
            df = TechnicalIndicators.add_stochastic(df)
            df = TechnicalIndicators.add_williams_r(df)
            
            # Indicadores de tendência
            df = TechnicalIndicators.add_moving_averages(df)
            df = TechnicalIndicators.add_bollinger_bands(df)
            df = TechnicalIndicators.add_atr(df)
            
            # Indicadores de volume
            df = TechnicalIndicators.add_volume_indicators(df)
            
            # Indicadores compostos
            df = TechnicalIndicators.add_bb_position(df)
            df = TechnicalIndicators.add_trend_strength(df)
            
            logger.info("✅ Todos os indicadores técnicos calculados com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular indicadores: {e}")
            # Retornar pelo menos os indicadores básicos
            df = TechnicalIndicators._add_basic_indicators_fallback(df)
        
        return df
    
    @staticmethod
    def add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcular RSI (Relative Strength Index)"""
        
        df = data.copy()
        
        if len(df) < period + 1:
            df['rsi'] = 50.0  # Valor neutro
            return df
        
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            df['rsi'] = rsi.fillna(50.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do RSI: {e}")
            df['rsi'] = 50.0
        
        return df
    
    @staticmethod
    def add_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calcular MACD (Moving Average Convergence Divergence)"""
        
        df = data.copy()
        
        if len(df) < slow + signal:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
            return df
        
        try:
            # Calcular EMAs
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            
            # MACD Line
            macd_line = ema_fast - ema_slow
            
            # Signal Line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            df['macd'] = macd_line.fillna(0.0)
            df['macd_signal'] = signal_line.fillna(0.0)
            df['macd_histogram'] = histogram.fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do MACD: {e}")
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
        
        return df
    
    @staticmethod
    def add_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calcular Bollinger Bands"""
        
        df = data.copy()
        
        if len(df) < period:
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
            return df
        
        try:
            # Média móvel simples
            sma = df['close'].rolling(window=period).mean()
            
            # Desvio padrão
            std = df['close'].rolling(window=period).std()
            
            # Bandas
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            df['bb_upper'] = upper_band.fillna(df['close'])
            df['bb_middle'] = sma.fillna(df['close'])
            df['bb_lower'] = lower_band.fillna(df['close'])
            
        except Exception as e:
            logger.warning(f"Erro no cálculo das Bollinger Bands: {e}")
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
        
        return df
    
    @staticmethod
    def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """Calcular médias móveis (SMA e EMA)"""
        
        df = data.copy()
        
        try:
            # SMAs
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # EMAs
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Preencher valores ausentes com o preço atual
            for col in ['sma_10', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']:
                df[col] = df[col].fillna(df['close'])
                
        except Exception as e:
            logger.warning(f"Erro no cálculo das médias móveis: {e}")
            # Fallback
            for col in ['sma_10', 'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']:
                df[col] = df['close']
        
        return df
    
    @staticmethod
    def add_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calcular Stochastic Oscillator"""
        
        df = data.copy()
        
        if len(df) < k_period:
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
            return df
        
        try:
            # Calcular %K
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            
            # Calcular %D (média móvel de %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            df['stoch_k'] = k_percent.fillna(50.0)
            df['stoch_d'] = d_percent.fillna(50.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do Stochastic: {e}")
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
        
        return df
    
    @staticmethod
    def add_williams_r(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcular Williams %R"""
        
        df = data.copy()
        
        if len(df) < period:
            df['williams_r'] = -50.0
            return df
        
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            df['williams_r'] = williams_r.fillna(-50.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do Williams %R: {e}")
            df['williams_r'] = -50.0
        
        return df
    
    @staticmethod
    def add_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcular ATR (Average True Range)"""
        
        df = data.copy()
        
        if len(df) < 2:
            df['atr'] = 0.0
            return df
        
        try:
            # True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # ATR (média móvel do True Range)
            atr = true_range.rolling(window=period).mean()
            
            df['atr'] = atr.fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do ATR: {e}")
            df['atr'] = 0.0
        
        return df
    
    @staticmethod
    def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores de volume"""
        
        df = data.copy()
        
        try:
            # Volume SMA
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Volume Ratio (current vs average)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # OBV (On-Balance Volume)
            obv = []
            obv_value = 0
            
            for i in range(len(df)):
                if i == 0:
                    obv_value = df['volume'].iloc[i]
                else:
                    if df['close'].iloc[i] > df['close'].iloc[i-1]:
                        obv_value += df['volume'].iloc[i]
                    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                        obv_value -= df['volume'].iloc[i]
                    # Se igual, OBV não muda
                
                obv.append(obv_value)
            
            df['obv'] = obv
            
            # Preencher NaN
            for col in ['volume_sma_10', 'volume_sma_20', 'volume_ratio']:
                if col in df.columns:
                    df[col] = df[col].fillna(1.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo dos indicadores de volume: {e}")
            df['volume_sma_10'] = df['volume']
            df['volume_sma_20'] = df['volume']
            df['volume_ratio'] = 1.0
            df['obv'] = df['volume'].cumsum()
        
        return df
    
    @staticmethod
    def add_bb_position(data: pd.DataFrame) -> pd.DataFrame:
        """Calcular posição dentro das Bollinger Bands"""
        
        df = data.copy()
        
        try:
            # Posição percentual dentro das bandas (0 = banda inferior, 1 = banda superior)
            bb_range = df['bb_upper'] - df['bb_lower']
            bb_position = (df['close'] - df['bb_lower']) / bb_range
            
            # Limitar entre 0 e 1
            bb_position = bb_position.clip(0, 1)
            
            df['bb_position'] = bb_position.fillna(0.5)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo da posição BB: {e}")
            df['bb_position'] = 0.5
        
        return df
    
    @staticmethod
    def add_trend_strength(data: pd.DataFrame) -> pd.DataFrame:
        """Calcular força da tendência"""
        
        df = data.copy()
        
        try:
            # ADX (Average Directional Index) simplificado
            if len(df) < 14:
                df['trend_strength'] = 0.5
                return df
            
            # Calcular DI+ e DI-
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Suavizar com média móvel
            period = 14
            plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
            minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
            tr_smooth = true_range.rolling(window=period).mean()
            
            # DI+ e DI-
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # ADX (média móvel do DX)
            adx = dx.rolling(window=period).mean()
            
            # Normalizar para 0-1
            trend_strength = adx / 100
            
            df['trend_strength'] = trend_strength.fillna(0.5)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo da força da tendência: {e}")
            df['trend_strength'] = 0.5
        
        return df
    
    @staticmethod
    def detect_rsi_divergences(data: pd.DataFrame, min_swing_pct: float = 2.5, max_lookback: int = 50) -> List[Dict]:
        """
        Detectar divergências RSI - Implementação da estratégia comprovada (64% retorno)
        
        Args:
            data: DataFrame com dados OHLC e RSI
            min_swing_pct: Percentual mínimo para considerar um swing
            max_lookback: Máximo de períodos para procurar divergências
            
        Returns:
            Lista de divergências detectadas
        """
        
        if len(data) < max_lookback or 'rsi' not in data.columns:
            return []
        
        divergences = []
        
        try:
            # Usar apenas os últimos dados para otimização
            recent_data = data.tail(max_lookback).copy()
            
            # Encontrar swings de preço (pivôs)
            price_swings = TechnicalIndicators._find_price_swings(recent_data, min_swing_pct)
            
            # Encontrar swings de RSI
            rsi_swings = TechnicalIndicators._find_rsi_swings(recent_data)
            
            # Detectar divergências
            for i in range(1, len(price_swings)):
                current_price_swing = price_swings[i]
                previous_price_swing = price_swings[i-1]
                
                # Procurar RSI correspondente
                current_rsi = TechnicalIndicators._find_nearest_rsi_swing(
                    current_price_swing, rsi_swings
                )
                previous_rsi = TechnicalIndicators._find_nearest_rsi_swing(
                    previous_price_swing, rsi_swings
                )
                
                if current_rsi and previous_rsi:
                    # Detectar divergência bullish
                    if (current_price_swing['type'] == 'low' and 
                        previous_price_swing['type'] == 'low' and
                        current_price_swing['price'] < previous_price_swing['price'] and
                        current_rsi['rsi'] > previous_rsi['rsi']):
                        
                        strength = abs(current_rsi['rsi'] - previous_rsi['rsi'])
                        
                        divergences.append({
                            'type': 'bullish',
                            'date': current_price_swing['date'],
                            'price': current_price_swing['price'],
                            'rsi': current_rsi['rsi'],
                            'strength': strength,
                            'description': f'Price lower low but RSI higher low'
                        })
                    
                    # Detectar divergência bearish
                    elif (current_price_swing['type'] == 'high' and 
                          previous_price_swing['type'] == 'high' and
                          current_price_swing['price'] > previous_price_swing['price'] and
                          current_rsi['rsi'] < previous_rsi['rsi']):
                        
                        strength = abs(previous_rsi['rsi'] - current_rsi['rsi'])
                        
                        divergences.append({
                            'type': 'bearish',
                            'date': current_price_swing['date'],
                            'price': current_price_swing['price'],
                            'rsi': current_rsi['rsi'],
                            'strength': strength,
                            'description': f'Price higher high but RSI lower high'
                        })
            
            # Filtrar divergências muito fracas
            strong_divergences = [d for d in divergences if d['strength'] >= 1.0]
            
            logger.info(f"Detectadas {len(strong_divergences)} divergências RSI")
            
            return strong_divergences
            
        except Exception as e:
            logger.error(f"Erro na detecção de divergências RSI: {e}")
            return []
    
    @staticmethod
    def _find_price_swings(data: pd.DataFrame, min_pct: float) -> List[Dict]:
        """Encontrar swings significativos no preço"""
        
        swings = []
        
        for i in range(2, len(data) - 2):
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            current_date = data.index[i]
            
            # Verificar se é um máximo local
            if (current_high > data['high'].iloc[i-1] and 
                current_high > data['high'].iloc[i-2] and
                current_high > data['high'].iloc[i+1] and 
                current_high > data['high'].iloc[i+2]):
                
                # Verificar se o swing é significativo
                recent_low = data['low'].iloc[max(0, i-10):i+1].min()
                swing_pct = ((current_high - recent_low) / recent_low) * 100
                
                if swing_pct >= min_pct:
                    swings.append({
                        'type': 'high',
                        'date': current_date,
                        'price': current_high,
                        'index': i
                    })
            
            # Verificar se é um mínimo local
            elif (current_low < data['low'].iloc[i-1] and 
                  current_low < data['low'].iloc[i-2] and
                  current_low < data['low'].iloc[i+1] and 
                  current_low < data['low'].iloc[i+2]):
                
                # Verificar se o swing é significativo
                recent_high = data['high'].iloc[max(0, i-10):i+1].max()
                swing_pct = ((recent_high - current_low) / current_low) * 100
                
                if swing_pct >= min_pct:
                    swings.append({
                        'type': 'low',
                        'date': current_date,
                        'price': current_low,
                        'index': i
                    })
        
        return swings
    
    @staticmethod
    def _find_rsi_swings(data: pd.DataFrame) -> List[Dict]:
        """Encontrar swings no RSI"""
        
        rsi_swings = []
        
        for i in range(2, len(data) - 2):
            current_rsi = data['rsi'].iloc[i]
            current_date = data.index[i]
            
            # Máximo local no RSI
            if (current_rsi > data['rsi'].iloc[i-1] and 
                current_rsi > data['rsi'].iloc[i-2] and
                current_rsi > data['rsi'].iloc[i+1] and 
                current_rsi > data['rsi'].iloc[i+2]):
                
                rsi_swings.append({
                    'type': 'high',
                    'date': current_date,
                    'rsi': current_rsi,
                    'index': i
                })
            
            # Mínimo local no RSI
            elif (current_rsi < data['rsi'].iloc[i-1] and 
                  current_rsi < data['rsi'].iloc[i-2] and
                  current_rsi < data['rsi'].iloc[i+1] and 
                  current_rsi < data['rsi'].iloc[i+2]):
                
                rsi_swings.append({
                    'type': 'low',
                    'date': current_date,
                    'rsi': current_rsi,
                    'index': i
                })
        
        return rsi_swings
    
    @staticmethod
    def _find_nearest_rsi_swing(price_swing: Dict, rsi_swings: List[Dict]) -> Optional[Dict]:
        """Encontrar o swing de RSI mais próximo no tempo"""
        
        if not rsi_swings:
            return None
        
        price_index = price_swing['index']
        price_type = price_swing['type']
        
        # Procurar RSI swing do mesmo tipo e próximo no tempo
        candidates = [
            rsi for rsi in rsi_swings 
            if rsi['type'] == price_type and abs(rsi['index'] - price_index) <= 5
        ]
        
        if not candidates:
            return None
        
        # Retornar o mais próximo
        nearest = min(candidates, key=lambda x: abs(x['index'] - price_index))
        return nearest
    
    @staticmethod
    def _add_basic_indicators_fallback(data: pd.DataFrame) -> pd.DataFrame:
        """Indicadores básicos como fallback em caso de erro"""
        
        df = data.copy()
        
        try:
            # RSI básico
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
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
            
            # SMAs básicas
            df['sma_20'] = df['close'].rolling(window=20).mean().fillna(df['close'])
            df['sma_50'] = df['close'].rolling(window=50).mean().fillna(df['close'])
            
            # Bollinger Bands básicas
            if len(df) >= 20:
                sma_20 = df['close'].rolling(window=20).mean()
                std_20 = df['close'].rolling(window=20).std()
                df['bb_upper'] = sma_20 + (std_20 * 2)
                df['bb_lower'] = sma_20 - (std_20 * 2)
                df['bb_middle'] = sma_20
            else:
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_middle'] = df['close']
            
            # Preencher NaN
            df = df.fillna(method='forward').fillna(method='backward')
            
            logger.info("✅ Indicadores básicos de fallback aplicados")
            
        except Exception as e:
            logger.error(f"❌ Erro nos indicadores de fallback: {e}")
            # Última tentativa - valores neutros
            df['rsi'] = 50.0
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
            df['sma_20'] = df['close']
            df['sma_50'] = df['close']
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_middle'] = df['close']
        
        return df
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, method: str = 'pivot') -> Dict:
        """
        Calcular níveis de suporte e resistência
        
        Args:
            data: DataFrame com dados OHLC
            method: Método de cálculo ('pivot', 'fibonacci', 'volume')
            
        Returns:
            Dicionário com níveis de suporte e resistência
        """
        
        if len(data) < 20:
            return {"support": [], "resistance": [], "method": method}
        
        current_price = data['close'].iloc[-1]
        
        if method == 'pivot':
            return TechnicalIndicators._calculate_pivot_levels(data)
        elif method == 'fibonacci':
            return TechnicalIndicators._calculate_fibonacci_levels(data)
        elif method == 'volume':
            return TechnicalIndicators._calculate_volume_levels(data)
        else:
            return TechnicalIndicators._calculate_pivot_levels(data)
    
    @staticmethod
    def _calculate_pivot_levels(data: pd.DataFrame) -> Dict:
        """Calcular níveis usando pontos de pivô"""
        
        # Usar últimos 50 períodos
        recent_data = data.tail(50)
        current_price = data['close'].iloc[-1]
        
        resistance_levels = []
        support_levels = []
        
        # Encontrar pivôs
        for i in range(2, len(recent_data) - 2):
            high = recent_data['high'].iloc[i]
            low = recent_data['low'].iloc[i]
            
            # Resistência (máximos locais)
            if (high > recent_data['high'].iloc[i-1] and 
                high > recent_data['high'].iloc[i-2] and
                high > recent_data['high'].iloc[i+1] and 
                high > recent_data['high'].iloc[i+2]):
                
                if high > current_price * 1.01:  # Acima do preço atual
                    resistance_levels.append(high)
            
            # Suporte (mínimos locais)
            if (low < recent_data['low'].iloc[i-1] and 
                low < recent_data['low'].iloc[i-2] and
                low < recent_data['low'].iloc[i+1] and 
                low < recent_data['low'].iloc[i+2]):
                
                if low < current_price * 0.99:  # Abaixo do preço atual
                    support_levels.append(low)
        
        # Agrupar níveis próximos
        resistance_levels = TechnicalIndicators._cluster_levels(resistance_levels)
        support_levels = TechnicalIndicators._cluster_levels(support_levels)
        
        return {
            "support": sorted(support_levels, reverse=True)[:3],
            "resistance": sorted(resistance_levels)[:3],
            "method": "pivot",
            "current_price": current_price
        }
    
    @staticmethod
    def _calculate_fibonacci_levels(data: pd.DataFrame) -> Dict:
        """Calcular níveis usando retração de Fibonacci"""
        
        # Encontrar swing high e low recentes
        recent_data = data.tail(100)
        
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        current_price = data['close'].iloc[-1]
        
        # Níveis de Fibonacci
        diff = swing_high - swing_low
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        resistance_levels = []
        support_levels = []
        
        for fib in fib_levels:
            level = swing_low + (diff * fib)
            
            if level > current_price * 1.01:
                resistance_levels.append(level)
            elif level < current_price * 0.99:
                support_levels.append(level)
        
        return {
            "support": sorted(support_levels, reverse=True)[:3],
            "resistance": sorted(resistance_levels)[:3],
            "method": "fibonacci",
            "swing_high": swing_high,
            "swing_low": swing_low,
            "current_price": current_price
        }
    
    @staticmethod
    def _calculate_volume_levels(data: pd.DataFrame) -> Dict:
        """Calcular níveis baseados em volume"""
        
        # Volume Profile simplificado
        recent_data = data.tail(100)
        current_price = data['close'].iloc[-1]
        
        # Dividir em bins de preço
        price_min = recent_data['low'].min()
        price_max = recent_data['high'].max()
        
        bins = np.linspace(price_min, price_max, 20)
        volume_profile = []
        
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            
            # Volume neste bin
            mask = (recent_data['low'] <= bin_high) & (recent_data['high'] >= bin_low)
            total_volume = recent_data.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'price': (bin_low + bin_high) / 2,
                'volume': total_volume
            })
        
        # Ordenar por volume
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        
        # Extrair níveis de alto volume
        high_volume_levels = [vp['price'] for vp in volume_profile[:5]]
        
        resistance_levels = [p for p in high_volume_levels if p > current_price * 1.01]
        support_levels = [p for p in high_volume_levels if p < current_price * 0.99]
        
        return {
            "support": sorted(support_levels, reverse=True)[:3],
            "resistance": sorted(resistance_levels)[:3],
            "method": "volume",
            "current_price": current_price
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.02) -> List[float]:
        """Agrupar níveis próximos"""
        
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                # Finalizar cluster atual
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Adicionar último cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    @staticmethod
    def get_market_regime(data: pd.DataFrame) -> Dict:
        """
        Determinar regime de mercado (trending vs ranging)
        
        Returns:
            Dicionário com informações do regime de mercado
        """
        
        if len(data) < 50:
            return {"regime": "unknown", "confidence": 0.5}
        
        try:
            # ADX para força da tendência
            data_with_trend = TechnicalIndicators.add_trend_strength(data)
            current_adx = data_with_trend['trend_strength'].iloc[-1]
            
            # Bollinger Band width
            bb_width = (data['bb_upper'].iloc[-1] - data['bb_lower'].iloc[-1]) / data['bb_middle'].iloc[-1]
            
            # Volatilidade
            returns = data['close'].pct_change().tail(20)
            volatility = returns.std()
            
            # Determinar regime
            if current_adx > 0.7 and bb_width > 0.1:
                regime = "strong_trending"
                confidence = min(current_adx + bb_width, 0.95)
            elif current_adx > 0.5 or bb_width > 0.08:
                regime = "weak_trending"
                confidence = (current_adx + bb_width) / 2
            else:
                regime = "ranging"
                confidence = 1 - current_adx
            
            return {
                "regime": regime,
                "confidence": round(confidence, 2),
                "adx": round(current_adx, 2),
                "bb_width": round(bb_width, 3),
                "volatility": round(volatility, 3),
                "recommendation": TechnicalIndicators._get_regime_recommendation(regime)
            }
            
        except Exception as e:
            logger.error(f"Erro na determinação do regime de mercado: {e}")
            return {"regime": "unknown", "confidence": 0.5}
    
    @staticmethod
    def _get_regime_recommendation(regime: str) -> str:
        """Obter recomendação baseada no regime de mercado"""
        
        recommendations = {
            "strong_trending": "Use trend-following strategies. RSI divergence works well.",
            "weak_trending": "Mixed signals. Use confluence of indicators.",
            "ranging": "Use mean-reversion strategies. Buy support, sell resistance.",
            "unknown": "Wait for clearer market direction."
        }
        
        return recommendations.get(regime, "Monitor market conditions.")