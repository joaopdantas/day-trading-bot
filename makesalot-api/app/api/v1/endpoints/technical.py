"""
Indicadores Técnicos Corrigidos para Pandas Moderno
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Classe para cálculo de indicadores técnicos compatível com pandas moderno"""
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Adicionar todos os indicadores técnicos ao DataFrame
        Versão corrigida para pandas moderno (sem method parameter)
        """
        
        if data.empty or len(data) < 2:
            logger.warning("Dados insuficientes para calcular indicadores")
            return data
        
        df = data.copy()
        
        try:
            # RSI
            df = TechnicalIndicators.add_rsi(df)
            
            # MACD
            df = TechnicalIndicators.add_macd(df)
            
            # Médias Móveis
            df = TechnicalIndicators.add_moving_averages(df)
            
            # Bollinger Bands
            df = TechnicalIndicators.add_bollinger_bands(df)
            
            # Preencher valores NaN usando métodos modernos do pandas
            # Forward fill seguido de backward fill
            df = df.ffill().bfill()
            
            # Se ainda houver NaN, preencher com valores padrão
            for col in df.columns:
                if df[col].isna().any():
                    if col == 'rsi':
                        df[col] = df[col].fillna(50.0)
                    elif col in ['macd', 'macd_signal', 'macd_histogram']:
                        df[col] = df[col].fillna(0.0)
                    elif col.startswith(('sma_', 'bb_')):
                        df[col] = df[col].fillna(df['close'])
                    else:
                        df[col] = df[col].fillna(0.0)
            
            logger.info("✅ Indicadores técnicos calculados com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular indicadores: {e}")
            # Fallback completo com valores seguros
            df = TechnicalIndicators._create_safe_fallback(df)
        
        return df
    
    @staticmethod
    def _create_safe_fallback(data: pd.DataFrame) -> pd.DataFrame:
        """Criar fallback seguro com valores padrão"""
        df = data.copy()
        
        # Garantir que temos pelo menos as colunas básicas
        if 'close' not in df.columns:
            df['close'] = 100.0
        
        # Adicionar indicadores com valores seguros
        df['rsi'] = 50.0
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_histogram'] = 0.0
        df['sma_20'] = df['close']
        df['sma_50'] = df['close']
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['bb_middle'] = df['close']
        
        logger.info("✅ Fallback seguro aplicado")
        return df
    
    @staticmethod
    def add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcular RSI com tratamento robusto de erros"""
        
        df = data.copy()
        
        if len(df) < period + 1:
            df['rsi'] = 50.0
            return df
        
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Usar rolling mean
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            # Evitar divisão por zero
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Preencher valores iniciais e NaN
            rsi = rsi.fillna(50.0)
            
            # Garantir que RSI está no range [0, 100]
            rsi = rsi.clip(0, 100)
            
            df['rsi'] = rsi
            
        except Exception as e:
            logger.warning(f"Erro no cálculo do RSI: {e}")
            df['rsi'] = 50.0
        
        return df
    
    @staticmethod
    def add_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calcular MACD com tratamento robusto"""
        
        df = data.copy()
        
        if len(df) < slow + signal:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
            return df
        
        try:
            # EMAs com min_periods para estabilidade
            ema_fast = df['close'].ewm(span=fast, min_periods=1).mean()
            ema_slow = df['close'].ewm(span=slow, min_periods=1).mean()
            
            # MACD Line
            macd_line = ema_fast - ema_slow
            
            # Signal Line
            signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            # Preencher e limpar
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
        """Calcular Bollinger Bands com tratamento robusto"""
        
        df = data.copy()
        
        if len(df) < period:
            df['bb_upper'] = df['close'] * 1.02
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close'] * 0.98
            return df
        
        try:
            # SMA com min_periods
            sma = df['close'].rolling(window=period, min_periods=1).mean()
            
            # Desvio padrão
            std = df['close'].rolling(window=period, min_periods=1).std()
            
            # Bandas com proteção contra std = 0
            std = std.fillna(df['close'] * 0.01)  # 1% como std mínimo
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Garantir que as bandas fazem sentido
            upper_band = upper_band.fillna(df['close'] * 1.02)
            lower_band = lower_band.fillna(df['close'] * 0.98)
            sma = sma.fillna(df['close'])
            
            df['bb_upper'] = upper_band
            df['bb_middle'] = sma
            df['bb_lower'] = lower_band
            
        except Exception as e:
            logger.warning(f"Erro no cálculo das Bollinger Bands: {e}")
            df['bb_upper'] = df['close'] * 1.02
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close'] * 0.98
        
        return df
    
    @staticmethod
    def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """Calcular médias móveis com tratamento robusto"""
        
        df = data.copy()
        
        try:
            # SMAs com min_periods para dados limitados
            df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # Preencher qualquer NaN restante
            for col in ['sma_10', 'sma_20', 'sma_50']:
                df[col] = df[col].fillna(df['close'])
                
        except Exception as e:
            logger.warning(f"Erro no cálculo das médias móveis: {e}")
            # Fallback seguro
            df['sma_10'] = df['close']
            df['sma_20'] = df['close']
            df['sma_50'] = df['close']
        
        return df
    
    @staticmethod
    def safe_fillna(series: pd.Series, fill_value=0.0) -> pd.Series:
        """Método seguro para preencher NaN compatível com todas as versões do pandas"""
        try:
            # Tentar método moderno primeiro
            return series.ffill().bfill().fillna(fill_value)
        except:
            try:
                # Fallback para versões mais antigas
                return series.fillna(method='ffill').fillna(method='bfill').fillna(fill_value)
            except:
                # Último recurso
                return series.fillna(fill_value)