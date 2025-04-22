"""
Pipeline Completa de Análise Técnica para Ponto de Controle

Este script implementa uma pipeline completa que integra:
1. Detecção de níveis de suporte e resistência
2. Reconhecimento de padrões de candlestick
3. Identificação de tendências
4. Geração de sinais combinados
5. Preparação de datasets para modelos ML
6. Engenharia de features para inputs ML
7. Treinamento e avaliação de modelos preditivos iniciais
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Criar pasta para salvar resultados
os.makedirs('outputs', exist_ok=True)

# Importar suas classes existentes (presumidas como disponíveis)
from PatternRecognition import PatternRecognition
from CandlestickPatterns import CandlestickPatterns


class TrendIdentifier:
    """Classe para identificação de tendências no mercado"""
    
    @staticmethod
    def identify_trends(df: pd.DataFrame, 
                      short_window: int = 20, 
                      long_window: int = 50,
                      price_col: str = 'close') -> pd.DataFrame:
        """
        Identifica tendências usando médias móveis e vários indicadores
        
        Args:
            df: DataFrame com dados OHLC
            short_window: Período da média móvel curta
            long_window: Período da média móvel longa
            price_col: Coluna de preço a ser usada
            
        Returns:
            DataFrame com colunas de tendência adicionadas
        """
        result_df = df.copy()
        
        # Calcular médias móveis
        result_df['sma_short'] = result_df[price_col].rolling(window=short_window).mean()
        result_df['sma_long'] = result_df[price_col].rolling(window=long_window).mean()
        
        # Identificar tendência com base no cruzamento de médias móveis
        result_df['trend_sma'] = np.where(result_df['sma_short'] > result_df['sma_long'], 1, 
                                         np.where(result_df['sma_short'] < result_df['sma_long'], -1, 0))
        
        # Calcular média móvel exponencial
        result_df['ema_short'] = result_df[price_col].ewm(span=short_window, adjust=False).mean()
        result_df['ema_long'] = result_df[price_col].ewm(span=long_window, adjust=False).mean()
        
        # Identificar tendência com EMA
        result_df['trend_ema'] = np.where(result_df['ema_short'] > result_df['ema_long'], 1,
                                         np.where(result_df['ema_short'] < result_df['ema_long'], -1, 0))
        
        # Calcular ADX para força da tendência (se disponível)
        if 'adx' in result_df.columns:
            result_df['trend_strength'] = result_df['adx'] / 100.0
        else:
            # Indicador simples de força da tendência baseado na distância entre médias móveis
            result_df['trend_strength'] = abs(result_df['sma_short'] - result_df['sma_long']) / result_df['sma_long']
        
        # Identificar Golden Cross (sinal de alta) e Death Cross (sinal de baixa)
        result_df['golden_cross'] = ((result_df['trend_sma'] == 1) & 
                                   (result_df['trend_sma'].shift(1) <= 0)).astype(int)
        result_df['death_cross'] = ((result_df['trend_sma'] == -1) & 
                                  (result_df['trend_sma'].shift(1) >= 0)).astype(int)
        
        # Detectar tendência atual (primária)
        result_df['uptrend'] = (result_df['trend_sma'] == 1).astype(int)
        result_df['downtrend'] = (result_df['trend_sma'] == -1).astype(int)
        result_df['sideways'] = (result_df['trend_sma'] == 0).astype(int)
        
        # Confirmar tendência com múltiplos indicadores (consenso)
        if 'macd_histogram' in result_df.columns:
            result_df['trend_macd'] = np.where(result_df['macd_histogram'] > 0, 1, 
                                             np.where(result_df['macd_histogram'] < 0, -1, 0))
            
            # Consenso de tendência (combinação de SMA, EMA e MACD)
            result_df['trend_consensus'] = (result_df['trend_sma'] + result_df['trend_ema'] + result_df['trend_macd']) / 3
        else:
            # Sem MACD, use apenas SMA e EMA
            result_df['trend_consensus'] = (result_df['trend_sma'] + result_df['trend_ema']) / 2
        
        return result_df


class SignalGenerator:
    """Classe para gerar sinais de trading combinados"""
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gerar sinais de negociação combinando vários indicadores e padrões
        
        Args:
            df: DataFrame com indicadores técnicos e padrões
            
        Returns:
            DataFrame com sinais de trading adicionados
        """
        result_df = df.copy()
        
        # Inicializar colunas de sinal
        result_df['signal'] = 0         # 1 = Compra, -1 = Venda, 0 = Manter
        result_df['signal_strength'] = 0 # Força do sinal de 0 a 1
        result_df['signal_source'] = None
        
        # 1. Sinais baseados em tendência e suporte/resistência
        # Compra: Em tendência de alta e no suporte
        buy_trend_support = ((result_df['uptrend'] == 1) & 
                           (result_df['at_support'] == 1))
        
        # Venda: Em tendência de baixa e na resistência
        sell_trend_resistance = ((result_df['downtrend'] == 1) & 
                               (result_df['at_resistance'] == 1))
        
        # 2. Sinais baseados em cruzamentos de médias móveis
        buy_cross = result_df['golden_cross'] == 1
        sell_cross = result_df['death_cross'] == 1
        
        # 3. Sinais de RSI (se disponível)
        buy_rsi = pd.Series(False, index=result_df.index)
        sell_rsi = pd.Series(False, index=result_df.index)
        
        if 'rsi' in result_df.columns:
            buy_rsi = result_df['rsi'] < 30  # Sobrevenda
            sell_rsi = result_df['rsi'] > 70  # Sobrecompra
        
        # 4. Sinais baseados em padrões de candlestick
        # Identificar padrões de alta
        bullish_patterns = [col for col in result_df.columns if 'pattern_' in col and 'bull' in col]
        # Identificar padrões de baixa
        bearish_patterns = [col for col in result_df.columns if 'pattern_' in col and 'bear' in col]
        
        # Criar sinais baseados em padrões
        buy_pattern = pd.Series(False, index=result_df.index)
        sell_pattern = pd.Series(False, index=result_df.index)
        
        if bullish_patterns:
            buy_pattern = result_df[bullish_patterns].sum(axis=1) > 0
            
        if bearish_patterns:
            sell_pattern = result_df[bearish_patterns].sum(axis=1) > 0
        
        # 5. Sinais baseados em breakouts de suporte e resistência
        breakout_up = pd.Series(False, index=result_df.index)
        breakout_down = pd.Series(False, index=result_df.index)
        
        if 'resistance_level' in result_df.columns:
            breakout_up = ((result_df['close'] > result_df['resistance_level']) & 
                         (result_df['close'].shift(1) <= result_df['resistance_level']))
            
        if 'support_level' in result_df.columns:
            breakout_down = ((result_df['close'] < result_df['support_level']) & 
                           (result_df['close'].shift(1) >= result_df['support_level']))
        
        # Combinar todos os sinais
        # Sinais de compra
        result_df.loc[buy_trend_support, 'signal'] = 1
        result_df.loc[buy_trend_support, 'signal_source'] = 'trend_support'
        result_df.loc[buy_trend_support, 'signal_strength'] = result_df.loc[buy_trend_support, 'support_strength'] * 0.8
        
        result_df.loc[buy_cross, 'signal'] = 1
        result_df.loc[buy_cross, 'signal_source'] = 'golden_cross'
        result_df.loc[buy_cross, 'signal_strength'] = 0.7
        
        result_df.loc[buy_rsi, 'signal'] = 1
        result_df.loc[buy_rsi, 'signal_source'] = 'rsi_oversold'
        if 'rsi' in result_df.columns:
            result_df.loc[buy_rsi, 'signal_strength'] = (30 - result_df.loc[buy_rsi, 'rsi']) / 30 * 0.6
        
        result_df.loc[buy_pattern, 'signal'] = 1
        result_df.loc[buy_pattern, 'signal_source'] = 'bullish_pattern'
        result_df.loc[buy_pattern, 'signal_strength'] = 0.65
        
        result_df.loc[breakout_up, 'signal'] = 1
        result_df.loc[breakout_up, 'signal_source'] = 'resistance_breakout'
        result_df.loc[breakout_up, 'signal_strength'] = 0.75
        
        # Sinais de venda
        result_df.loc[sell_trend_resistance, 'signal'] = -1
        result_df.loc[sell_trend_resistance, 'signal_source'] = 'trend_resistance'
        result_df.loc[sell_trend_resistance, 'signal_strength'] = result_df.loc[sell_trend_resistance, 'resistance_strength'] * 0.8
        
        result_df.loc[sell_cross, 'signal'] = -1
        result_df.loc[sell_cross, 'signal_source'] = 'death_cross'
        result_df.loc[sell_cross, 'signal_strength'] = 0.7
        
        result_df.loc[sell_rsi, 'signal'] = -1
        result_df.loc[sell_rsi, 'signal_source'] = 'rsi_overbought'
        if 'rsi' in result_df.columns:
            result_df.loc[sell_rsi, 'signal_strength'] = (result_df.loc[sell_rsi, 'rsi'] - 70) / 30 * 0.6
        
        result_df.loc[sell_pattern, 'signal'] = -1
        result_df.loc[sell_pattern, 'signal_source'] = 'bearish_pattern'
        result_df.loc[sell_pattern, 'signal_strength'] = 0.65
        
        result_df.loc[breakout_down, 'signal'] = -1
        result_df.loc[breakout_down, 'signal_source'] = 'support_breakdown'
        result_df.loc[breakout_down, 'signal_strength'] = 0.75
        
        # Ajustar força do sinal baseado em confirmação por múltiplos indicadores
        signal_count = (buy_trend_support | buy_cross | buy_rsi | buy_pattern | breakout_up).astype(int) + \
                       (sell_trend_resistance | sell_cross | sell_rsi | sell_pattern | breakout_down).astype(int)
                       
        # Aumentar a força do sinal se confirmado por múltiplos indicadores
        result_df['signal_strength'] = result_df['signal_strength'] * (1 + 0.1 * (signal_count - 1))
        result_df['signal_strength'] = result_df['signal_strength'].clip(0, 1)  # Limitar entre 0 e 1
        
        return result_df


class FeatureEngineering:
    """Classe para engenharia de features para modelos ML"""
    
    @staticmethod
    def create_features(df: pd.DataFrame, 
                       lookback_periods: list = [1, 3, 5, 10, 20],
                       price_cols: list = ['open', 'high', 'low', 'close'],
                       target_horizon: int = 5) -> pd.DataFrame:
        """
        Criar features para modelos de ML a partir de dados de mercado
        
        Args:
            df: DataFrame com dados técnicos e padrões
            lookback_periods: Lista de períodos para calcular variações e médias
            price_cols: Colunas de preço para usar na criação de features
            target_horizon: Horizonte de previsão para o target (dias)
            
        Returns:
            DataFrame com features adicionadas e target
        """
        result_df = df.copy()
        
        # 1. Criar targets para previsão
        # a) Target de direção (classificação)
        result_df[f'target_direction_{target_horizon}d'] = np.where(
            result_df['close'].shift(-target_horizon) > result_df['close'], 1, 0
        )
        
        # b) Target de variação percentual (regressão)
        result_df[f'target_return_{target_horizon}d'] = (
            result_df['close'].shift(-target_horizon) / result_df['close'] - 1
        )
        
        # 2. Features de retorno para diferentes períodos
        for period in lookback_periods:
            result_df[f'return_{period}d'] = result_df['close'].pct_change(period)
            
            # Volatilidade do período
            result_df[f'volatility_{period}d'] = result_df['close'].pct_change().rolling(period).std()
            
            # Range médio do período
            result_df[f'avg_range_{period}d'] = (result_df['high'] - result_df['low']).rolling(period).mean() / result_df['close']
            
            # Volume relativo do período
            if 'volume' in result_df.columns:
                result_df[f'rel_volume_{period}d'] = result_df['volume'] / result_df['volume'].rolling(period).mean()
            
        # 3. Features de médias móveis e indicadores
        if 'sma_short' in result_df.columns and 'sma_long' in result_df.columns:
            # Distância percentual entre SMAs
            result_df['sma_diff_pct'] = (result_df['sma_short'] - result_df['sma_long']) / result_df['sma_long']
        
        if 'rsi' in result_df.columns:
            # RSI normalizado (-1 a 1)
            result_df['rsi_norm'] = (result_df['rsi'] - 50) / 50
            
        if 'macd' in result_df.columns and 'macd_signal' in result_df.columns:
            # Diferença normalizada entre MACD e signal
            result_df['macd_diff_norm'] = result_df['macd'] - result_df['macd_signal']
            if 'close' in result_df.columns:
                result_df['macd_diff_norm'] /= result_df['close']
        
        # 4. Features de padrões e sinais
        # Contar padrões de alta e baixa
        bullish_patterns = [col for col in result_df.columns if 'pattern_' in col and 'bull' in col]
        bearish_patterns = [col for col in result_df.columns if 'pattern_' in col and 'bear' in col]
        
        if bullish_patterns:
            result_df['bullish_pattern_count'] = result_df[bullish_patterns].sum(axis=1)
        
        if bearish_patterns:
            result_df['bearish_pattern_count'] = result_df[bearish_patterns].sum(axis=1)
        
        # 5. Features de suporte e resistência
        if 'at_support' in result_df.columns and 'at_resistance' in result_df.columns:
            # Distância aos níveis de suporte/resistência
            if 'support_level' in result_df.columns:
                result_df['dist_to_support'] = (result_df['close'] - result_df['support_level']) / result_df['close']
                result_df['dist_to_support'].fillna(0, inplace=True)
            
            if 'resistance_level' in result_df.columns:
                result_df['dist_to_resistance'] = (result_df['resistance_level'] - result_df['close']) / result_df['close']
                result_df['dist_to_resistance'].fillna(0, inplace=True)
        
        # 6. Features de tendência
        if 'trend_consensus' in result_df.columns:
            # Já temos uma feature de consenso de tendência
            pass
            
        # 7. Features baseadas em características de tempo
        # Dia da semana (codificação cíclica)
        if not result_df.index.empty and isinstance(result_df.index, pd.DatetimeIndex):
            result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df.index.dayofweek / 7)
            result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df.index.dayofweek / 7)
            
            # Mês do ano (codificação cíclica)
            result_df['month_sin'] = np.sin(2 * np.pi * result_df.index.month / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df.index.month / 12)
        
        return result_df


class MLModelTrainer:
    """Classe para treinar e avaliar modelos de previsão"""
    
    @staticmethod
    def prepare_data_for_ml(df: pd.DataFrame, 
                          target_col: str,
                          feature_cols: list = None,
                          test_size: float = 0.2) -> tuple:
        """
        Preparar dados para modelos de Machine Learning
        
        Args:
            df: DataFrame com features e target
            target_col: Nome da coluna alvo
            feature_cols: Lista de colunas de features a serem usadas
            test_size: Proporção dos dados para teste
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Remover linhas com NaN no target ou features
        df_clean = df.dropna(subset=[target_col])
        
        # Selecionar features automaticamente se não forem especificadas
        if feature_cols is None:
            # Excluir colunas de target e não numéricas
            exclude_cols = [col for col in df_clean.columns if 'target_' in col] + \
                          [col for col in df_clean.columns if df_clean[col].dtype == 'object']
            
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols and col != target_col]
            
            # Log das features usadas
            logger.info(f"Selecionadas automaticamente {len(feature_cols)} features para o modelo")
            
        # Eliminar NaN nas features
        df_clean = df_clean.dropna(subset=feature_cols)
        
        # Divisão temporal para dados de séries temporais
        if isinstance(df_clean.index, pd.DatetimeIndex):
            # Ordenar por data
            df_clean = df_clean.sort_index()
            
            # Dividir cronologicamente
            train_size = int(len(df_clean) * (1 - test_size))
            train_data = df_clean.iloc[:train_size]
            test_data = df_clean.iloc[train_size:]
            
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
        else:
            # Divisão aleatória para dados não temporais
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Escalonar features numéricas
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                              columns=X_train.columns, 
                              index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), 
                             columns=X_test.columns, 
                             index=X_test.index)
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    @staticmethod
    def train_direction_model(X_train, y_train):
        """Treinar modelo de classificação para direção do preço"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calcular importância das features
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, feature_importance
    
    @staticmethod
    def train_return_model(X_train, y_train):
        """Treinar modelo de regressão para retorno esperado"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calcular importância das features
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return model, feature_importance
    
    @staticmethod
    def evaluate_direction_model(model, X_test, y_test):
        """Avaliar modelo de classificação"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return results, y_pred, y_prob
    
    @staticmethod
    def evaluate_return_model(model, X_test, y_test):
        """Avaliar modelo de regressão"""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
            
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        return results, y_pred


def fetch_data(symbol="MSFT", days=365):
    """
    Buscar dados de exemplo ou carregar dados locais
    (Substitua por sua função real de obtenção de dados)
    """
    try:
        # Tentar carregar dados do Yahoo Finance
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(symbol, start=start_date, end=end_date)
        
        logger.info(f"Dados obtidos do Yahoo Finance para {symbol} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        logger.warning(f"Erro ao obter dados do Yahoo Finance: {e}")
        
        # Criar dados sintéticos para teste se necessário
        logger.info("Criando dados sintéticos para teste")
        dates = pd.date_range(end=datetime.now(), periods=days)
        np.random.seed(42)
        
        # Simular preços com tendência e volatilidade realistas
        close = np.random.normal(0, 1, days).cumsum() + 100
        # Garantir que preços sejam positivos
        close = np.maximum(close, 1)
        
        # Criar alta/baixa/abertura realistas
        high = close * np.random.uniform(1.0, 1.05, days)
        low = close * np.random.uniform(0.95, 1.0, days)
        open_price = low + np.random.uniform(0, 1, days) * (high - low)
        volume = np.random.uniform(1000000, 10000000, days)
        
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        return df


def run_technical_analysis_pipeline(df, symbol="MSFT"):
    """Executar a pipeline completa de análise técnica"""
    logger.info("Iniciando pipeline de análise técnica")
    
    # Garantir que nomes de colunas estejam em minúsculas
    df.columns = [col.lower() for col in df.columns]
    
    # 1. Detectar níveis de suporte e resistência
    logger.info("Detectando níveis de suporte e resistência...")
    try:
        df_levels = PatternRecognition.detect_support_resistance_advanced(df)
        logger.info("Níveis de suporte e resistência detectados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao detectar suporte/resistência: {e}")
        # Continuar sem esta etapa se houver erro
        df_levels = df.copy()
    
    # 2. Reconhecer padrões de candlestick
    logger.info("Analisando padrões de candlestick...")
    try:
        df_patterns = CandlestickPatterns.recognize_patterns(df_levels)
        logger.info("Padrões de candlestick reconhecidos com sucesso")
    except Exception as e:
        logger.error(f"Erro ao reconhecer padrões de candlestick: {e}")
        # Continuar sem esta etapa se houver erro
        df_patterns = df_levels.copy()
    
    # 3. Calcular indicadores técnicos adicionais (se necessário)
    logger.info("Calculando indicadores técnicos adicionais...")
    try:
        # Esta função depende da sua implementação de indicadores técnicos
        # Se você não tiver uma implementação específica, podemos adicionar alguns indicadores básicos
        
        # RSI (Relative Strength Index)
        delta = df_patterns['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_patterns['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df_patterns['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_patterns['close'].ewm(span=26, adjust=False).mean()
        df_patterns['macd'] = exp1 - exp2
        df_patterns['macd_signal'] = df_patterns['macd'].ewm(span=9, adjust=False).mean()
        df_patterns['macd_histogram'] = df_patterns['macd'] - df_patterns['macd_signal']
        
        # Bollinger Bands
        df_patterns['sma20'] = df_patterns['close'].rolling(window=20).mean()
        std20 = df_patterns['close'].rolling(window=20).std()
        df_patterns['bb_upper'] = df_patterns['sma20'] + (std20 * 2)
        df_patterns['bb_lower'] = df_patterns['sma20'] - (std20 * 2)
        
        # ADX (Average Directional Index) para medição de força de tendência
        high_diff = df_patterns['high'].diff()
        low_diff = -df_patterns['low'].diff()
        
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
        
        tr1 = df_patterns['high'] - df_patterns['low']
        tr2 = abs(df_patterns['high'] - df_patterns['close'].shift(1))
        tr3 = abs(df_patterns['low'] - df_patterns['close'].shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = true_range.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df_patterns['adx'] = dx.rolling(window=14).mean()
        
        logger.info("Indicadores técnicos calculados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao calcular indicadores técnicos: {e}")
        # Continuar sem esta etapa se houver erro
    
    # 4. Identificar tendências
    logger.info("Identificando tendências de mercado...")
    try:
        df_trends = TrendIdentifier.identify_trends(df_patterns)
        logger.info("Tendências identificadas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao identificar tendências: {e}")
        # Continuar sem esta etapa se houver erro
        df_trends = df_patterns.copy()
    
    # 5. Gerar sinais combinados
    logger.info("Gerando sinais de trading...")
    try:
        df_signals = SignalGenerator.generate_signals(df_trends)
        logger.info("Sinais gerados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao gerar sinais: {e}")
        # Continuar sem esta etapa se houver erro
        df_signals = df_trends.copy()
    
    # 6. Criar features para modelos de ML
    logger.info("Criando features para modelos de ML...")
    try:
        df_features = FeatureEngineering.create_features(
            df_signals, 
            lookback_periods=[1, 3, 5, 10, 20], 
            target_horizon=5
        )
        logger.info("Features criadas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao criar features: {e}")
        # Continuar sem esta etapa se houver erro
        df_features = df_signals.copy()
    
    # 7. Treinar e avaliar modelos
    logger.info("Preparando dados para treinamento de modelos...")
    try:
        # Preparar dados para modelo de direção (classificação)
        target_direction = f'target_direction_5d'  # Previsão de direção em 5 dias
        X_train, X_test, y_train, y_test, feature_cols = MLModelTrainer.prepare_data_for_ml(
            df_features, target_direction, test_size=0.2
        )
        
        # Treinar modelo de direção
        logger.info("Treinando modelo de classificação para direção...")
        direction_model, dir_feature_importance = MLModelTrainer.train_direction_model(X_train, y_train)
        direction_results, dir_y_pred, dir_y_prob = MLModelTrainer.evaluate_direction_model(
            direction_model, X_test, y_test
        )
        
        logger.info(f"Resultados do modelo de direção: {direction_results}")
        
        # Preparar dados para modelo de retorno (regressão)
        target_return = f'target_return_5d'  # Previsão de retorno em 5 dias
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_cols_reg = MLModelTrainer.prepare_data_for_ml(
            df_features, target_return, test_size=0.2
        )
        
        # Treinar modelo de retorno
        logger.info("Treinando modelo de regressão para retorno...")
        return_model, ret_feature_importance = MLModelTrainer.train_return_model(X_train_reg, y_train_reg)
        return_results, ret_y_pred = MLModelTrainer.evaluate_return_model(
            return_model, X_test_reg, y_test_reg
        )
        
        logger.info(f"Resultados do modelo de retorno: {return_results}")
        
    except Exception as e:
        logger.error(f"Erro no treinamento de modelos: {e}")
        direction_model = None
        return_model = None
        dir_feature_importance = None
        ret_feature_importance = None
        direction_results = {}
        return_results = {}
    
    # 8. Visualizar resultados
    logger.info("Gerando visualizações...")
    try:
        # Configuração de visualização
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (14, 8)
        
        # Criar diretório para salvar visualizações
        vis_dir = os.path.join('outputs', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Gráfico de preços com suporte, resistência e sinais
        plt.figure(figsize=(16, 10))
        
        # Plot de preços (candles)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        
        # Plot de fechamento
        ax1.plot(df_signals.index, df_signals['close'], label='Preço de Fechamento', color='blue', alpha=0.6)
        
        # Adicionar médias móveis
        if 'sma_short' in df_signals.columns and 'sma_long' in df_signals.columns:
            ax1.plot(df_signals.index, df_signals['sma_short'], label='SMA Curta', color='orange')
            ax1.plot(df_signals.index, df_signals['sma_long'], label='SMA Longa', color='red')
        
        # Adicionar níveis de suporte e resistência
        if 'support_level' in df_signals.columns:
            for level in df_signals['support_level'].dropna().unique():
                if level > 0:  # Validar nível
                    ax1.axhline(level, linestyle='--', alpha=0.7, color='green', linewidth=1)
        
        if 'resistance_level' in df_signals.columns:
            for level in df_signals['resistance_level'].dropna().unique():
                if level > 0:  # Validar nível
                    ax1.axhline(level, linestyle='--', alpha=0.7, color='red', linewidth=1)
        
        # Adicionar sinais de compra e venda
        buy_signals = df_signals[df_signals['signal'] == 1]
        sell_signals = df_signals[df_signals['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Compra')
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Venda')
        
        ax1.set_title(f'Análise Técnica para {symbol}', fontsize=15)
        ax1.set_ylabel('Preço', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot de indicadores
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=ax1)
        
        # RSI
        if 'rsi' in df_signals.columns:
            ax2.plot(df_signals.index, df_signals['rsi'], label='RSI', color='purple')
            ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
            ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
            ax2.set_ylabel('RSI', fontsize=12)
            ax2.set_ylim(0, 100)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot de MACD
        ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        
        if 'macd' in df_signals.columns and 'macd_signal' in df_signals.columns:
            ax3.plot(df_signals.index, df_signals['macd'], label='MACD', color='blue')
            ax3.plot(df_signals.index, df_signals['macd_signal'], label='Sinal', color='red')
            
            if 'macd_histogram' in df_signals.columns:
                ax3.bar(df_signals.index, df_signals['macd_histogram'], label='Histograma', color='gray', alpha=0.3)
            
            ax3.set_ylabel('MACD', fontsize=12)
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{symbol}_technical_analysis.png'), dpi=300)
        plt.close()
        
        # 2. Visualização de importância de features
        if dir_feature_importance is not None:
            plt.figure(figsize=(12, 6))
            dir_feature_importance.head(15).plot(kind='barh')
            plt.title('Importância das Features - Modelo de Direção', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{symbol}_feature_importance_direction.png'), dpi=300)
            plt.close()
        
        if ret_feature_importance is not None:
            plt.figure(figsize=(12, 6))
            ret_feature_importance.head(15).plot(kind='barh')
            plt.title('Importância das Features - Modelo de Retorno', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{symbol}_feature_importance_return.png'), dpi=300)
            plt.close()
        
        # 3. Visualização de performance do modelo
        if direction_model is not None:
            plt.figure(figsize=(14, 6))
            
            # Cronologicamente (para séries temporais)
            combined = pd.DataFrame({
                'Atual': y_test,
                'Previsto': dir_y_pred,
                'Probabilidade': dir_y_prob
            }, index=X_test.index)
            
            ax = combined[['Atual', 'Previsto']].plot(figsize=(14, 6))
            ax2 = ax.twinx()
            combined['Probabilidade'].plot(ax=ax2, color='green', alpha=0.5)
            
            plt.title('Performance do Modelo de Direção', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{symbol}_model_performance_direction.png'), dpi=300)
            plt.close()
        
        if return_model is not None:
            plt.figure(figsize=(14, 6))
            
            # Cronologicamente (para séries temporais)
            combined = pd.DataFrame({
                'Atual': y_test_reg,
                'Previsto': ret_y_pred
            }, index=X_test_reg.index)
            
            combined.plot(figsize=(14, 6))
            plt.title('Performance do Modelo de Retorno', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{symbol}_model_performance_return.png'), dpi=300)
            plt.close()
        
        logger.info(f"Visualizações salvas em: {vis_dir}")
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")
    
    # 9. Preparar e retornar resultados
    results = {
        'data': df_signals,
        'models': {
            'direction': direction_model,
            'return': return_model
        },
        'feature_importance': {
            'direction': dir_feature_importance,
            'return': ret_feature_importance
        },
        'performance': {
            'direction': direction_results,
            'return': return_results
        },
        'visualizations_path': vis_dir
    }
    
    return results


def save_results(results, symbol="MSFT"):
    """Salvar resultados em arquivos para análise posterior"""
    
    # Definir diretório de saída
    output_dir = os.path.join('outputs', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar dados processados
    results['data'].to_csv(os.path.join(output_dir, f'{symbol}_processed_data.csv'))
    
    # Salvar métricas de performance
    with open(os.path.join(output_dir, f'{symbol}_model_metrics.txt'), 'w') as f:
        f.write("=== Métricas do Modelo de Direção ===\n")
        for metric, value in results['performance']['direction'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n=== Métricas do Modelo de Retorno ===\n")
        for metric, value in results['performance']['return'].items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Salvar importância de features
    if results['feature_importance']['direction'] is not None:
        results['feature_importance']['direction'].to_csv(
            os.path.join(output_dir, f'{symbol}_feature_importance_direction.csv')
        )
    
    if results['feature_importance']['return'] is not None:
        results['feature_importance']['return'].to_csv(
            os.path.join(output_dir, f'{symbol}_feature_importance_return.csv')
        )
    
    # Salvar modelos (usando pickle)
    import pickle
    
    if results['models']['direction'] is not None:
        with open(os.path.join(output_dir, f'{symbol}_direction_model.pkl'), 'wb') as f:
            pickle.dump(results['models']['direction'], f)
    
    if results['models']['return'] is not None:
        with open(os.path.join(output_dir, f'{symbol}_return_model.pkl'), 'wb') as f:
            pickle.dump(results['models']['return'], f)
    
    logger.info(f"Resultados salvos em: {output_dir}")
    return output_dir


def generate_report(results, symbol="MSFT"):
    """Gerar relatório de análise técnica"""
    
    report_dir = os.path.join('outputs', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # Extrair dados importantes
    df = results['data']
    last_price = df['close'].iloc[-1]
    
    # Contar sinais recentes (últimas 5 barras)
    recent_buys = df['signal'].iloc[-5:].eq(1).sum()
    recent_sells = df['signal'].iloc[-5:].eq(-1).sum()
    
    # Obter sinal mais recente
    latest_signal = df['signal'].iloc[-1]
    latest_signal_source = df['signal_source'].iloc[-1] if 'signal_source' in df.columns else "N/A"
    
    # Calcular probabilidade de alta pelo modelo (se disponível)
    model_probability = None
    if results['models']['direction'] is not None:
        # Preparar dados para previsão
        latest_data = df.iloc[-1:].copy()
        feature_cols = results['feature_importance']['direction'].index.tolist()
        valid_cols = [col for col in feature_cols if col in latest_data.columns]
        
        if len(valid_cols) > 0:
            X = latest_data[valid_cols]
            # Normalizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            try:
                # Prever probabilidade
                model_probability = results['models']['direction'].predict_proba(X_scaled)[0][1]
            except Exception as e:
                logger.warning(f"Não foi possível fazer previsão com o modelo: {e}")
    
    # Gerar relatório em texto
    report_path = os.path.join(report_dir, f'{symbol}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"=== RELATÓRIO DE ANÁLISE TÉCNICA: {symbol} ===\n")
        f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Último preço: {last_price:.2f}\n")
        
        if 'sma_short' in df.columns and 'sma_long' in df.columns:
            f.write(f"SMA Curta: {df['sma_short'].iloc[-1]:.2f}\n")
            f.write(f"SMA Longa: {df['sma_long'].iloc[-1]:.2f}\n")
        
        if 'trend_consensus' in df.columns:
            trend = df['trend_consensus'].iloc[-1]
            trend_str = "ALTA" if trend > 0.3 else "BAIXA" if trend < -0.3 else "LATERAL"
            f.write(f"Tendência atual: {trend_str} ({trend:.2f})\n")
        
        f.write(f"\nSinais recentes (últimos 5 períodos):\n")
        f.write(f"- Sinais de compra: {recent_buys}\n")
        f.write(f"- Sinais de venda: {recent_sells}\n")
        
        f.write(f"\nÚltimo sinal gerado: ")
        if latest_signal == 1:
            f.write(f"COMPRA (Fonte: {latest_signal_source})\n")
        elif latest_signal == -1:
            f.write(f"VENDA (Fonte: {latest_signal_source})\n")
        else:
            f.write(f"NEUTRO\n")
        
        if model_probability is not None:
            f.write(f"\nProjeção do modelo ML:\n")
            f.write(f"- Probabilidade de alta em 5 dias: {model_probability:.2%}\n")
        
        f.write("\n=== PERFORMANCE DOS MODELOS ===\n")
        for metric, value in results['performance']['direction'].items():
            f.write(f"- {metric}: {value:.4f}\n")
        
        f.write("\n=== FEATURES MAIS IMPORTANTES ===\n")
        if results['feature_importance']['direction'] is not None:
            top_features = results['feature_importance']['direction'].head(5)
            for feature, importance in top_features.items():
                f.write(f"- {feature}: {importance:.4f}\n")
    
    logger.info(f"Relatório gerado em: {report_path}")
    return report_path


def main():
    """Ponto de entrada principal do programa"""
    # Configuração
    symbols = ["MSFT", "AAPL", "GOOG"]  # Símbolos para análise
    days = 365  # Dias de histórico
    
    # Processar cada símbolo
    for symbol in symbols:
        try:
            logger.info(f"Iniciando análise para {symbol}")
            
            # Buscar dados
            df = fetch_data(symbol=symbol, days=days)
            
            # Executar pipeline de análise
            results = run_technical_analysis_pipeline(df, symbol=symbol)
            
            # Salvar resultados
            output_dir = save_results(results, symbol=symbol)
            
            # Gerar relatório
            report_path = generate_report(results, symbol=symbol)
            
            logger.info(f"Análise para {symbol} concluída com sucesso")
            logger.info(f"Resultados em: {output_dir}")
            logger.info(f"Relatório em: {report_path}")
            
        except Exception as e:
            logger.error(f"Erro ao processar {symbol}: {e}")
    
    logger.info("Pipeline de análise técnica concluída")


if __name__ == "__main__":
    main()