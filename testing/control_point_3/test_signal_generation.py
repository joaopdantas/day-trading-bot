import unittest
import pandas as pd
import numpy as np
from src.indicators.technical import SignalGeneration, TechnicalIndicators, PatternRecognition

class TestSignalGeneration(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 150, 100),
            'high': np.random.uniform(120, 160, 100),
            'low': np.random.uniform(90, 110, 100),
            'close': np.random.uniform(100, 150, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        # Make the data more realistic
        for i in range(1, len(self.test_data)):
            self.test_data.iloc[i, self.test_data.columns.get_loc('high')] = max(
                self.test_data.iloc[i, self.test_data.columns.get_loc('open')],
                self.test_data.iloc[i, self.test_data.columns.get_loc('close')]
            ) + np.random.uniform(0, 5)
            self.test_data.iloc[i, self.test_data.columns.get_loc('low')] = min(
                self.test_data.iloc[i, self.test_data.columns.get_loc('open')],
                self.test_data.iloc[i, self.test_data.columns.get_loc('close')]
            ) - np.random.uniform(0, 5)
        
        # Add all technical indicators and patterns
        self.df_with_indicators = TechnicalIndicators.add_all_indicators(self.test_data)
        self.df_with_indicators = PatternRecognition.recognize_candlestick_patterns(self.df_with_indicators)
        self.df_with_indicators = PatternRecognition.detect_support_resistance(self.df_with_indicators)
        self.df_with_indicators = PatternRecognition.detect_trend(self.df_with_indicators)

    def test_combine_signals(self):
        # Test signal combination
        result = SignalGeneration.combine_signals(
            self.df_with_indicators,
            rsi_thresholds={'oversold': 30, 'overbought': 70},
            macd_threshold=0,
            volume_factor=1.5,
            trend_strength_threshold=2.0
        )
        
        # Check if all required columns are present
        required_columns = ['buy_signal', 'sell_signal', 'signal_strength']
        for column in required_columns:
            self.assertIn(column, result.columns)
        
        # Check if signals are binary (0 or 1)
        self.assertTrue(all(result['buy_signal'].isin([0, 1])))
        self.assertTrue(all(result['sell_signal'].isin([0, 1])))
        
        # Check if signal strength is within expected range (-100 to 100)
        self.assertTrue(all(result['signal_strength'].between(-100, 100, inclusive='both')))

    def test_generate_trade_recommendations(self):
        # First generate signals
        df_with_signals = SignalGeneration.combine_signals(self.df_with_indicators)
        
        # Then test trade recommendation generation
        result = SignalGeneration.generate_trade_recommendations(
            df_with_signals,
            min_confidence=0.7,
            max_risk_percent=5.0
        )
        
        # Check if all required columns are present
        required_columns = ['trade_recommendation', 'recommendation_reason']
        for column in required_columns:
            self.assertIn(column, result.columns)
        
        # Check if recommendations are valid
        valid_recommendations = ['BUY', 'SELL', 'HOLD']
        self.assertTrue(all(result['trade_recommendation'].isin(valid_recommendations)))
        
        # Check if reasons are provided for all recommendations
        self.assertTrue(all(result['recommendation_reason'].notna()))

    def test_edge_cases(self):
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = SignalGeneration.combine_signals(empty_df)
        self.assertTrue(result_empty.empty)
        
        # Test with missing columns
        incomplete_df = pd.DataFrame({'close': [100, 101, 102]})
        result_incomplete = SignalGeneration.combine_signals(incomplete_df)
        self.assertFalse(result_incomplete.empty)  # Should handle gracefully
        
        # Test with NaN values
        df_with_nan = self.df_with_indicators.copy()
        df_with_nan.loc[0, 'rsi'] = np.nan
        result_nan = SignalGeneration.combine_signals(df_with_nan)
        self.assertFalse(result_nan.empty)  # Should handle NaN values

    def test_signal_consistency(self):
        # Generate signals
        signals = SignalGeneration.combine_signals(self.df_with_indicators)
        
        # Check that we don't have simultaneous buy and sell signals
        self.assertTrue(all((signals['buy_signal'] + signals['sell_signal']) <= 1))
        
        # Check signal strength correlation with buy/sell signals
        buy_signals = signals[signals['buy_signal'] == 1]
        sell_signals = signals[signals['sell_signal'] == 1]
        
        # If we have any buy signals, their strength should be positive
        if not buy_signals.empty:
            self.assertTrue(all(buy_signals['signal_strength'] > 0))
        
        # If we have any sell signals, their strength should be negative
        if not sell_signals.empty:
            self.assertTrue(all(sell_signals['signal_strength'] < 0))

if __name__ == '__main__':
    unittest.main()