"""
Portfolio Manager - True Ultimate Portfolio Implementation
File: src/backtesting/portfolio_manager.py

Implements the TRUE Ultimate Portfolio approach using separate backtests 
with split capital, exactly like the working portfolio optimization methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Portfolio Manager that implements the TRUE Ultimate Portfolio approach
    by running separate backtests and combining results (not a single strategy)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        strategies_config: Dict = None,
        assets: List[str] = None
    ):
        """
        Initialize Portfolio Manager with strategy configurations
        
        Args:
            initial_capital: Total capital to split across strategies
            transaction_cost: Transaction cost percentage
            strategies_config: Dictionary defining strategies and their parameters
            assets: List of assets to trade (default: ['MSFT'] for single asset testing)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Default strategy configurations (matching portfolio optimization)
        if strategies_config is None:
            self.strategies_config = {
                'Technical Analysis': {
                    'class': 'TechnicalAnalysisStrategy',
                    'params': {
                        'sma_short': 20, 'sma_long': 50,
                        'rsi_oversold': 30, 'rsi_overbought': 70
                    }
                },
                'MLTrading Strategy': {
                    'class': 'MLTradingStrategy', 
                    'params': {
                        'confidence_threshold': 0.40,
                        'rsi_oversold': 30, 'rsi_overbought': 70
                    }
                }
            }
        else:
            self.strategies_config = strategies_config
        
        # Default assets (for single asset testing, this will be ['MSFT'])
        if assets is None:
            self.assets = ['MSFT']  # Single asset for testing
        else:
            self.assets = assets
        
        # Calculate capital allocation
        self.total_combinations = len(self.strategies_config) * len(self.assets)
        self.capital_per_combination = self.initial_capital // self.total_combinations
        self.combination_weight = 1.0 / self.total_combinations
        
        logger.info(f"Portfolio Manager initialized:")
        logger.info(f"   Strategies: {list(self.strategies_config.keys())}")
        logger.info(f"   Assets: {self.assets}")
        logger.info(f"   Total combinations: {self.total_combinations}")
        logger.info(f"   Capital per combination: ${self.capital_per_combination:,}")
    
    def run_portfolio_backtest(
        self, 
        data: pd.DataFrame, 
        backtester_class, 
        strategy_classes: Dict
    ) -> Dict:
        """
        Run the TRUE Ultimate Portfolio backtest methodology
        Exactly like the working portfolio optimization script
        
        Args:
            data: Market data for backtesting
            backtester_class: ProductionBacktester class
            strategy_classes: Dictionary mapping strategy names to classes
        
        Returns:
            Combined portfolio results dictionary
        """
        
        logger.info("🏆 RUNNING TRUE ULTIMATE PORTFOLIO BACKTEST")
        
        portfolio_results = {}
        total_return = 0
        total_trades = 0
        all_combination_results = []
        
        for strategy_name, strategy_config in self.strategies_config.items():
            logger.info(f"📊 Testing {strategy_name}:")
            
            strategy_results = {}
            
            for asset in self.assets:
                combination_name = f"{strategy_name}_{asset}"
                logger.info(f"   {combination_name} (${self.capital_per_combination:,} capital):")
                
                try:
                    # Create strategy instance
                    strategy_class_name = strategy_config['class']
                    if strategy_class_name not in strategy_classes:
                        raise ValueError(f"Strategy class {strategy_class_name} not found in strategy_classes")
                    
                    strategy_class = strategy_classes[strategy_class_name]
                    strategy = strategy_class(**strategy_config['params'])
                    
                    # Run separate backtest with split capital
                    backtester = backtester_class(
                        initial_capital=self.capital_per_combination,
                        transaction_cost=self.transaction_cost,
                        max_position_size=1.0  # Full capital of the allocation
                    )
                    
                    backtester.set_strategy(strategy)
                    results = backtester.run_backtest(data)
                    
                    # Store individual combination results
                    combination_result = {
                        'strategy': strategy_name,
                        'asset': asset,
                        'return': results['total_return'],
                        'trades': results['total_trades'],
                        'win_rate': results.get('win_rate', 0),
                        'sharpe': results.get('sharpe_ratio', 0),
                        'capital': self.capital_per_combination,
                        'final_value': self.capital_per_combination * (1 + results['total_return']),
                        'weight': self.combination_weight,
                        'backtester': backtester  # Store for signal/trade history access
                    }
                    
                    all_combination_results.append(combination_result)
                    strategy_results[asset] = combination_result
                    
                    # Add to portfolio totals (weighted)
                    total_return += results['total_return'] * self.combination_weight
                    total_trades += results['total_trades']
                    
                    logger.info(f"      Return: {results['total_return']*100:+6.2f}%")
                    logger.info(f"      Trades: {results['total_trades']:2d}")
                    logger.info(f"      Final Value: ${combination_result['final_value']:.2f}")
                    
                except Exception as e:
                    logger.error(f"      ❌ Error: {e}")
            
            portfolio_results[strategy_name] = strategy_results
        
        # Calculate final portfolio performance
        logger.info(f"🎯 ULTIMATE PORTFOLIO PERFORMANCE:")
        logger.info(f"   Portfolio Return: {total_return*100:+6.2f}%")
        logger.info(f"   Total Trades: {total_trades}")
        logger.info(f"   Trade Frequency: {total_trades/12:.1f} trades/month")
        logger.info(f"   Strategy-Asset Combinations: {self.total_combinations}")
        
        # Create combined results (matching original format)
        combined_results = {
            'strategy_name': '🏆 Ultimate Portfolio Strategy',
            'total_return': total_return,
            'total_trades': total_trades,
            'trade_frequency_monthly': total_trades / 12,
            'combinations': self.total_combinations,
            'individual_results': all_combination_results,
            'portfolio_breakdown': portfolio_results,
            'methodology': 'Portfolio Manager (split capital)'
        }
        
        # Calculate additional metrics
        if all_combination_results:
            combined_results['win_rate'] = np.mean([r['win_rate'] for r in all_combination_results])
            combined_results['sharpe_ratio'] = np.mean([r['sharpe'] for r in all_combination_results])
        
        # Add final portfolio value
        total_final_value = sum(r['final_value'] for r in all_combination_results)
        combined_results['final_value'] = total_final_value
        combined_results['initial_capital'] = self.initial_capital
        
        return combined_results
    
    def get_combined_signals_and_trades(self, portfolio_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract and combine signals and trades from all strategy combinations
        
        Args:
            portfolio_results: Results from run_portfolio_backtest
            
        Returns:
            Tuple of (combined_signals_df, combined_trades_df)
        """
        
        all_signals = []
        all_trades = []
        
        if 'individual_results' not in portfolio_results:
            return pd.DataFrame(), pd.DataFrame()
        
        for combination_result in portfolio_results['individual_results']:
            if 'backtester' in combination_result:
                backtester = combination_result['backtester']
                strategy_name = combination_result['strategy']
                asset = combination_result['asset']
                
                # Get signals
                signals_df = backtester.get_signals_history()
                if not signals_df.empty:
                    signals_df = signals_df.copy()
                    signals_df['strategy_source'] = strategy_name
                    signals_df['asset'] = asset
                    signals_df['combination'] = f"{strategy_name}_{asset}"
                    all_signals.append(signals_df)
                
                # Get trades
                trades_df = backtester.get_trade_history()
                if not trades_df.empty:
                    trades_df = trades_df.copy()
                    trades_df['strategy_source'] = strategy_name
                    trades_df['asset'] = asset
                    trades_df['combination'] = f"{strategy_name}_{asset}"
                    all_trades.append(trades_df)
        
        # Combine DataFrames
        combined_signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        return combined_signals, combined_trades
    
    def verify_portfolio_calculation(self, individual_results: Dict, portfolio_results: Dict) -> Dict:
        """
        Verify that Portfolio calculation matches expected values
        
        Args:
            individual_results: Results from individual strategy tests
            portfolio_results: Results from portfolio backtest
            
        Returns:
            Verification results dictionary
        """
        
        logger.info("✅ ULTIMATE PORTFOLIO CALCULATION VERIFICATION:")
        
        # Extract portfolio results
        ultimate_return = portfolio_results['total_return']
        ultimate_trades = portfolio_results['total_trades']
        
        # Calculate expected results based on individual strategies
        strategy_names = list(self.strategies_config.keys())
        capital_per_strategy = self.initial_capital // len(strategy_names)
        
        expected_total_value = 0
        expected_trades = 0
        
        logger.info("Individual Strategy Performance:")
        for strategy_name in strategy_names:
            if strategy_name in individual_results:
                individual_return = individual_results[strategy_name]['total_return']
                individual_trades = individual_results[strategy_name]['total_trades']
                profit = individual_return * capital_per_strategy
                
                expected_total_value += capital_per_strategy + profit
                expected_trades += individual_trades
                
                logger.info(f"   {strategy_name}: {individual_return*100:+6.2f}% on ${capital_per_strategy:,} = ${profit:.2f} profit")
        
        expected_return = (expected_total_value - self.initial_capital) / self.initial_capital
        
        logger.info(f"\nExpected Ultimate Portfolio:")
        logger.info(f"   Expected Return: {expected_return*100:+6.2f}%")
        logger.info(f"   Expected Trades: {expected_trades}")
        logger.info(f"   Expected Final Value: ${expected_total_value:.2f}")
        
        logger.info(f"\nActual Ultimate Portfolio:")
        logger.info(f"   Actual Return: {ultimate_return*100:+6.2f}%")
        logger.info(f"   Actual Trades: {ultimate_trades}")
        logger.info(f"   Actual Final Value: ${portfolio_results['final_value']:.2f}")
        
        # Verification
        return_diff = abs(ultimate_return - expected_return)
        trade_diff = abs(ultimate_trades - expected_trades)
        value_diff = abs(portfolio_results['final_value'] - expected_total_value)
        
        logger.info(f"\nVerification:")
        logger.info(f"   Return Difference: {return_diff*100:.3f}%")
        logger.info(f"   Trade Difference: {trade_diff}")
        logger.info(f"   Value Difference: ${value_diff:.2f}")
        
        if return_diff < 0.001 and trade_diff == 0 and value_diff < 1.0:
            logger.info("   ✅ PERFECT: Ultimate Portfolio calculation is CORRECT!")
            verification_status = "PERFECT"
        elif return_diff < 0.01 and trade_diff <= 1:
            logger.info("   👍 GOOD: Ultimate Portfolio calculation is close to expected")
            verification_status = "GOOD"
        else:
            logger.info("   ⚠️  ISSUE: Ultimate Portfolio calculation differs from expected")
            verification_status = "ISSUE"
        
        return {
            'verification_status': verification_status,
            'expected_return': expected_return,
            'actual_return': ultimate_return,
            'expected_trades': expected_trades,
            'actual_trades': ultimate_trades,
            'return_difference': return_diff,
            'trade_difference': trade_diff,
            'value_difference': value_diff
        }


class UltimatePortfolioRunner:
    """
    Convenient wrapper for running Ultimate Portfolio tests
    """
    
    def __init__(self, assets: List[str] = None, initial_capital: float = 10000):
        """
        Initialize Ultimate Portfolio Runner
        
        Args:
            assets: List of assets to trade (default: ['MSFT'])
            initial_capital: Total capital to allocate
        """
        self.assets = assets or ['MSFT']
        self.initial_capital = initial_capital
        self.portfolio_manager = None
        self.results = None
        
    def run_ultimate_portfolio_test(
        self, 
        data: pd.DataFrame, 
        backtester_class, 
        strategy_classes: Dict,
        custom_strategies_config: Dict = None
    ) -> Dict:
        """
        Run Ultimate Portfolio test using the TRUE methodology
        
        Args:
            data: Market data for backtesting
            backtester_class: ProductionBacktester class
            strategy_classes: Dictionary mapping strategy names to classes
            custom_strategies_config: Optional custom strategy configuration
            
        Returns:
            Portfolio results dictionary
        """
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.initial_capital,
            assets=self.assets,
            strategies_config=custom_strategies_config
        )
        
        # Run the portfolio backtest
        self.results = self.portfolio_manager.run_portfolio_backtest(
            data, backtester_class, strategy_classes
        )
        
        return self.results
    
    def compare_with_individual_strategies(self, individual_results: Dict) -> Dict:
        """
        Compare Ultimate Portfolio results with individual strategy results
        
        Args:
            individual_results: Dictionary of individual strategy results
            
        Returns:
            Comparison results dictionary
        """
        
        if not self.results or not self.portfolio_manager:
            logger.error("No Ultimate Portfolio results to compare")
            return {}
        
        logger.info("🔍 ULTIMATE PORTFOLIO vs INDIVIDUAL STRATEGIES:")
        
        ultimate_return = self.results['total_return']
        ultimate_trades = self.results['total_trades']
        
        logger.info("ACTUAL RESULTS:")
        logger.info(f"   🏆 Ultimate Portfolio: {ultimate_return*100:+6.2f}% return, {ultimate_trades} trades")
        
        for strategy_name, results in individual_results.items():
            if isinstance(results, dict) and 'total_return' in results:
                logger.info(f"   📊 {strategy_name}: {results['total_return']*100:+6.2f}% return, {results['total_trades']} trades")
        
        # Run verification
        verification = self.portfolio_manager.verify_portfolio_calculation(
            individual_results, self.results
        )
        
        return verification
    
    def get_signals_and_trades_for_visualization(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get combined signals and trades for visualization
        
        Returns:
            Tuple of (signals_df, trades_df)
        """
        
        if not self.results or not self.portfolio_manager:
            return pd.DataFrame(), pd.DataFrame()
        
        return self.portfolio_manager.get_combined_signals_and_trades(self.results)


# Utility functions
def create_default_strategy_config() -> Dict:
    """Create default strategy configuration for Ultimate Portfolio"""
    return {
        'Technical Analysis': {
            'class': 'TechnicalAnalysisStrategy',
            'params': {
                'sma_short': 20, 'sma_long': 50,
                'rsi_oversold': 30, 'rsi_overbought': 70
            }
        },
        'MLTrading Strategy': {
            'class': 'MLTradingStrategy', 
            'params': {
                'confidence_threshold': 0.40,
                'rsi_oversold': 30, 'rsi_overbought': 70
            }
        }
    }


def create_multi_asset_strategy_config(assets: List[str]) -> Dict:
    """
    Create strategy configuration for multi-asset Ultimate Portfolio
    
    Args:
        assets: List of assets to trade
        
    Returns:
        Strategy configuration dictionary
    """
    
    # Optimize asset selection based on portfolio optimization results
    optimized_assets = {
        'technical': [asset for asset in assets if asset != 'AAPL'],  # AAPL performs poorly with Technical Analysis
        'ml': assets  # ML handles all assets including AAPL
    }
    
    return {
        'Technical Analysis': {
            'class': 'TechnicalAnalysisStrategy',
            'params': {
                'sma_short': 20, 'sma_long': 50,
                'rsi_oversold': 30, 'rsi_overbought': 70
            },
            'assets': optimized_assets['technical']
        },
        'MLTrading Strategy': {
            'class': 'MLTradingStrategy', 
            'params': {
                'confidence_threshold': 0.40,
                'rsi_oversold': 30, 'rsi_overbought': 70
            },
            'assets': optimized_assets['ml']
        }
    }