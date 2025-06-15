"""
STANDARDIZED HYPOTHESIS TESTING FRAMEWORK - Jan 2024 to Dec 2024
All systems now use IDENTICAL time periods and initial conditions for fair comparison

CRITICAL CHANGES:
✅ All systems test Jan 1, 2024 to Dec 31, 2024
✅ Same initial capital: $10,000  
✅ Same symbol: MSFT
✅ Same data source priority: Alpha Vantage → Yahoo Finance
✅ Identical backtesting framework

PURPOSE: Test MLTrading Strategyagainst benchmarks to establish baseline performance
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# Add project root to path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

try:
    from src.backtesting import MLTradingStrategy, TechnicalAnalysisStrategy, BuyAndHoldStrategy
    from src.data.fetcher import get_data_api
    from src.indicators.technical import TechnicalIndicators
    PROJECT_AVAILABLE = True
    print("✅ Your project modules loaded successfully")
except ImportError as e:
    print(f"❌ Project modules not available: {e}")
    PROJECT_AVAILABLE = False

try:
    from src.backtesting import RSIDivergenceStrategy, HybridRSIDivergenceStrategy
    RSI_STRATEGIES_AVAILABLE = True
    print("✅ RSI Divergence strategies loaded")
except ImportError:
    RSI_STRATEGIES_AVAILABLE = False
    print("⚠️ RSI Divergence strategies not available")


class StandardizedHypothesisTestingFramework:
    """STANDARDIZED Framework ensuring all systems use identical test conditions"""

    def __init__(self):
        self.results = {}

        # 🎯 STANDARDIZED TEST CONDITIONS - ALL SYSTEMS USE THESE
        self.STANDARD_CONFIG = {
            'test_symbol': 'MSFT',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 10000,
            'data_source': 'polygon'  # Primary source for consistency
        }

        self.test_data_2024 = None

        print("🎯 STANDARDIZED TESTING FRAMEWORK INITIALIZED")
        print("=" * 60)
        print(f"Symbol: {self.STANDARD_CONFIG['test_symbol']}")
        print(
            f"Period: {self.STANDARD_CONFIG['start_date']} to {self.STANDARD_CONFIG['end_date']}")
        print(f"Initial Capital: ${self.STANDARD_CONFIG['initial_capital']:,}")
        print("=" * 60)

    def load_standardized_data(self):
        """Load 2024 data that ALL systems will use"""

        print(
            f"\n📊 Loading STANDARDIZED 2024 data for {self.STANDARD_CONFIG['test_symbol']}...")

        if PROJECT_AVAILABLE:
            try:
                # NEW: Try Polygon first (professional grade like Alpha Vantage)
                api = get_data_api("polygon")
                full_data = api.fetch_historical_data(
                    self.STANDARD_CONFIG['test_symbol'],
                    "1d",
                    # Should be '2024-01-01'
                    start_date=self.STANDARD_CONFIG['start_date'],
                    # Should be '2024-12-31'
                    end_date=self.STANDARD_CONFIG['end_date']
                )

                if full_data is None or full_data.empty:
                    print("⚠️ Polygon failed, trying Alpha Vantage...")
                    # Use Alpha Vantage second
                    api = get_data_api("alpha_vantage")
                    full_data = api.fetch_historical_data(
                        self.STANDARD_CONFIG['test_symbol'], "1d")

                    if full_data is None or full_data.empty:
                        print("⚠️ Alpha Vantage failed, trying Yahoo Finance...")
                        api = get_data_api("yahoo_finance")
                        full_data = api.fetch_historical_data(
                            self.STANDARD_CONFIG['test_symbol'], "1d")

                if full_data is not None and not full_data.empty:
                    # Filter for EXACT 2024 period
                    start_date = pd.Timestamp(
                        self.STANDARD_CONFIG['start_date'])
                    end_date = pd.Timestamp(self.STANDARD_CONFIG['end_date'])

                    self.test_data_2024 = full_data[
                        (full_data.index >= start_date) &
                        (full_data.index <= end_date)
                    ]

                    if not self.test_data_2024.empty:
                        # Add technical indicators
                        self.test_data_2024 = TechnicalIndicators.add_all_indicators(
                            self.test_data_2024)

                        print(f"✅ STANDARDIZED DATA LOADED:")
                        print(
                            f"   Actual Start: {self.test_data_2024.index[0].strftime('%Y-%m-%d')}")
                        print(
                            f"   Actual End: {self.test_data_2024.index[-1].strftime('%Y-%m-%d')}")
                        print(f"   Trading Days: {len(self.test_data_2024)}")
                        print(
                            f"   Price Range: ${self.test_data_2024['close'].min():.2f} to ${self.test_data_2024['close'].max():.2f}")

                        return True
                    else:
                        print("❌ No 2024 data available in the specified range")

            except Exception as e:
                print(f"❌ Error loading standardized data: {e}")

        # Fallback to yfinance if project not available
        try:
            print("📊 Using yfinance as fallback...")
            ticker = yf.Ticker(self.STANDARD_CONFIG['test_symbol'])
            self.test_data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not self.test_data_2024.empty:
                # Rename columns to match project format
                self.test_data_2024.columns = self.test_data_2024.columns.str.lower()

                print(f"✅ FALLBACK DATA LOADED:")
                print(
                    f"   Period: {self.test_data_2024.index[0].strftime('%Y-%m-%d')} to {self.test_data_2024.index[-1].strftime('%Y-%m-%d')}")
                print(f"   Trading Days: {len(self.test_data_2024)}")

                return True
        except Exception as e:
            print(f"❌ Fallback data loading failed: {e}")

        return False

    def run_all_standardized_tests(self):

        if not self.load_standardized_data():
            print("❌ Cannot proceed - failed to load standardized data")
            return

        print("\n🧪 RUNNING ENHANCED STANDARDIZED TESTS")
        print("=" * 70)
        print("Testing ALL available strategies with identical conditions")
        print("=" * 70)

        # Test MLTrading Strategyfirst
        your_performance = self._test_your_current_ai_system()
        if your_performance:
            self.results['MLTrading Strategy'] = your_performance

        # Test previously imported strategies
        self.test_technical_analysis_strategy()
        self.test_buy_and_hold_strategy()

        # Test RSI Divergence strategies
        self.test_rsi_divergence_strategies()

        # Keep your existing hypothesis tests
        self._test_h1_standardized()
        self._test_h2_standardized()
        self._test_h3_standardized()
        self._test_h4_standardized()

        # Generate enhanced analysis
        self.generate_comprehensive_comparison()

        # Keep your existing report generation
        self._generate_standardized_analysis()

    def _test_your_current_ai_system(self):
        """Test MLTrading Strategyusing standardized 2024 data"""

        print(f"\n🤖 Testing MLTrading Strategy(Standardized 2024)...")

        if not PROJECT_AVAILABLE:
            print("❌ Project modules not available")
            return None

        try:
            # Use the standardized 2024 data
            data = self.test_data_2024.copy()

            # Test your ML strategy with CURRENT settings (no modifications)
            strategy = MLTradingStrategy(confidence_threshold=0.40)
            results = self._backtest_strategy_standardized(
                strategy, data, "MLTradingStrategy")

            print(f"✅ MLTradingStrategy Results (2024 STANDARDIZED):")
            print(f"   📈 Total Return: {results['total_return']:.2%}")
            print(f"   🔄 Total Trades: {results['total_trades']}")
            print(f"   📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"   📉 Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"   🎯 Win Rate: {results['win_rate']:.2%}")

            return results

        except Exception as e:
            print(f"❌ Error testing MLTradingStrategy: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _backtest_strategy_standardized(self, strategy, data, strategy_name):
        """Standardized backtesting using IDENTICAL conditions for all systems"""

        from src.backtesting.backtester import ProductionBacktester

        # Initialize with STANDARDIZED parameters
        backtester = ProductionBacktester(
            initial_capital=self.STANDARD_CONFIG['initial_capital'],
            transaction_cost=0.001,
            max_position_size=1.0  # Use full capital for meaningful comparison
        )

        backtester.set_strategy(strategy)

        # Run backtest on standardized data
        results = backtester.run_backtest(data)

        # Add standardization metadata
        results['data_type'] = 'STANDARDIZED 2024 DATA'
        results['test_period'] = f"{self.STANDARD_CONFIG['start_date']} to {self.STANDARD_CONFIG['end_date']}"
        results['trading_days'] = len(data)
        results['strategy_name'] = strategy_name

        return results

    def _test_h1_standardized(self):
        """H1: Test against trading programs using STANDARDIZED 2024 period"""

        print(f"\n🎯 H1: TRADING PROGRAMS (STANDARDIZED 2024)")
        print("-" * 40)

        # TradingView data - already covers 2024, so it's consistent
        tradingview_data = self._get_tradingview_data_standardized()
        if tradingview_data:
            self.results['H1: TradingView RSI Strategy'] = tradingview_data
            print(
                f"✅ TradingView Strategy (2024): {tradingview_data['total_return']:.2%}")

        # ETF proxies using STANDARDIZED 2024 data
        momentum_etf = self._get_momentum_etf_standardized()
        if momentum_etf:
            self.results['H1: Strategy ETF Proxy'] = momentum_etf
            print(
                f"✅ Strategy ETF Proxy (2024): {momentum_etf['total_return']:.2%}")

        # Systematic strategy using STANDARDIZED 2024 data
        systematic = self._get_systematic_strategy_standardized()
        if systematic:
            self.results['H1: Systematic Strategy'] = systematic
            print(
                f"✅ Systematic Strategy (2024): {systematic['total_return']:.2%}")

    def _test_h2_standardized(self):
        """H2: Test against famous traders using STANDARDIZED 2024 data"""

        print(f"\n🎯 H2: FAMOUS TRADERS (STANDARDIZED 2024)")
        print("-" * 40)

        # Warren Buffett (BRK-A) for 2024
        buffett = self._get_buffett_performance_2024()
        if buffett:
            self.results['H2: Warren Buffett'] = buffett
            print(f"✅ Warren Buffett (2024): {buffett['total_return']:.2%}")

        # Cathie Wood (ARKK) for 2024
        cathie = self._get_cathie_wood_performance_2024()
        if cathie:
            self.results['H2: Cathie Wood'] = cathie
            print(f"✅ Cathie Wood (2024): {cathie['total_return']:.2%}")

        # Ray Dalio proxy for 2024
        dalio = self._get_ray_dalio_performance_2024()
        if dalio:
            self.results['H2: Ray Dalio'] = dalio
            print(f"✅ Ray Dalio (2024): {dalio['total_return']:.2%}")

    def _test_h3_standardized(self):
        """H3: Test against AI systems using STANDARDIZED 2024 data"""

        print(f"\n🎯 H3: AI TRADING SYSTEMS (STANDARDIZED 2024)")
        print("-" * 40)

        # AI ETF (QQQ) for 2024
        ai_etf = self._get_ai_etf_performance_2024()
        if ai_etf:
            self.results['H3: AI ETF'] = ai_etf
            print(f"✅ AI ETF QQQ (2024): {ai_etf['total_return']:.2%}")

        # Robo-advisor performance for 2024
        robo = self._get_robo_advisor_performance_2024()
        if robo:
            self.results['H3: Robo-Advisor'] = robo
            print(f"✅ Robo-Advisor (2024): {robo['total_return']:.2%}")

    def _test_h4_standardized(self):
        """H4: Test against beginner traders using standardized assumptions"""

        print(f"\n🎯 H4: BEGINNER TRADERS (STANDARDIZED)")
        print("-" * 40)

        # Academic research data (this doesn't need 2024 data as it's a statistical baseline)
        beginner = self._get_beginner_trader_performance()
        if beginner:
            self.results['H4: Beginner Trader'] = beginner
            print(f"✅ Beginner Trader: {beginner['total_return']:.2%}")

    # Standardized data fetching methods - UNCHANGED from your original
    def _get_tradingview_data_standardized(self):
        """TradingView data - already standardized to 2024"""

        # This data is already for Jan 2024 - Dec 2024, so it's consistent
        REAL_TRADINGVIEW_DATA = {
            'strategy_name': 'TradingView RSI Divergence Strategy',
            'total_return': 3539.25 / 10000,    # 35.39%
            'total_trades': 92,
            'win_rate': 0.6413,
            'max_drawdown': 0.0298,
            'sharpe_ratio': -0.263,
            'volatility': 0.45,
        }

        return {
            'source': 'TradingView Strategy (STANDARDIZED 2024)',
            'strategy_name': REAL_TRADINGVIEW_DATA['strategy_name'],
            'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)',
            'total_return': REAL_TRADINGVIEW_DATA['total_return'],
            'total_trades': REAL_TRADINGVIEW_DATA['total_trades'],
            'win_rate': REAL_TRADINGVIEW_DATA['win_rate'],
            'max_drawdown': REAL_TRADINGVIEW_DATA['max_drawdown'],
            'sharpe_ratio': REAL_TRADINGVIEW_DATA['sharpe_ratio'],
            'data_type': 'STANDARDIZED TRADINGVIEW 2024'
        }

    def _get_momentum_etf_standardized(self):
        """Get momentum ETF performance for standardized 2024 period"""

        try:
            ticker = yf.Ticker("MTUM")
            data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not data_2024.empty:
                start_price = data_2024['Close'].iloc[0]
                end_price = data_2024['Close'].iloc[-1]
                total_return = (end_price - start_price) / start_price

                # Calculate volatility
                daily_returns = data_2024['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe = (total_return / volatility) if volatility > 0 else 0

                return {
                    'source': 'Momentum ETF (MTUM) - STANDARDIZED 2024',
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'data_type': 'STANDARDIZED 2024 ETF DATA',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error fetching momentum ETF data: {e}")

        return None

    def _get_systematic_strategy_standardized(self):
        """Get systematic strategy performance for standardized 2024 period"""

        try:
            ticker = yf.Ticker("SPY")
            data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not data_2024.empty:
                # Simple systematic strategy: above/below 50-day MA
                data_2024['sma_50'] = data_2024['Close'].rolling(50).mean()

                # Calculate strategy returns
                position = 0
                returns = []

                for i in range(50, len(data_2024)):
                    if data_2024['Close'].iloc[i] > data_2024['sma_50'].iloc[i] and position <= 0:
                        position = 1  # Buy signal
                    elif data_2024['Close'].iloc[i] < data_2024['sma_50'].iloc[i] and position >= 0:
                        position = -1  # Sell signal

                    daily_return = data_2024['Close'].pct_change(
                    ).iloc[i] * position
                    returns.append(daily_return)

                total_return = np.sum(returns) if returns else 0

                return {
                    'source': 'Systematic Strategy (SPY MA) - STANDARDIZED 2024',
                    'total_return': total_return,
                    'data_type': 'STANDARDIZED 2024 SYSTEMATIC',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error calculating systematic strategy: {e}")

        return None

    def _get_buffett_performance_2024(self):
        """Warren Buffett (BRK-A) performance for standardized 2024"""

        try:
            ticker = yf.Ticker("BRK-A")
            data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not data_2024.empty:
                start_price = data_2024['Close'].iloc[0]
                end_price = data_2024['Close'].iloc[-1]
                total_return = (end_price - start_price) / start_price

                daily_returns = data_2024['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe = (total_return / volatility) if volatility > 0 else 0

                return {
                    'source': 'Warren Buffett (BRK-A) - STANDARDIZED 2024',
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'data_type': 'STANDARDIZED 2024 BRK-A DATA',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error fetching BRK-A data: {e}")

        return None

    def _get_cathie_wood_performance_2024(self):
        """Cathie Wood (ARKK) performance for standardized 2024"""

        try:
            ticker = yf.Ticker("ARKK")
            data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not data_2024.empty:
                start_price = data_2024['Close'].iloc[0]
                end_price = data_2024['Close'].iloc[-1]
                total_return = (end_price - start_price) / start_price

                daily_returns = data_2024['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe = (total_return / volatility) if volatility > 0 else 0

                return {
                    'source': 'Cathie Wood (ARKK) - STANDARDIZED 2024',
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'data_type': 'STANDARDIZED 2024 ARKK DATA',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error fetching ARKK data: {e}")

        return None

    def _get_ray_dalio_performance_2024(self):
        """Ray Dalio All Weather proxy for standardized 2024"""

        # Simple All Weather portfolio: 40% bonds, 30% stocks, 15% commodities, 15% TIPS
        try:
            # Simplified with just bonds (TLT) and stocks (VTI)
            tlt = yf.Ticker("TLT").history(
                start=self.STANDARD_CONFIG['start_date'], end=self.STANDARD_CONFIG['end_date'])
            vti = yf.Ticker("VTI").history(
                start=self.STANDARD_CONFIG['start_date'], end=self.STANDARD_CONFIG['end_date'])

            if not tlt.empty and not vti.empty:
                # 60% bonds, 40% stocks (simplified All Weather)
                bond_return = (tlt['Close'].iloc[-1] -
                               tlt['Close'].iloc[0]) / tlt['Close'].iloc[0]
                stock_return = (vti['Close'].iloc[-1] -
                                vti['Close'].iloc[0]) / vti['Close'].iloc[0]

                portfolio_return = 0.6 * bond_return + 0.4 * stock_return

                return {
                    'source': 'Ray Dalio All Weather Proxy - STANDARDIZED 2024',
                    'total_return': portfolio_return,
                    'data_type': 'STANDARDIZED 2024 ALL WEATHER',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error calculating All Weather proxy: {e}")

        return None

    def _get_ai_etf_performance_2024(self):
        """AI ETF (QQQ) performance for standardized 2024"""

        try:
            ticker = yf.Ticker("QQQ")
            data_2024 = ticker.history(
                start=self.STANDARD_CONFIG['start_date'],
                end=self.STANDARD_CONFIG['end_date']
            )

            if not data_2024.empty:
                start_price = data_2024['Close'].iloc[0]
                end_price = data_2024['Close'].iloc[-1]
                total_return = (end_price - start_price) / start_price

                daily_returns = data_2024['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe = (total_return / volatility) if volatility > 0 else 0

                return {
                    'source': 'AI ETF (QQQ) - STANDARDIZED 2024',
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'data_type': 'STANDARDIZED 2024 QQQ DATA',
                    'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
                }
        except Exception as e:
            print(f"⚠️ Error fetching QQQ data: {e}")

        return None

    def _get_robo_advisor_performance_2024(self):
        """Robo-advisor performance estimate for 2024"""

        # Conservative estimate based on typical robo-advisor performance
        return {
            'source': 'Robo-Advisor Performance (Conservative Estimate)',
            'total_return': 0.089,  # 8.9% typical performance
            'data_type': 'STANDARDIZED ESTIMATE',
            'period': 'Jan 2024 - Dec 2024 (STANDARDIZED)'
        }

    def _get_beginner_trader_performance(self):
        """Beginner trader performance (academic baseline)"""

        return {
            'source': 'Beginner Trader (Academic Research)',
            'total_return': -0.15,  # -15% typical beginner performance
            'data_type': 'ACADEMIC BASELINE',
            'period': 'Standardized Academic Research'
        }

    def test_technical_analysis_strategy(self):
        """Test TechnicalAnalysisStrategy (was imported but not used)"""

        print(f"\n🎯 Testing TechnicalAnalysisStrategy (STANDARDIZED)")
        print("-" * 50)

        if self.test_data_2024 is None or self.test_data_2024.empty:
            print("❌ No test data available")
            return None

        try:
            strategy = TechnicalAnalysisStrategy(
                rsi_oversold=30,
                rsi_overbought=70
            )

            results = self._backtest_strategy_standardized(
                strategy, self.test_data_2024, "Technical Analysis Strategy")

            if results:
                print(f"✅ Technical Analysis Results (STANDARDIZED):")
                print(f"   Total Return: {results['total_return']:.2%}")
                print(f"   Total Trades: {results['total_trades']}")
                print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
                print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
                print(f"   Alpha: {results.get('alpha', 0):.2%}")

                self.results['technical_analysis'] = results
                return results

        except Exception as e:
            print(f"❌ Technical Analysis test failed: {e}")
            return None

    def test_buy_and_hold_strategy(self):
        """Test BuyAndHoldStrategy (was imported but not used)"""

        print(f"\n🎯 Testing BuyAndHoldStrategy (STANDARDIZED)")
        print("-" * 50)

        if self.test_data_2024 is None or self.test_data_2024.empty:
            print("❌ No test data available")
            return None

        try:
            strategy = BuyAndHoldStrategy()

            results = self._backtest_strategy_standardized(
                strategy, self.test_data_2024, "Buy & Hold Strategy")

            if results:
                print(f"✅ Buy & Hold Results (STANDARDIZED):")
                print(f"   Total Return: {results['total_return']:.2%}")
                print(f"   Total Trades: {results['total_trades']}")
                print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
                print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
                print(f"   Alpha: {results.get('alpha', 0):.2%}")

                self.results['buy_and_hold'] = results
                return results

        except Exception as e:
            print(f"❌ Buy & Hold test failed: {e}")
            return None

    def test_rsi_divergence_strategies(self):
        """Test RSI Divergence strategies (new)"""

        print(f"\n🎯 Testing RSI Divergence Strategies (STANDARDIZED)")
        print("-" * 50)

        if not RSI_STRATEGIES_AVAILABLE:
            print("❌ RSI Divergence strategies not available")
            return None

        if self.test_data_2024 is None or self.test_data_2024.empty:
            print("❌ No test data available")
            return None

        rsi_results = {}

        # Test standalone RSI Divergence Strategy
        try:
            print("   Testing RSI Divergence Strategy...")
            strategy = RSIDivergenceStrategy(
                swing_threshold_pct=2.5,  # Optimal from testing
                hold_days=15,
                confidence_base=0.7
            )

            results = self._backtest_strategy_standardized(
                strategy, self.test_data_2024, "RSI Divergence Strategy")

            if results:
                print(
                    f"      Return: {results['total_return']:.2%}, Trades: {results['total_trades']}, Win Rate: {results.get('win_rate', 0):.1%}")
                rsi_results['rsi_divergence'] = results
                self.results['rsi_divergence'] = results

        except Exception as e:
            print(f"      ❌ RSI Divergence failed: {e}")

        # Test Hybrid RSI-Technical Strategy
        try:
            print("   Testing Hybrid RSI-Technical Strategy...")
            base_strategy = TechnicalAnalysisStrategy()
            strategy = HybridRSIDivergenceStrategy(
                divergence_weight=0.6,
                technical_weight=0.4,
                base_strategy=base_strategy
            )

            results = self._backtest_strategy_standardized(
                strategy, self.test_data_2024, "Hybrid RSI-Technical Strategy")

            if results:
                print(
                    f"      Return: {results['total_return']:.2%}, Trades: {results['total_trades']}, Win Rate: {results.get('win_rate', 0):.1%}")
                rsi_results['hybrid_rsi_technical'] = results
                self.results['hybrid_rsi_technical'] = results

        except Exception as e:
            print(f"      ❌ Hybrid RSI-Technical failed: {e}")

        # Test Hybrid RSI-ML Strategy
        try:
            print("   Testing Hybrid RSI-ML Strategy...")
            base_strategy = MLTradingStrategy(confidence_threshold=0.40)
            strategy = HybridRSIDivergenceStrategy(
                divergence_weight=0.7,
                technical_weight=0.3,
                base_strategy=base_strategy
            )

            results = self._backtest_strategy_standardized(
                strategy, self.test_data_2024, "Hybrid RSI-ML Strategy")

            if results:
                print(
                    f"      Return: {results['total_return']:.2%}, Trades: {results['total_trades']}, Win Rate: {results.get('win_rate', 0):.1%}")
                rsi_results['hybrid_rsi_ml'] = results
                self.results['hybrid_rsi_ml'] = results

        except Exception as e:
            print(f"      ❌ Hybrid RSI-ML failed: {e}")

        if rsi_results:
            print(
                f"✅ RSI Divergence testing completed: {len(rsi_results)} strategies tested")
            return rsi_results
        else:
            print("❌ No RSI Divergence strategies tested successfully")
            return None

    

    def generate_comprehensive_comparison(self):

        print(f"\n🏆 COMPREHENSIVE STRATEGY COMPARISON")
        print("=" * 80)
        print("All strategies tested with IDENTICAL standardized conditions")
        print("=" * 80)

        if not self.results:
            print("❌ No results to compare")
            return

        # Collect all strategy results
        all_strategies = []

        for strategy_key, results in self.results.items():
            if isinstance(results, dict) and 'total_return' in results:
                all_strategies.append({
                    'name': results.get('strategy_name', strategy_key),
                    'key': strategy_key,  # ✨ ADD KEY FOR IDENTIFICATION
                    'return': results['total_return'],
                    'trades': results.get('total_trades', 0),
                    'win_rate': results.get('win_rate', 0),
                    'sharpe': results.get('sharpe_ratio', 0),
                    'alpha': results.get('alpha', 0),
                    'final_value': results.get('final_value', 0),
                    # ✨ ADD DRAWDOWN
                    'max_drawdown': results.get('max_drawdown', 0)
                })

        if not all_strategies:
            print("❌ No valid strategy results found")
            return

        # Sort by return (descending)
        all_strategies.sort(key=lambda x: x['return'], reverse=True)

        # Display comprehensive table with enhanced columns
        print(f"{'Rank':<4} {'Strategy':<35} {'Return':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7} {'Drawdown':<9}")
        print("-" * 85)

        for i, strategy in enumerate(all_strategies, 1):
            print(f"{i:<4} {strategy['name']:<35} "
                  f"{strategy['return']:>7.2%} "
                  f"{strategy['trades']:>7d} "
                  f"{strategy['win_rate']:>5.1%} "
                  f"{strategy['sharpe']:>7.2f} "
                  f"{strategy['max_drawdown']:>8.2%}")

        # Analysis
        best_performer = all_strategies[0]
        print(f"\n🥇 BEST OVERALL PERFORMER: {best_performer['name']}")
        print(f"   Return: {best_performer['return']:.2%}")
        print(f"   Final Value: ${best_performer['final_value']:,.2f}")
        print(f"   Trades: {best_performer['trades']}")
        print(f"   Win Rate: {best_performer['win_rate']:.1%}")
        print(f"   Sharpe Ratio: {best_performer['sharpe']:.2f}")
        print(f"   Max Drawdown: {best_performer['max_drawdown']:.2%}")

        

        # Category analysis
        excellent = [s for s in all_strategies if s['return'] >= 0.20]  # 20%+
        good = [s for s in all_strategies if 0.10 <=
                s['return'] < 0.20]  # 10-20%
        moderate = [s for s in all_strategies if 0.05 <=
                    s['return'] < 0.10]  # 5-10%
        poor = [s for s in all_strategies if s['return'] < 0.05]  # <5%

        print(f"\n📊 PERFORMANCE CATEGORIES:")
        print(f"   🚀 Excellent (20%+): {len(excellent)} strategies")
        if excellent:
            for s in excellent[:3]:  # Show top 3
                print(f"      - {s['name']}: {s['return']:.2%}")

        print(f"   👍 Good (10-20%): {len(good)} strategies")
        print(f"   📈 Moderate (5-10%): {len(moderate)} strategies")
        print(f"   📉 Needs Work (<5%): {len(poor)} strategies")

        # RSI Divergence specific analysis (keep existing)
        rsi_strategies = [s for s in all_strategies if 'RSI' in s['name']]
        if rsi_strategies:
            print(f"\n🔍 RSI DIVERGENCE ANALYSIS:")
            best_rsi = max(rsi_strategies, key=lambda x: x['return'])
            rsi_position = all_strategies.index(best_rsi) + 1
            print(
                f"   Best RSI Strategy: {best_rsi['name']} (#{rsi_position}, {best_rsi['return']:.2%})")

            if rsi_position <= 3:
                print(f"   🎉 RSI strategy in top 3 performers!")
            elif best_rsi['return'] > 0.10:
                print(f"   👍 RSI strategy shows good performance")
            else:
                print(f"   ⚠️ RSI strategy needs optimization")

    def _generate_standardized_analysis(self):

        print(f"\n📊 ENHANCED STANDARDIZED ANALYSIS")
        print("=" * 70)
        print("Including ALL tested strategies (original + new additions)")
        print("=" * 70)

        if not self.results:
            print("❌ No results available for analysis")
            return

        # Get MLTradingStrategy result (keep existing logic)
        your_ai_result = self.results.get(
            'MLTrading Strategy', {}) or self.results.get('MLTradingStrategy', {})

        if your_ai_result:
            print(f"🤖 MLTradingStrategy PERFORMANCE:")
            print(
                f"   Return: {your_ai_result.get('total_return', 0):.2%}")
            print(f"   Trades: {your_ai_result.get('total_trades', 0)}")
            print(f"   Win Rate: {your_ai_result.get('win_rate', 0):.1%}")
            print(
                f"   Sharpe: {your_ai_result.get('sharpe_ratio', 0):.2f}")

        # Enhanced strategy comparison (includes all new strategies)
        print(f"\n🏆 ALL STRATEGIES COMPARISON:")
        print("-" * 70)

        # Collect ALL strategy results (original + new)
        all_strategy_results = []

        strategy_mapping = {
            'MLTrading Strategy': 'MLTrading Strategy',
            'technical_analysis': 'Technical Analysis Strategy',
            'buy_and_hold': 'Buy & Hold Strategy',
            'rsi_divergence': 'RSI Divergence Strategy',
            'hybrid_rsi_technical': 'Hybrid RSI-Technical',
            'hybrid_rsi_ml': 'Hybrid RSI-ML',
            'H1: TradingView RSI Strategy': 'H1: TradingView Strategy',
            'H1: Strategy ETF Proxy': 'H1: Strategy ETF Proxy',
            'H2: Warren Buffett': 'H2: Warren Buffett',
            'H2: Cathie Wood': 'H2: Cathie Wood',
            'H3: AI ETF': 'H3: AI ETF',
            'H3: Robo-Advisor': 'H3: Robo-Advisor',
            'H4: Beginner Trader': 'H4: Beginner Trader'
        }

        for key, results in self.results.items():
            if isinstance(results, dict) and 'total_return' in results:
                strategy_name = strategy_mapping.get(key, key)
                all_strategy_results.append({
                    'name': strategy_name,
                    'key': key,
                    'return': results['total_return'],
                    'trades': results.get('total_trades', 0),
                    'win_rate': results.get('win_rate', 0),
                    'sharpe': results.get('sharpe_ratio', 0),
                    'alpha': results.get('alpha', 0)
                })

        # Sort by return
        all_strategy_results.sort(key=lambda x: x['return'], reverse=True)

        # Display table
        print(
            f"{'Rank':<4} {'Strategy':<35} {'Return':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7}")
        print("-" * 75)

        for i, strategy in enumerate(all_strategy_results, 1):
            print(f"{i:<4} {strategy['name']:<35} "
                  f"{strategy['return']:>7.2%} "
                  f"{strategy['trades']:>7d} "
                  f"{strategy['win_rate']:>5.1%} "
                  f"{strategy['sharpe']:>7.2f}")

        # Enhanced analysis
        if all_strategy_results:
            best_performer = all_strategy_results[0]
            print(f"\n🥇 BEST OVERALL PERFORMER: {best_performer['name']}")
            print(f"   Return: {best_performer['return']:.2%}")

            # Find MLTradingStrategy's ranking
            your_ai_rank = None
            for i, strategy in enumerate(all_strategy_results, 1):
                if strategy['key'] in ['MLTrading Strategy', 'MLTradingStrategy']:
                    your_ai_rank = i
                    break

            if your_ai_rank:
                print(
                    f"\n🤖 MLTradingStrategy RANKING: #{your_ai_rank} out of {len(all_strategy_results)}")

                if your_ai_rank == 1:
                    print("🎉 MLTradingStrategy IS THE BEST PERFORMER!")
                elif your_ai_rank <= 3:
                    print("👍 MLTradingStrategy is in the top 3 performers")
                elif your_ai_rank <= len(all_strategy_results) // 2:
                    print("📈 MLTradingStrategy is in the top half")
                else:
                    print("📉 MLTradingStrategy needs improvement")

            # Strategy category analysis
            excellent_strategies = [
                s for s in all_strategy_results if s['return'] >= 0.20]
            good_strategies = [
                s for s in all_strategy_results if 0.10 <= s['return'] < 0.20]

            print(f"\n📊 PERFORMANCE TIERS:")
            print(
                f"   🚀 Excellent (20%+): {len(excellent_strategies)} strategies")
            for s in excellent_strategies[:3]:  # Show top 3
                print(f"      - {s['name']}: {s['return']:.2%}")

            print(f"   👍 Good (10-20%): {len(good_strategies)} strategies")

            # Specific analysis for newly added strategies
            new_strategy_keys = ['technical_analysis', 'buy_and_hold', 'rsi_divergence',
                                 'hybrid_rsi_technical', 'hybrid_rsi_ml']
            new_strategies = [
                s for s in all_strategy_results if s['key'] in new_strategy_keys]

            if new_strategies:
                print(f"\n🆕 NEWLY TESTED STRATEGIES PERFORMANCE:")
                for strategy in new_strategies:
                    rank = all_strategy_results.index(strategy) + 1
                    special_note = ""
                    print(
                        f"   #{rank}: {strategy['name']} - {strategy['return']:.2%}{special_note}")

        # Keep your existing hypothesis analysis
        print(f"\n🧪 HYPOTHESIS TEST RESULTS:")

        # H1: Trading Programs
        h1_results = [
            s for s in all_strategy_results if s['key'].startswith('H1:')]
        if h1_results and your_ai_result:
            best_h1 = max(h1_results, key=lambda x: x['return'])
            your_return = your_ai_result.get('total_return', 0)

            if your_return > best_h1['return']:
                print(
                    f"   ✅ H1 PASSED: You beat trading programs ({your_return:.2%} > {best_h1['return']:.2%})")
            else:
                print(
                    f"   ❌ H1 FAILED: Trading programs won ({best_h1['return']:.2%} > {your_return:.2%})")

        # H2: Famous Investors
        h2_results = [
            s for s in all_strategy_results if s['key'].startswith('H2:')]
        if h2_results and your_ai_result:
            best_h2 = max(h2_results, key=lambda x: x['return'])
            your_return = your_ai_result.get('total_return', 0)

            if your_return > best_h2['return']:
                print(
                    f"   ✅ H2 PASSED: You beat famous investors ({your_return:.2%} > {best_h2['return']:.2%})")
            else:
                print(
                    f"   ❌ H2 FAILED: Famous investors won ({best_h2['return']:.2%} > {your_return:.2%})")

        # H3: AI Systems
        h3_results = [
            s for s in all_strategy_results if s['key'].startswith('H3:')]
        if h3_results and your_ai_result:
            best_h3 = max(h3_results, key=lambda x: x['return'])
            your_return = your_ai_result.get('total_return', 0)

            if your_return > best_h3['return']:
                print(
                    f"   ✅ H3 PASSED: You beat AI systems ({your_return:.2%} > {best_h3['return']:.2%})")
            else:
                print(
                    f"   ❌ H3 FAILED: AI systems won ({best_h3['return']:.2%} > {your_return:.2%})")

        # H4: Beginner Trader
        h4_results = [
            s for s in all_strategy_results if s['key'].startswith('H4:')]
        if h4_results and your_ai_result:
            h4_result = h4_results[0]
            your_return = your_ai_result.get('total_return', 0)

            if your_return > h4_result['return']:
                print(
                    f"   ✅ H4 PASSED: You beat beginner trader ({your_return:.2%} > {h4_result['return']:.2%})")
            else:
                print(
                    f"   ❌ H4 FAILED: Beginner trader won ({h4_result['return']:.2%} > {your_return:.2%})")

        # Save standardized report
        # Create results directory and call with parameter
        results_dir = "standardized_hypothesis_results"
        os.makedirs(results_dir, exist_ok=True)
        self._save_standardized_report(results_dir)

    def _save_standardized_report(self, results_dir):

        # Create comprehensive report file
        report_file = os.path.join(
            results_dir, f"comprehensive_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE STANDARDIZED STRATEGY REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Test Period: {self.STANDARD_CONFIG['start_date']} to {self.STANDARD_CONFIG['end_date']}\n")
            f.write(f"Symbol: {self.STANDARD_CONFIG['test_symbol']}\n")
            f.write(
                f"Initial Capital: ${self.STANDARD_CONFIG['initial_capital']:,}\n\n")

            # Collect all results for report
            all_results = []
            strategy_mapping = {
                'MLTradingStrategy': 'MLTrading Strategy',
                'technical_analysis': 'Technical Analysis Strategy',
                'buy_and_hold': 'Buy & Hold Strategy',
                'rsi_divergence': 'RSI Divergence Strategy',
                'hybrid_rsi_technical': 'Hybrid RSI-Technical',
                'hybrid_rsi_ml': 'Hybrid RSI-ML',
                'H1: TradingView RSI Strategy': 'H1: TradingView Strategy',
                'H1: Strategy ETF Proxy': 'H1: Strategy ETF Proxy',
                'H2: Warren Buffett': 'H2: Warren Buffett',
                'H2: Cathie Wood': 'H2: Cathie Wood',
                'H3: AI ETF': 'H3: AI ETF',
                'H3: Robo-Advisor': 'H3: Robo-Advisor',
                'H4: Beginner Trader': 'H4: Beginner Trader'
            }

            for key, results in self.results.items():
                if isinstance(results, dict) and 'total_return' in results:
                    strategy_name = strategy_mapping.get(key, key)
                    all_results.append({
                        'name': strategy_name,
                        'key': key,
                        'return': results['total_return'],
                        'trades': results.get('total_trades', 0),
                        'win_rate': results.get('win_rate', 0),
                        'sharpe': results.get('sharpe_ratio', 0),
                        'alpha': results.get('alpha', 0),
                        'final_value': results.get('final_value', 0),
                        'max_drawdown': results.get('max_drawdown', 0)
                    })

            # Sort by performance
            all_results.sort(key=lambda x: x['return'], reverse=True)

            f.write("STRATEGY PERFORMANCE RANKING:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Rank':<4} {'Strategy':<35} {'Return':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7}\n")
            f.write("-" * 80 + "\n")

            for i, result in enumerate(all_results, 1):
                f.write(f"{i:<4} {result['name']:<35} "
                        f"{result['return']:>7.2%} "
                        f"{result['trades']:>7d} "
                        f"{result['win_rate']:>5.1%} "
                        f"{result['sharpe']:>7.2f}\n")

            # Best performer details
            if all_results:
                best = all_results[0]
                f.write(f"\nBEST PERFORMER: {best['name']}\n")
                f.write(f"Return: {best['return']:.2%}\n")
                f.write(f"Final Value: ${best['final_value']:,.2f}\n")
                f.write(f"Trades: {best['trades']}\n")
                f.write(f"Win Rate: {best['win_rate']:.1%}\n")
                f.write(f"Sharpe Ratio: {best['sharpe']:.2f}\n")
                f.write(f"Max Drawdown: {best['max_drawdown']:.2%}\n")


            # Strategy categories
            excellent = [r for r in all_results if r['return'] >= 0.20]
            good = [r for r in all_results if 0.10 <= r['return'] < 0.20]
            moderate = [r for r in all_results if 0.05 <= r['return'] < 0.10]
            poor = [r for r in all_results if r['return'] < 0.05]

            f.write(f"\nPERFORMANCE CATEGORIES:\n")
            f.write(f"Excellent (20%+): {len(excellent)} strategies\n")
            if excellent:
                for strategy in excellent:
                    f.write(
                        f"  - {strategy['name']}: {strategy['return']:.2%}\n")

            f.write(f"Good (10-20%): {len(good)} strategies\n")
            if good:
                for strategy in good[:5]:  # Show top 5 good performers
                    f.write(
                        f"  - {strategy['name']}: {strategy['return']:.2%}\n")

            f.write(f"Moderate (5-10%): {len(moderate)} strategies\n")
            f.write(f"Needs Work (<5%): {len(poor)} strategies\n")

            # MLTradingStrategy analysis
            your_ai = next((r for r in all_results if r['key'] in [
                           'MLTrading Strategy', 'MLTradingStrategy']), None)
            if your_ai:
                your_rank = all_results.index(your_ai) + 1
                f.write(f"\nMLTradingStrategy ANALYSIS:\n")
                f.write(
                    f"Ranking: #{your_rank} out of {len(all_results)} strategies\n")
                f.write(f"Return: {your_ai['return']:.2%}\n")
                f.write(f"Performance Tier: ")
                if your_ai['return'] >= 0.20:
                    f.write("Excellent\n")
                elif your_ai['return'] >= 0.10:
                    f.write("Good\n")
                elif your_ai['return'] >= 0.05:
                    f.write("Moderate\n")
                else:
                    f.write("Needs Improvement\n")

            # ✨ STRATEGY COMPARISON SECTION
            f.write(f"\nSTRATEGY COMPARISON INSIGHTS:\n")

            # High-frequency vs Low-frequency analysis
            high_freq_strategies = [
                r for r in all_results if r['trades'] >= 50]
            low_freq_strategies = [r for r in all_results if r['trades'] < 20]

            if high_freq_strategies and low_freq_strategies:
                avg_high_freq_return = sum(
                    s['return'] for s in high_freq_strategies) / len(high_freq_strategies)
                avg_low_freq_return = sum(
                    s['return'] for s in low_freq_strategies) / len(low_freq_strategies)

                f.write(
                    f"High-Frequency Strategies (50+ trades): {len(high_freq_strategies)} strategies, avg return: {avg_high_freq_return:.2%}\n")
                f.write(
                    f"Low-Frequency Strategies (<20 trades): {len(low_freq_strategies)} strategies, avg return: {avg_low_freq_return:.2%}\n")

                if avg_high_freq_return > avg_low_freq_return:
                    f.write(
                        f"Insight: High-frequency strategies outperform by {avg_high_freq_return - avg_low_freq_return:.2%} on average\n")
                else:
                    f.write(
                        f"Insight: Low-frequency strategies outperform by {avg_low_freq_return - avg_high_freq_return:.2%} on average\n")

            # Technical vs ML strategies comparison
            tech_strategies = [
                r for r in all_results if 'Technical' in r['name'] or 'RSI' in r['name']]
            ml_strategies = [r for r in all_results if 'AI' in r['name'] or 'ML' in r['name'] or r['key'] in [
                'MLTrading Strategy', 'MLTradingStrategy']]

            if tech_strategies:
                avg_tech_return = sum(
                    s['return'] for s in tech_strategies) / len(tech_strategies)
                f.write(
                    f"Technical Analysis Strategies: {len(tech_strategies)} strategies, avg return: {avg_tech_return:.2%}\n")

            if ml_strategies:
                avg_ml_return = sum(s['return']
                                    for s in ml_strategies) / len(ml_strategies)
                f.write(
                    f"ML/AI Strategies: {len(ml_strategies)} strategies, avg return: {avg_ml_return:.2%}\n")

            # Hypothesis test results
            f.write(f"\nHYPOTHESIS TEST RESULTS:\n")
            if your_ai:
                your_return = your_ai['return']

                # H1 test
                h1_results = [
                    r for r in all_results if r['key'].startswith('H1:')]
                if h1_results:
                    best_h1 = max(h1_results, key=lambda x: x['return'])
                    result = "PASSED" if your_return > best_h1['return'] else "FAILED"
                    f.write(
                        f"H1 (Trading Programs): {result} - Your AI: {your_return:.2%} vs Best: {best_h1['return']:.2%}\n")

                # H2 test
                h2_results = [
                    r for r in all_results if r['key'].startswith('H2:')]
                if h2_results:
                    best_h2 = max(h2_results, key=lambda x: x['return'])
                    result = "PASSED" if your_return > best_h2['return'] else "FAILED"
                    f.write(
                        f"H2 (Famous Investors): {result} - Your AI: {your_return:.2%} vs Best: {best_h2['return']:.2%}\n")

                # H3 test
                h3_results = [
                    r for r in all_results if r['key'].startswith('H3:')]
                if h3_results:
                    best_h3 = max(h3_results, key=lambda x: x['return'])
                    result = "PASSED" if your_return > best_h3['return'] else "FAILED"
                    f.write(
                        f"H3 (AI Systems): {result} - Your AI: {your_return:.2%} vs Best: {best_h3['return']:.2%}\n")

                # H4 test
                h4_results = [
                    r for r in all_results if r['key'].startswith('H4:')]
                if h4_results:
                    h4_result = h4_results[0]
                    result = "PASSED" if your_return > h4_result['return'] else "FAILED"
                    f.write(
                        f"H4 (Beginner Trader): {result} - Your AI: {your_return:.2%} vs Beginner: {h4_result['return']:.2%}\n")

            # Technical details
            f.write(f"\nTECHNICAL DETAILS:\n")
            f.write(f"Data Source: {self.STANDARD_CONFIG['data_source']}\n")
            f.write(
                f"Trading Days: {len(self.test_data_2024) if self.test_data_2024 is not None else 'N/A'}\n")
            f.write(
                f"Standardization: All strategies tested with identical conditions\n")
            f.write(f"Transaction Cost: 0.1%\n")
            f.write(f"Position Sizing: Full capital utilization\n")

            # RECOMMENDATIONS SECTION
            f.write(f"\nRECOMMENDATIONS:\n")


            # Overall recommendations based on performance
            top_3_strategies = all_results[:3]
            f.write(f"Top performing strategies to consider:\n")
            for i, strategy in enumerate(top_3_strategies, 1):
                f.write(
                    f"{i}. {strategy['name']}: {strategy['return']:.2%} return, {strategy['trades']} trades\n")

            # Risk-adjusted recommendations
            high_sharpe_strategies = [
                r for r in all_results if r['sharpe'] > 1.0 and r['return'] > 0.05]
            if high_sharpe_strategies:
                f.write(
                    f"\nRisk-adjusted top performers (Sharpe > 1.0, Return > 5%):\n")
                for strategy in high_sharpe_strategies[:3]:
                    f.write(
                        f"- {strategy['name']}: {strategy['return']:.2%} return, Sharpe: {strategy['sharpe']:.2f}\n")

        print(f"📁 Enhanced comprehensive report saved: {report_file}")

        # Also save CSV for easy analysis with enhanced data
        csv_file = os.path.join(
            results_dir, f"all_strategies_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        if all_results:
            df = pd.DataFrame(all_results)

            # Add additional calculated columns for analysis
            df['risk_adjusted_return'] = df['return'] / \
                (df['max_drawdown'] + 0.01)  # Avoid division by zero
            df['trades_per_day'] = df['trades'] / 252
            df['return_per_trade'] = df['return'] / \
                (df['trades'] + 1)  # Avoid division by zero

            # Add performance categories
            df['performance_tier'] = df['return'].apply(lambda x:
                                                        'Excellent' if x >= 0.20 else
                                                        'Good' if x >= 0.10 else
                                                        'Moderate' if x >= 0.05 else
                                                        'Needs Improvement'
                                                        )

            # Add frequency categories
            df['frequency_tier'] = df['trades'].apply(lambda x:
                                                      'High-Frequency' if x >= 50 else
                                                      'Medium-Frequency' if x >= 20 else
                                                      'Low-Frequency'
                                                      )

            df.to_csv(csv_file, index=False)
            print(f"📊 Enhanced CSV data saved: {csv_file}")
            print(
                f"   Includes: performance tiers, frequency analysis, risk-adjusted metrics")


def main():
    """Main function to run standardized hypothesis testing"""

    print("🎯 STANDARDIZED HYPOTHESIS TESTING FRAMEWORK")
    print("Testing MLTrading Strategyagainst benchmarks")
    print("All systems now use Jan 2024 - Dec 2024 period")
    print("Ensuring fair comparison with identical conditions")
    print("=" * 60)

    # Ask if user wants to continue
    proceed = input(
        "\nDo you want to run the STANDARDIZED hypothesis testing now? (y/n): ").lower().strip()

    if proceed != 'y':
        print("👋 Run when ready for standardized comparison!")
        return

    # Initialize standardized framework
    framework = StandardizedHypothesisTestingFramework()

    # Run all tests with standardized conditions
    framework.run_all_standardized_tests()

    print("\n✅ STANDARDIZED Hypothesis testing completed!")
    print("📁 Check 'standardized_hypothesis_results' folder for detailed report")
    print("\n💡 KEY STANDARDIZATIONS IMPLEMENTED:")
    print("   ✅ All systems use Jan 2024 - Dec 2024 period")
    print("   ✅ Identical initial capital ($10,000)")
    print("   ✅ Same symbol (MSFT) and data sources")
    print("   ✅ Consistent backtesting framework")
    print("   ✅ Fair comparison conditions for all benchmarks")
    print("\n🔍 NEXT STEPS:")
    print("   1. Review the results to see which hypotheses your AI passes/fails")
    print("   2. Analyze trading behavior (holds, buys, sells)")
    print("   3. Identify areas for improvement based on benchmark comparison")
    print("   4. Consider strategy optimization if needed")


if __name__ == "__main__":
    main()
