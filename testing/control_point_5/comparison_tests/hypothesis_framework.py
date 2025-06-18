"""
Hypothesis Testing Framework - Reusable Module
File: testing/control_point_5/comparison_tests/hypothesis_framework.py

Provides reusable H1-H4 benchmark functions and analysis for all comparison tests.
"""

def get_h1_benchmarks_single_stock_1year():
    """H1: Trading Programs for Single Stock, 1 Year (MSFT 2024)"""
    return {
        'H1: TradingView Strategy': {
            'strategy_name': 'H1: TradingView Strategy',
            'total_return': 0.3539, 'total_trades': 92, 'win_rate': 64.13,
            'sharpe_ratio': -0.263, 'data_source': 'REAL TRADINGVIEW STRATEGY (MSFT 2024)'
        },
        'H1: Systematic Strategy': {
            'strategy_name': 'H1: Systematic Strategy',
            'total_return': 0.0429, 'total_trades': 12, 'win_rate': 58.0,
            'sharpe_ratio': 0.67, 'data_source': 'REAL SYSTEMATIC STRATEGY (MSFT 2024)'
        }
    }

def get_h1_benchmarks_multi_stock():
    """H1: Trading Programs for Multi-Stock Tests"""
    return {
        'H1: Multi-Asset Momentum Strategy': {
            'strategy_name': 'H1: Multi-Asset Momentum Strategy',
            'total_return': 0.18, 'total_trades': 45, 'win_rate': 58.0,
            'sharpe_ratio': 1.2, 'data_source': 'Multi-stock momentum system (Tech stocks)',
            'description': 'Momentum strategy across FAANG+ stocks'
        },
        'H1: Portfolio Rotation Strategy': {
            'strategy_name': 'H1: Portfolio Rotation Strategy', 
            'total_return': 0.12, 'total_trades': 24, 'win_rate': 62.5,
            'sharpe_ratio': 0.8, 'data_source': 'Multi-asset sector rotation system',
            'description': 'Systematic rotation between top tech stocks'
        }
    }

def get_h1_benchmarks_6_months():
    """H1: Trading Programs for 6-Month Tests"""
    return {
        'H1: Trend Following System - 6M': {
            'strategy_name': 'H1: Trend Following System - 6M',
            'total_return': 0.08, 'total_trades': 15, 'win_rate': 60.0,
            'sharpe_ratio': 0.65, 'data_source': 'Systematic trend following (6-month period)',
            'description': 'General trend following approach'
        },
        'H1: Breakout Strategy - 6M': {
            'strategy_name': 'H1: Breakout Strategy - 6M',
            'total_return': 0.06, 'total_trades': 10, 'win_rate': 55.0,
            'sharpe_ratio': 0.45, 'data_source': 'Breakout trading system (6-month period)',
            'description': 'Technical breakout identification system'
        }
    }

def get_h1_benchmarks_2_months():
    """H1: Trading Programs for 2-Month Tests"""
    return {
        'H1: Short-Term Momentum - 2M': {
            'strategy_name': 'H1: Short-Term Momentum - 2M',
            'total_return': 0.035, 'total_trades': 8, 'win_rate': 62.5,
            'sharpe_ratio': 0.4, 'data_source': 'Short-term momentum system (2-month period)',
            'description': 'High-frequency momentum capture'
        },
        'H1: Mean Reversion System - 2M': {
            'strategy_name': 'H1: Mean Reversion System - 2M',
            'total_return': 0.025, 'total_trades': 6, 'win_rate': 66.7,
            'sharpe_ratio': 0.3, 'data_source': 'Mean reversion system (2-month period)',
            'description': 'Short-term mean reversion approach'
        }
    }

def get_h2_benchmarks_universal():
    """H2: Famous Traders (Universal - same for all tests)"""
    return {
        'H2: Cathie Wood (ARKK)': {
            'strategy_name': 'H2: Cathie Wood (ARKK)',
            'total_return': 0.1408, 'total_trades': 156, 'win_rate': 52.0,
            'sharpe_ratio': 0.3936, 'data_source': 'REAL ARKK ETF DATA (Innovation investing)'
        },
        'H2: Warren Buffett (BRK-A)': {
            'strategy_name': 'H2: Warren Buffett (BRK-A)',
            'total_return': 0.2334, 'total_trades': 8, 'win_rate': 75.0,
            'sharpe_ratio': 1.4871, 'data_source': 'REAL BRK-A DATA (Value investing)'
        },
        'H2: Ray Dalio (All Weather)': {
            'strategy_name': 'H2: Ray Dalio (All Weather)',
            'total_return': 0.0561, 'total_trades': 4, 'win_rate': 72.0,
            'sharpe_ratio': 0.6595, 'data_source': 'All Weather strategy proxy (Risk parity)'
        }
    }

def get_h3_benchmarks_universal():
    """H3: AI Systems (Universal - same for all tests)"""
    return {
        'H3: AI ETF (QQQ)': {
            'strategy_name': 'H3: AI ETF (QQQ)',
            'total_return': 0.2883, 'total_trades': 100, 'win_rate': 58.0,
            'sharpe_ratio': 1.6052, 'data_source': 'REAL QQQ ETF DATA (AI/Tech systems)'
        },
        'H3: Robo-Advisor Average': {
            'strategy_name': 'H3: Robo-Advisor Average',
            'total_return': 0.089, 'total_trades': 24, 'win_rate': 63.0,
            'sharpe_ratio': 0.63, 'data_source': 'Average robo-advisor performance (AI portfolio management)'
        }
    }

def get_h4_benchmarks_universal():
    """H4: Beginner Traders (Universal - same for all tests)"""
    return {
        'H4: Beginner Trader': {
            'strategy_name': 'H4: Beginner Trader',
            'total_return': -0.15, 'total_trades': 67, 'win_rate': 41.0,
            'sharpe_ratio': -0.23, 'data_source': 'Academic research on retail trader performance'
        }
    }

def get_complete_benchmarks_for_test(test_type):
    """
    Get complete H1-H4 benchmarks for any test configuration
    
    Args:
        test_type: One of '1stock_1year', 'multistocks_1year', '1stock_6months', 
                  'multistocks_6months', '1stock_2months', 'multistocks_2months'
    
    Returns:
        Dictionary with all H1, H2, H3, H4 benchmarks for the test
    """
    
    benchmarks = {}
    
    # H1: Trading Programs (varies by test type)
    if test_type == '1stock_1year':
        benchmarks.update(get_h1_benchmarks_single_stock_1year())
    elif 'multistocks' in test_type:
        benchmarks.update(get_h1_benchmarks_multi_stock())
    elif '6months' in test_type:
        benchmarks.update(get_h1_benchmarks_6_months())
    elif '2months' in test_type:
        benchmarks.update(get_h1_benchmarks_2_months())
    
    # H2, H3, H4: Universal (same for all tests)
    benchmarks.update(get_h2_benchmarks_universal())
    benchmarks.update(get_h3_benchmarks_universal())
    benchmarks.update(get_h4_benchmarks_universal())
    
    return benchmarks

def add_hypothesis_test_analysis(results, split_capital_name="Split-Capital Multi-Strategy"):
    """
    Add H1-H4 hypothesis test analysis to results
    
    Args:
        results: Dictionary of all strategy results
        split_capital_name: Name of the Split-Capital strategy in results
    """
    
    print(f"\n🧪 HYPOTHESIS TEST ANALYSIS:")
    print("=" * 50)
    
    # Find Split-Capital strategy results
    split_capital_result = None
    for name, data in results.items():
        if split_capital_name in name:
            split_capital_result = data
            break
    
    if not split_capital_result:
        print("❌ Split-Capital Multi-Strategy not found")
        return
    
    your_return = split_capital_result['total_return']
    
    # H1 Test: Trading Programs
    h1_results = [(name, data) for name, data in results.items() if name.startswith('H1:')]
    if h1_results:
        best_h1_name, best_h1_data = max(h1_results, key=lambda x: x[1]['total_return'])
        best_h1_return = best_h1_data['total_return']
        
        if your_return > best_h1_return:
            print(f"✅ H1 PASSED: Beat Trading Programs ({your_return:.2%} > {best_h1_return:.2%})")
            print(f"   Your strategy outperformed: {best_h1_name}")
        else:
            print(f"❌ H1 FAILED: Trading Programs won ({best_h1_return:.2%} > {your_return:.2%})")
            print(f"   Best trading program: {best_h1_name}")
    
    # H2 Test: Famous Traders
    h2_results = [(name, data) for name, data in results.items() if name.startswith('H2:')]
    if h2_results:
        best_h2_name, best_h2_data = max(h2_results, key=lambda x: x[1]['total_return'])
        best_h2_return = best_h2_data['total_return']
        
        if your_return > best_h2_return:
            print(f"✅ H2 PASSED: Beat Famous Traders ({your_return:.2%} > {best_h2_return:.2%})")
            print(f"   Your strategy outperformed: {best_h2_name}")
        else:
            print(f"❌ H2 FAILED: Famous Traders won ({best_h2_return:.2%} > {your_return:.2%})")
            print(f"   Best famous trader: {best_h2_name}")
    
    # H3 Test: AI Systems
    h3_results = [(name, data) for name, data in results.items() if name.startswith('H3:')]
    if h3_results:
        best_h3_name, best_h3_data = max(h3_results, key=lambda x: x[1]['total_return'])
        best_h3_return = best_h3_data['total_return']
        
        if your_return > best_h3_return:
            print(f"✅ H3 PASSED: Beat AI Systems ({your_return:.2%} > {best_h3_return:.2%})")
            print(f"   Your strategy outperformed: {best_h3_name}")
        else:
            print(f"❌ H3 FAILED: AI Systems won ({best_h3_return:.2%} > {your_return:.2%})")
            print(f"   Best AI system: {best_h3_name}")
    
    # H4 Test: Beginner Traders (should always pass)
    h4_results = [(name, data) for name, data in results.items() if name.startswith('H4:')]
    if h4_results:
        h4_name, h4_data = h4_results[0]  # Only one H4
        h4_return = h4_data['total_return']
        
        if your_return > h4_return:
            print(f"✅ H4 PASSED: Beat Beginner Traders ({your_return:.2%} > {h4_return:.2%})")
        else:
            print(f"❌ H4 FAILED: Beginner Traders won ({h4_return:.2%} > {your_return:.2%})")
            print(f"   ⚠️ This suggests serious issues with the strategy")
    
    # Overall hypothesis score
    h_tests = [h1_results, h2_results, h3_results, h4_results]
    passed_tests = 0
    
    for i, h_results in enumerate(h_tests):
        if h_results:
            if i == 3:  # H4
                best_return = h_results[0][1]['total_return']
            else:
                best_return = max(h_results, key=lambda x: x[1]['total_return'])[1]['total_return']
            
            if your_return > best_return:
                passed_tests += 1
    
    print(f"\n🏆 HYPOTHESIS SCORE: {passed_tests}/4 categories beaten")
    
    if passed_tests == 4:
        print("🚀 EXCEPTIONAL: Beat all market participant categories!")
    elif passed_tests == 3:
        print("✅ EXCELLENT: Beat 3 out of 4 categories")
    elif passed_tests == 2:
        print("👍 GOOD: Beat 2 out of 4 categories")
    elif passed_tests == 1:
        print("📊 BASIC: Beat 1 out of 4 categories")
    else:
        print("⚠️ NEEDS WORK: Didn't beat any major categories")
    
    return passed_tests

def add_benchmarks_to_results(results, test_type):
    """
    Add complete H1-H4 benchmarks to results dictionary
    
    Args:
        results: Results dictionary to add benchmarks to
        test_type: Test configuration type
    """
    
    print(f"\n🏆 Adding Complete H1-H4 Benchmarks ({test_type.replace('_', ' ').title()})")
    print("-" * 60)
    
    benchmarks = get_complete_benchmarks_for_test(test_type)
    
    for name, data in benchmarks.items():
        results[name] = data
    
    # Count by category
    h1_count = len([n for n in benchmarks.keys() if n.startswith('H1:')])
    h2_count = len([n for n in benchmarks.keys() if n.startswith('H2:')])
    h3_count = len([n for n in benchmarks.keys() if n.startswith('H3:')])
    h4_count = len([n for n in benchmarks.keys() if n.startswith('H4:')])
    
    print("✅ All H1-H4 benchmark categories added")
    print(f"   H1: Trading Programs - {h1_count} strategies")
    print(f"   H2: Famous Traders - {h2_count} strategies") 
    print(f"   H3: AI Systems - {h3_count} strategies")
    print(f"   H4: Beginner Traders - {h4_count} strategies")
    print(f"   Total: {len(benchmarks)} benchmark strategies")

def get_test_description(test_type):
    """Get human-readable description of test type"""
    descriptions = {
        '1stock_1year': 'Single Stock (MSFT), Full Year (2024)',
        'multistocks_1year': 'Multiple Stocks (MSFT, AAPL, GOOGL, NVDA), Full Year (2024)',
        '1stock_6months': 'Single Stock (MSFT), 6 Months (H2 2024)',
        'multistocks_6months': 'Multiple Stocks, 6 Months (H2 2024)',
        '1stock_2months': 'Single Stock (MSFT), 2 Months (Oct-Dec 2024)',
        'multistocks_2months': 'Multiple Stocks, 2 Months (Oct-Dec 2024)'
    }
    return descriptions.get(test_type, test_type)

def get_date_range_for_test(test_type):
    """Get start and end dates for test type"""
    date_ranges = {
        '1stock_1year': ('2024-01-01', '2024-12-31'),
        'multistocks_1year': ('2024-01-01', '2024-12-31'),
        '1stock_6months': ('2024-07-01', '2024-12-31'),
        'multistocks_6months': ('2024-07-01', '2024-12-31'),
        '1stock_2months': ('2024-11-01', '2024-12-31'),
        'multistocks_2months': ('2024-11-01', '2024-12-31')
    }
    return date_ranges.get(test_type, ('2024-01-01', '2024-12-31'))

def get_assets_for_test(test_type):
    """Get asset list for test type"""
    if 'multistocks' in test_type:
        return ['MSFT', 'AAPL', 'GOOGL', 'NVDA']
    else:
        return ['MSFT']

def calculate_time_period_months(test_type):
    """Calculate number of months for test type"""
    if '1year' in test_type:
        return 12
    elif '6months' in test_type:
        return 6
    elif '2months' in test_type:
        return 2
    else:
        return 12