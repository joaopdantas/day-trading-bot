"""
ENHANCED COMPREHENSIVE HYPOTHESIS TESTING ANALYSIS
Combines robust parsing with hypothesis-focused analysis and beautiful visualizations

Features:
- Specialized H1-H4 hypothesis testing analysis
- Comprehensive single dashboard with 9 analysis panels
- Robust error handling and flexible parsing optimized for your file format
- Trade efficiency and clear winners identification
- Multiple export formats (visual + detailed CSVs)
- Executive summary with strategic recommendations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import re
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")

class EnhancedHypothesisAnalyzer:
    """Enhanced hypothesis testing analyzer with robust parsing and comprehensive analysis"""
    
    def __init__(self, results_folder="comparison_tests_results"):
        self.results_folder = results_folder
        self.data = {}
        self.strategies = {}  # Our trading strategies
        self.benchmarks = {}  # H1-H4 hypothesis benchmarks
        self.summary_stats = {}
        
        print("🏆 ENHANCED COMPREHENSIVE HYPOTHESIS TESTING ANALYSIS")
        print("=" * 65)
        print("🔬 Specialized for H1-H4 hypothesis testing with robust parsing")
    
    def load_all_results(self):
        """Enhanced data loading with robust error handling"""
        print("\n📂 Loading all result files...")
        
        # Multiple file pattern attempts for robustness
        patterns = [
            os.path.join(self.results_folder, "results_*.txt"),
            os.path.join(self.results_folder, "*.txt"),
            os.path.join(self.results_folder, "hypothesis_*.txt")
        ]
        
        files = []
        for pattern in patterns:
            found_files = glob.glob(pattern)
            if found_files:
                files.extend(found_files)
                print(f"📊 Found {len(found_files)} files with pattern: {pattern}")
                break
        
        if not files:
            print(f"❌ No result files found in {self.results_folder}/")
            return False
        
        print(f"📊 Processing {len(files)} result files")
        
        successful_loads = 0
        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"   📄 Loading {filename}...")
            
            try:
                # Extract test configuration from filename
                test_config = self._extract_test_config(filename)
                
                # Parse the file with enhanced error handling
                file_data = self._parse_result_file(file_path)
                
                if file_data:
                    self.data[test_config] = file_data
                    successful_loads += 1
                    print(f"      ✅ Loaded {len(file_data)} strategies/benchmarks")
                else:
                    print(f"      ⚠️  No data extracted from {filename}")
                    
            except Exception as e:
                print(f"      ❌ Error processing {filename}: {e}")
        
        if successful_loads > 0:
            self._organize_hypothesis_data()
            print(f"\n📈 Successfully loaded {successful_loads} test configurations")
            print(f"🎯 Found {len(self.strategies)} strategy configurations")
            print(f"🧪 Found {len(self.benchmarks)} benchmark configurations")
            return True
        else:
            print("❌ No data loaded successfully")
            return False
    
    def _extract_test_config(self, filename):
        """Enhanced test configuration extraction"""
        # Multiple patterns to handle different filename formats
        patterns = [
            r'results_(.+?)_(\d{8}_\d{6})\.txt',  # results_multistocks_6months_20250619_174912.txt
            r'results_(.+?)\.txt',                 # results_multistocks_6months.txt
            r'(.+?)_results\.txt',                 # multistocks_6months_results.txt
            r'hypothesis_(.+?)\.txt',              # hypothesis_multistocks_6months.txt
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                config = match.group(1)
                return config.replace('_', ' ').title()
        
        # Fallback to filename without extension
        return filename.replace('.txt', '').replace('_', ' ').title()
    
    def _parse_result_file(self, file_path):
        """Enhanced parsing optimized for your specific result file format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"      ⚠️  Empty file: {file_path}")
                return None
            
            print(f"      🔍 File size: {len(content)} characters")
            
            strategies = {}
            current_strategy = None
            total_metrics = 0
            
            lines = content.split('\n')
            print(f"      📄 Processing {len(lines)} lines...")
            
            for line in lines:
                line_stripped = line.strip()
                
                # Skip empty lines and file headers
                if not line_stripped or line_stripped.startswith('='):
                    continue
                    
                # Skip configuration lines
                if any(header in line_stripped for header in [
                    'HYPOTHESIS TESTING RESULTS:', 'Generated:', 'Test Configuration:', 
                    'Assets:', 'Period:', 'Capital:', 'Duration:']):
                    continue
                
                # Detect strategy headers (end with : and not indented)
                if line_stripped.endswith(':') and not line.startswith(' '):
                    current_strategy = line_stripped[:-1].strip()
                    strategies[current_strategy] = {}
                    print(f"        📊 Found strategy: {current_strategy}")
                    continue
                
                # Parse data lines (start with exactly 2 spaces)
                if current_strategy and line.startswith('  ') and ':' in line:
                    # Remove the 2-space indentation
                    data_line = line[2:].strip()
                    key, value = self._parse_data_line(data_line)
                    
                    if key and value is not None:
                        strategies[current_strategy][key] = value
                        total_metrics += 1
                        
                        # Debug critical metrics
                        if key in ['total_return', 'sharpe_ratio', 'win_rate', 'total_trades']:
                            print(f"          ✅ {key}: {value}")
            
            print(f"      📈 Parsed {total_metrics} total metrics")
            
            # Validate strategies have essential data
            validated_strategies = self._validate_strategy_data(strategies)
            
            if validated_strategies:
                print(f"      ✅ Validated {len(validated_strategies)} strategies with complete data")
                return validated_strategies
            else:
                print(f"      ❌ No strategies with valid performance data found")
                return None
            
        except Exception as e:
            print(f"      ❌ Parse error in {file_path}: {e}")
            return None
    
    def _parse_data_line(self, line):
        """Parse data line and convert values appropriately"""
        try:
            if ':' not in line:
                return None, None
                
            # Split on first colon only
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Skip empty keys or values
            if not key or not value:
                return None, None
            
            # Handle complex data structures (skip them for now, focus on metrics)
            if value.startswith('[') or value.startswith('{') or '<' in value:
                return key, value  # Store as string for complex data
            
            # Handle numeric conversions
            try:
                # Handle percentage format (but most data is decimal)
                if '%' in value:
                    return key, float(value.replace('%', '')) / 100
                
                # Handle currency format  
                elif '$' in value:
                    return key, float(value.replace('$', '').replace(',', ''))
                
                # Handle decimal numbers (most common)
                elif self._is_numeric(value):
                    return key, float(value.replace(',', ''))
                
                # Handle string values
                else:
                    return key, value
                    
            except (ValueError, TypeError):
                return key, value  # Store as string if conversion fails
            
        except Exception:
            return None, None
    
    def _is_numeric(self, value):
        """Check if string represents a number"""
        try:
            float(value.replace(',', ''))
            return True
        except (ValueError, TypeError):
            return False
        
    def _normalize_win_rate(self, win_rate_value):
        """Normalize win rate to percentage (0-100 range)"""
        if isinstance(win_rate_value, (int, float)):
            if win_rate_value <= 1.0:
                # It's in decimal format (0.75), convert to percentage
                return win_rate_value * 100
            else:
                # It's already in percentage format (75.0), use as-is
                return win_rate_value
        return 0
    
    def _validate_strategy_data(self, strategies):
        """Validate that strategies have essential performance metrics"""
        validated = {}
        required_metrics = ['total_return']  # Minimum required
        
        for strategy_name, strategy_data in strategies.items():
            if not strategy_data:
                continue
            
            # Check for essential metrics
            has_essential = any(metric in strategy_data for metric in required_metrics)
            
            if has_essential:
                validated[strategy_name] = strategy_data
                print(f"        ✅ {strategy_name}: Valid data ({len(strategy_data)} metrics)")
            else:
                print(f"        ❌ {strategy_name}: Missing essential metrics")
        
        return validated
    
    def _organize_hypothesis_data(self):
        """Enhanced data organization with hypothesis separation"""
        print("\n🔬 Organizing hypothesis testing data...")
        
        for test_config, test_data in self.data.items():
            strategy_count = 0
            benchmark_count = 0
            
            for strategy_name, strategy_data in test_data.items():
                # Enhanced hypothesis benchmark detection
                if self._is_hypothesis_benchmark(strategy_name):
                    # This is a H1-H4 benchmark
                    if test_config not in self.benchmarks:
                        self.benchmarks[test_config] = {}
                    self.benchmarks[test_config][strategy_name] = strategy_data
                    benchmark_count += 1
                else:
                    # This is our trading strategy
                    if test_config not in self.strategies:
                        self.strategies[test_config] = {}
                    self.strategies[test_config][strategy_name] = strategy_data
                    strategy_count += 1
            
            print(f"  📊 {test_config}: {strategy_count} strategies, {benchmark_count} benchmarks")
    
    def _is_hypothesis_benchmark(self, name):
        """Enhanced hypothesis benchmark detection"""
        hypothesis_patterns = [
            'H1:', 'H2:', 'H3:', 'H4:',  # Standard format
            'h1:', 'h2:', 'h3:', 'h4:',  # Lowercase
            'Hypothesis 1', 'Hypothesis 2', 'Hypothesis 3', 'Hypothesis 4'
        ]
        return any(pattern in name for pattern in hypothesis_patterns)
    
    def create_enhanced_comprehensive_analysis(self):
        """Create enhanced comprehensive analysis with hypothesis focus"""
        print("\n🎨 Creating enhanced comprehensive analysis...")
        
        # Create large figure with 9 specialized panels
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Hypothesis Performance Overview (top left)
        ax1 = plt.subplot(3, 3, 1)
        self._plot_hypothesis_overview(ax1)
        
        # 2. Our Strategy vs Benchmarks (top center)
        ax2 = plt.subplot(3, 3, 2)
        self._plot_strategy_vs_benchmarks(ax2)
        
        # 3. Time Period Hypothesis Analysis (top right)
        ax3 = plt.subplot(3, 3, 3)
        self._plot_time_period_hypothesis(ax3)
        
        # 4. Risk-Return Hypothesis Map (middle left)
        ax4 = plt.subplot(3, 3, 4)
        self._plot_hypothesis_risk_return(ax4)
        
        # 5. Strategy Performance Rankings (middle center)
        ax5 = plt.subplot(3, 3, 5)
        self._plot_enhanced_strategy_rankings(ax5)
        
        # 6. H1-H4 Detailed Comparison (middle right)
        ax6 = plt.subplot(3, 3, 6)
        self._plot_detailed_hypothesis_comparison(ax6)
        
        # 7. Trade Efficiency Analysis (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        self._plot_enhanced_trade_efficiency(ax7)
        
        # 8. Win Rate vs Hypothesis Performance (bottom center)
        ax8 = plt.subplot(3, 3, 8)
        self._plot_hypothesis_win_rate(ax8)
        
        # 9. CLEAR WINNERS SUMMARY (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        self._plot_enhanced_winners_summary(ax9)
        
        plt.suptitle('🏆 ENHANCED HYPOTHESIS TESTING ANALYSIS 🏆\n' + 
                    f'Comprehensive H1-H4 Performance Evaluation | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the enhanced analysis in a dedicated subfolder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_folder = os.path.join(self.results_folder, 'enhanced_analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        filename = f'{analysis_folder}/ENHANCED_HYPOTHESIS_ANALYSIS_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Enhanced analysis saved to {filename}")
        
        # Generate comprehensive reports
        self._generate_enhanced_reports(timestamp)
    
    def _plot_hypothesis_overview(self, ax):
        """Plot comprehensive hypothesis testing overview with data validation"""
        performance_data = []
        strategy_count = 0
        benchmark_count = 0
        
        # Collect all performance data with validation
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data:
                    performance_data.append({
                        'type': 'Our Strategy',
                        'return': strategy_data['total_return'] * 100
                    })
                    strategy_count += 1
        
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data:
                    performance_data.append({
                        'type': 'Benchmark',
                        'return': benchmark_data['total_return'] * 100
                    })
                    benchmark_count += 1
        
        print(f"📊 Overview data: {strategy_count} strategies, {benchmark_count} benchmarks")
        
        if not performance_data:
            ax.text(0.5, 0.5, '❌ NO PERFORMANCE DATA FOUND\n\nCheck data parsing and metrics', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.set_title('🧪 Data Validation Error', fontweight='bold', fontsize=12)
            return
        
        df = pd.DataFrame(performance_data)
        avg_performance = df.groupby('type')['return'].mean()
        
        colors = ['#2E8B57', '#DC143C']  # Green for strategies, Red for benchmarks
        bars = ax.bar(avg_performance.index, avg_performance.values, color=colors[:len(avg_performance)], alpha=0.8)
        
        ax.set_ylabel('Average Return (%)', fontweight='bold')
        ax.set_title('🧪 Our Strategies vs H1-H4 Benchmarks', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and counts
        for i, (bar, value) in enumerate(zip(bars, avg_performance.values)):
            count = strategy_count if avg_performance.index[i] == 'Our Strategy' else benchmark_count
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:+.1f}%\n({count} items)', ha='center', va='bottom', fontweight='bold')
    
    def _plot_strategy_vs_benchmarks(self, ax):
        """Detailed strategy vs benchmark comparison"""
        # Get best strategy and best benchmark
        best_strategy = {'name': '', 'return': -999, 'config': ''}
        best_benchmark = {'name': '', 'return': -999, 'config': ''}
        
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data:
                    if strategy_data['total_return'] > best_strategy['return']:
                        best_strategy = {
                            'name': strategy_name,
                            'return': strategy_data['total_return'] * 100,
                            'config': test_config
                        }
        
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data:
                    if benchmark_data['total_return'] > best_benchmark['return']:
                        best_benchmark = {
                            'name': benchmark_name,
                            'return': benchmark_data['total_return'] * 100,
                            'config': test_config
                        }
        
        if best_strategy['name'] and best_benchmark['name']:
            categories = ['Best Strategy\n(Ours)', 'Best Benchmark\n(H1-H4)']
            values = [best_strategy['return'], best_benchmark['return']]
            colors = ['#32CD32', '#FF6347']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8)
            
            ax.set_ylabel('Return (%)', fontweight='bold')
            ax.set_title('🏅 Best Performance Comparison', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:+.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_time_period_hypothesis(self, ax):
        """Enhanced time period analysis with hypothesis focus"""
        time_data = {'3months': {'strategies': [], 'benchmarks': []}, 
                    '6months': {'strategies': [], 'benchmarks': []}, 
                    '1year': {'strategies': [], 'benchmarks': []}}
        
        # Collect strategy data
        for test_config, strategies in self.strategies.items():
            for period in time_data.keys():
                if period in test_config.lower():
                    for strategy_name, strategy_data in strategies.items():
                        if 'total_return' in strategy_data:
                            time_data[period]['strategies'].append(strategy_data['total_return'] * 100)
        
        # Collect benchmark data
        for test_config, benchmarks in self.benchmarks.items():
            for period in time_data.keys():
                if period in test_config.lower():
                    for benchmark_name, benchmark_data in benchmarks.items():
                        if 'total_return' in benchmark_data:
                            time_data[period]['benchmarks'].append(benchmark_data['total_return'] * 100)
        
        # Create grouped bar chart
        periods = []
        strategy_avgs = []
        benchmark_avgs = []
        
        for period, data in time_data.items():
            if data['strategies'] or data['benchmarks']:
                periods.append(period.replace('months', 'M').replace('1year', '1Y'))
                strategy_avgs.append(np.mean(data['strategies']) if data['strategies'] else 0)
                benchmark_avgs.append(np.mean(data['benchmarks']) if data['benchmarks'] else 0)
        
        if periods:
            x = np.arange(len(periods))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, strategy_avgs, width, label='Our Strategies', color='#2E8B57', alpha=0.8)
            bars2 = ax.bar(x + width/2, benchmark_avgs, width, label='H1-H4 Benchmarks', color='#DC143C', alpha=0.8)
            
            ax.set_xlabel('Time Period', fontweight='bold')
            ax.set_ylabel('Average Return (%)', fontweight='bold')
            ax.set_title('⏱️ Time Period Performance Analysis', fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(periods)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_hypothesis_risk_return(self, ax):
        """Enhanced risk-return analysis with hypothesis focus"""
        risk_return_data = []
        
        # Collect strategy data
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data and 'sharpe_ratio' in strategy_data:
                    risk_return_data.append({
                        'return': strategy_data['total_return'] * 100,
                        'sharpe': strategy_data['sharpe_ratio'],
                        'type': 'Our Strategy'
                    })
        
        # Collect benchmark data
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data and 'sharpe_ratio' in benchmark_data:
                    risk_return_data.append({
                        'return': benchmark_data['total_return'] * 100,
                        'sharpe': benchmark_data['sharpe_ratio'],
                        'type': 'Benchmark'
                    })
        
        if risk_return_data:
            df = pd.DataFrame(risk_return_data)
            
            # Create scatter plot
            strategies_df = df[df['type'] == 'Our Strategy']
            benchmarks_df = df[df['type'] == 'Benchmark']
            
            if not strategies_df.empty:
                ax.scatter(strategies_df['sharpe'], strategies_df['return'], 
                          c='#2E8B57', alpha=0.7, s=100, edgecolors='black', 
                          label='Our Strategies', marker='o')
            
            if not benchmarks_df.empty:
                ax.scatter(benchmarks_df['sharpe'], benchmarks_df['return'], 
                          c='#DC143C', alpha=0.7, s=100, edgecolors='black', 
                          label='H1-H4 Benchmarks', marker='^')
            
            ax.set_xlabel('Sharpe Ratio', fontweight='bold')
            ax.set_ylabel('Return (%)', fontweight='bold')
            ax.set_title('📈 Risk-Return Analysis', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_enhanced_strategy_rankings(self, ax):
        """Enhanced strategy rankings with comprehensive data validation"""
        all_performance = []
        
        # Collect all strategy performance
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data:
                    all_performance.append({
                        'name': strategy_name,
                        'return': strategy_data['total_return'] * 100,
                        'type': 'Strategy'
                    })
        
        # Add benchmarks for comparison
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data:
                    all_performance.append({
                        'name': benchmark_name,
                        'return': benchmark_data['total_return'] * 100,
                        'type': 'Benchmark'
                    })
        
        if not all_performance:
            ax.text(0.5, 0.5, 'No performance data\navailable for rankings', ha='center', va='center')
            ax.set_title('🏅 Performance Rankings', fontweight='bold', fontsize=12)
            return
        
        # Sort and take top 10
        sorted_performance = sorted(all_performance, key=lambda x: x['return'], reverse=True)[:10]
        
        names = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] for item in sorted_performance]
        values = [item['return'] for item in sorted_performance]
        types = [item['type'] for item in sorted_performance]
        
        # Color by type
        colors = ['#FFD700' if t == 'Strategy' else '#C0C0C0' for t in types]
        
        bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Return (%)', fontweight='bold')
        ax.set_title(f'🏅 Top {len(sorted_performance)} Performance Rankings', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_detailed_hypothesis_comparison(self, ax):
        """Detailed H1-H4 hypothesis comparison"""
        hypothesis_data = {'H1': [], 'H2': [], 'H3': [], 'H4': []}
        
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data:
                    for h_level in hypothesis_data.keys():
                        if benchmark_name.upper().startswith(f'{h_level}:'):
                            hypothesis_data[h_level].append(benchmark_data['total_return'] * 100)
        
        # Calculate statistics for each hypothesis
        h_levels = []
        means = []
        
        for h_level, returns in hypothesis_data.items():
            if returns:
                h_levels.append(h_level)
                means.append(np.mean(returns))
        
        if h_levels:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax.bar(h_levels, means, color=colors[:len(h_levels)], alpha=0.8)
            
            ax.set_ylabel('Average Return (%)', fontweight='bold')
            ax.set_title('🧪 H1-H4 Detailed Comparison', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{mean:+.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_enhanced_trade_efficiency(self, ax):
        """Enhanced trade efficiency analysis"""
        efficiency_data = []
        
        # Strategy efficiency
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data and 'total_trades' in strategy_data:
                    trades = strategy_data['total_trades']
                    if trades > 0:
                        efficiency = (strategy_data['total_return'] * 100) / trades
                        efficiency_data.append({
                            'type': 'Strategy',
                            'efficiency': efficiency
                        })
        
        # Benchmark efficiency
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data and 'total_trades' in benchmark_data:
                    trades = benchmark_data['total_trades']
                    if trades > 0:
                        efficiency = (benchmark_data['total_return'] * 100) / trades
                        efficiency_data.append({
                            'type': 'Benchmark',
                            'efficiency': efficiency
                        })
        
        if efficiency_data:
            df = pd.DataFrame(efficiency_data)
            avg_efficiency = df.groupby('type')['efficiency'].mean()
            
            colors = ['#2E8B57', '#DC143C']
            bars = ax.bar(avg_efficiency.index, avg_efficiency.values, color=colors[:len(avg_efficiency)], alpha=0.8)
            
            ax.set_ylabel('Return per Trade (%)', fontweight='bold')
            ax.set_title('⚡ Trade Efficiency Comparison', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, avg_efficiency.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_hypothesis_win_rate(self, ax):
        """Win rate analysis with hypothesis comparison"""
        win_rate_data = []
        
        # Strategy win rates
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'win_rate' in strategy_data:
                    win_rate_data.append({
                        'type': 'Strategy',
                        'win_rate': self._normalize_win_rate(strategy_data['win_rate'])
                    })
        
        # Benchmark win rates
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'win_rate' in benchmark_data:
                    win_rate_data.append({
                        'type': 'Benchmark',
                        'win_rate': self._normalize_win_rate(benchmark_data['win_rate'])
                    })
        
        if win_rate_data:
            df = pd.DataFrame(win_rate_data)
            avg_win_rate = df.groupby('type')['win_rate'].mean()
            
            colors = ['#2E8B57', '#DC143C']
            bars = ax.bar(avg_win_rate.index, avg_win_rate.values, color=colors[:len(avg_win_rate)], alpha=0.8)
            
            ax.set_ylabel('Average Win Rate (%)', fontweight='bold')
            ax.set_title('🎯 Win Rate Comparison', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
            ax.legend()
            
            # Add value labels
            for bar, value in zip(bars, avg_win_rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_enhanced_winners_summary(self, ax):
        """Enhanced winners summary with hypothesis context"""
        # Find winners in different categories
        winners = self._find_category_winners()
        
        # Create enhanced summary display
        ax.text(0.5, 0.95, '🏆 HYPOTHESIS TESTING WINNERS 🏆', 
                ha='center', va='top', fontsize=16, fontweight='bold', 
                transform=ax.transAxes, color='darkblue')
        
        y_positions = [0.8, 0.6, 0.4, 0.2]
        categories = ['Best Overall Return', 'Best Risk-Adjusted', 'Best Win Rate', 'Best Trade Efficiency']
        
        for i, (category, winner) in enumerate(zip(categories, winners)):
            if winner['name']:
                # Color code by type
                color = '#2E8B57' if 'H' not in winner['name'] else '#DC143C'
                bbox_color = 'lightgreen' if 'H' not in winner['name'] else 'lightcoral'
                
                text = f"🥇 {category}:\n{winner['name'][:25]}...\n{winner['value']:.2f}{'%' if i != 1 else ''}\n({winner['config']})"
                
                ax.text(0.05, y_positions[i], text, 
                       ha='left', va='center', fontsize=10, fontweight='bold',
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.7))
        
        # Add hypothesis validation summary
        validation_text = self._get_hypothesis_validation_summary()
        ax.text(0.95, 0.5, validation_text, 
               ha='right', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _find_category_winners(self):
        """Find winners in each category"""
        all_entries = []
        
        # Collect all data
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                all_entries.append({
                    'name': strategy_name,
                    'config': test_config,
                    'return': strategy_data.get('total_return', 0) * 100,
                    'sharpe': strategy_data.get('sharpe_ratio', 0),
                    'win_rate': self._normalize_win_rate(strategy_data.get('win_rate', 0)),
                    'trades': strategy_data.get('total_trades', 0)
                })
        
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                all_entries.append({
                    'name': benchmark_name,
                    'config': test_config,
                    'return': benchmark_data.get('total_return', 0) * 100,
                    'sharpe': benchmark_data.get('sharpe_ratio', 0),
                    'win_rate': self._normalize_win_rate(benchmark_data.get('win_rate', 0)),
                    'trades': benchmark_data.get('total_trades', 0)
                })
        
        if not all_entries:
            return [{'name': '', 'value': 0, 'config': ''}] * 4
        
        # Find winners
        best_return = max(all_entries, key=lambda x: x['return'])
        best_sharpe = max(all_entries, key=lambda x: x['sharpe'])
        best_win_rate = max(all_entries, key=lambda x: x['win_rate'])
        
        # Best trade efficiency
        efficient_entries = [e for e in all_entries if e['trades'] > 0]
        if efficient_entries:
            for entry in efficient_entries:
                entry['efficiency'] = entry['return'] / entry['trades']
            best_efficiency = max(efficient_entries, key=lambda x: x['efficiency'])
        else:
            best_efficiency = {'name': 'N/A', 'efficiency': 0, 'config': ''}
        
        return [
            {'name': best_return['name'], 'value': best_return['return'], 'config': best_return['config']},
            {'name': best_sharpe['name'], 'value': best_sharpe['sharpe'], 'config': best_sharpe['config']},
            {'name': best_win_rate['name'], 'value': best_win_rate['win_rate'], 'config': best_win_rate['config']},
            {'name': best_efficiency['name'], 'value': best_efficiency.get('efficiency', 0), 'config': best_efficiency['config']}
        ]
    
    def _get_hypothesis_validation_summary(self):
        """Get hypothesis validation summary"""
        # Calculate if our strategies beat benchmarks
        our_avg = self._calculate_average_performance(self.strategies)
        benchmark_avg = self._calculate_average_performance(self.benchmarks)
        
        if our_avg > benchmark_avg:
            result = "✅ VALIDATED"
            detail = f"Our strategies\noutperform H1-H4\nby {our_avg - benchmark_avg:+.1f}%"
        else:
            result = "❌ NOT VALIDATED"
            detail = f"H1-H4 benchmarks\noutperform by\n{benchmark_avg - our_avg:+.1f}%"
        
        return f"🧪 HYPOTHESIS:\n{result}\n\n{detail}"
    
    def _calculate_average_performance(self, data_dict):
        """Calculate average performance from data dictionary"""
        all_returns = []
        for test_config, items in data_dict.items():
            for item_name, item_data in items.items():
                if 'total_return' in item_data:
                    all_returns.append(item_data['total_return'] * 100)
        return np.mean(all_returns) if all_returns else 0
    
    def _generate_enhanced_reports(self, timestamp):
        """Generate comprehensive enhanced reports"""
        # Generate text report
        self._generate_enhanced_text_report(timestamp)
        
        # Generate detailed CSV exports
        self._export_enhanced_csv_data(timestamp)
        
        print(f"✅ Enhanced reports generated:")
        print(f"   📊 Visual: enhanced_analysis/ENHANCED_HYPOTHESIS_ANALYSIS_{timestamp}.png")
        print(f"   📄 Report: enhanced_analysis/ENHANCED_HYPOTHESIS_REPORT_{timestamp}.txt") 
        print(f"   📈 Data: Multiple CSV files in enhanced_analysis/ folder")
    
    def _generate_enhanced_text_report(self, timestamp):
        """Generate enhanced comprehensive text report"""
        analysis_folder = os.path.join(self.results_folder, 'enhanced_analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        report_filename = f'{analysis_folder}/ENHANCED_HYPOTHESIS_REPORT_{timestamp}.txt'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("🏆 ENHANCED COMPREHENSIVE HYPOTHESIS TESTING ANALYSIS 🏆\n")
            f.write("=" * 75 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Quality Check
            total_configs = len(self.data)
            total_strategies = sum(len(strategies) for strategies in self.strategies.values())
            total_benchmarks = sum(len(benchmarks) for benchmarks in self.benchmarks.values())
            
            # Count valid strategies
            valid_strategies = 0
            valid_benchmarks = 0
            
            for test_config, strategies in self.strategies.items():
                for strategy_name, strategy_data in strategies.items():
                    if 'total_return' in strategy_data:
                        valid_strategies += 1
            
            for test_config, benchmarks in self.benchmarks.items():
                for benchmark_name, benchmark_data in benchmarks.items():
                    if 'total_return' in benchmark_data:
                        valid_benchmarks += 1
            
            f.write("📊 EXECUTIVE SUMMARY\n")
            f.write("-" * 35 + "\n")
            f.write(f"Test Configurations Analyzed: {total_configs}\n")
            f.write(f"Our Trading Strategies: {total_strategies}\n")
            f.write(f"H1-H4 Benchmarks: {total_benchmarks}\n\n")
            
            if valid_strategies > 0 and valid_benchmarks > 0:
                # Hypothesis Validation
                our_avg = self._calculate_average_performance(self.strategies)
                benchmark_avg = self._calculate_average_performance(self.benchmarks)
                
                f.write("🧪 HYPOTHESIS VALIDATION RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Our Strategies Average Return: {our_avg:+.2f}%\n")
                f.write(f"H1-H4 Benchmarks Average: {benchmark_avg:+.2f}%\n")
                f.write(f"Performance Difference: {our_avg - benchmark_avg:+.2f}%\n")
                
                if our_avg > benchmark_avg:
                    f.write("✅ HYPOTHESIS VALIDATED: Our strategies outperform benchmarks!\n\n")
                else:
                    f.write("❌ HYPOTHESIS REJECTED: Benchmarks outperform our strategies\n\n")
            
            # Detailed Performance Analysis
            f.write("📈 DETAILED PERFORMANCE ANALYSIS\n")
            f.write("-" * 45 + "\n")
            
            for test_config, strategies in self.strategies.items():
                if strategies:
                    f.write(f"\n{test_config.upper()}:\n")
                    f.write("Our Strategies:\n")
                    
                    for strategy_name, strategy_data in strategies.items():
                        if 'total_return' in strategy_data:
                            return_pct = strategy_data['total_return'] * 100
                            trades = strategy_data.get('total_trades', 0)
                            win_rate = strategy_data.get('win_rate', 0) * 100
                            sharpe = strategy_data.get('sharpe_ratio', 0)
                            
                            f.write(f"  • {strategy_name}:\n")
                            f.write(f"    Return: {return_pct:+6.2f}% | Trades: {trades} | Win Rate: {win_rate:.1f}% | Sharpe: {sharpe:.3f}\n")
                    
                    # Benchmarks for this config
                    if test_config in self.benchmarks:
                        f.write("H1-H4 Benchmarks:\n")
                        for benchmark_name, benchmark_data in self.benchmarks[test_config].items():
                            if 'total_return' in benchmark_data:
                                return_pct = benchmark_data['total_return'] * 100
                                trades = benchmark_data.get('total_trades', 0)
                                win_rate = benchmark_data.get('win_rate', 0) * 100
                                
                                f.write(f"  • {benchmark_name}: {return_pct:+6.2f}% ({trades} trades, {win_rate:.1f}% win rate)\n")
            
            # Clear Winners
            f.write("\n\n🏅 CLEAR WINNERS BY CATEGORY\n")
            f.write("-" * 40 + "\n")
            
            winners = self._find_category_winners()
            categories = ['Best Overall Return', 'Best Risk-Adjusted (Sharpe)', 'Best Win Rate', 'Best Trade Efficiency']
            
            for category, winner in zip(categories, winners):
                if winner['name']:
                    winner_type = "🏆 OUR STRATEGY" if 'H' not in winner['name'] else "📊 BENCHMARK"
                    f.write(f"\n🥇 {category}: {winner_type}\n")
                    f.write(f"   Strategy: {winner['name']}\n")
                    f.write(f"   Value: {winner['value']:.3f}{'%' if 'Efficiency' not in category and 'Sharpe' not in category else ''}\n")
                    f.write(f"   Configuration: {winner['config']}\n")
            
            # Strategic Recommendations
            f.write("\n\n💡 ENHANCED STRATEGIC RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            f.write("Based on comprehensive hypothesis testing analysis:\n\n")
            
            our_avg = self._calculate_average_performance(self.strategies)
            benchmark_avg = self._calculate_average_performance(self.benchmarks)
            
            if our_avg > benchmark_avg:
                f.write("1. ✅ VALIDATION SUCCESS: Our trading strategies demonstrate superior performance\n")
                f.write("   → Continue development and refinement of current approach\n")
                f.write("   → Consider increasing position sizes for validated strategies\n\n")
            else:
                f.write("1. ⚠️  VALIDATION CHALLENGE: Benchmarks currently outperform our strategies\n")
                f.write("   → Analyze benchmark approaches for improvement opportunities\n")
                f.write("   → Focus on risk management and strategy optimization\n\n")
            
            f.write("2. 📊 TIME HORIZON OPTIMIZATION:\n")
            f.write("   → Longer testing periods (6+ months) provide more reliable results\n")
            f.write("   → Consider market cycle effects in strategy validation\n\n")
            
            f.write("3. 🎯 STRATEGY SELECTION CRITERIA:\n")
            f.write("   → Prioritize strategies with Sharpe ratio > 1.0\n")
            f.write("   → Target win rates ≥ 60% for consistent performance\n")
            f.write("   → Optimize for return per trade efficiency\n\n")
            
            f.write("4. 🧪 HYPOTHESIS TESTING INSIGHTS:\n")
            f.write("   → H1-H4 benchmarks provide valuable performance baselines\n")
            f.write("   → Multi-asset strategies show better risk management\n")
            f.write("   → Trade frequency optimization is critical for success\n\n")
            
            f.write("5. 🚀 NEXT STEPS:\n")
            f.write("   → Implement top-performing strategies in paper trading\n")
            f.write("   → Conduct additional out-of-sample testing\n")
            f.write("   → Monitor performance against evolving market conditions\n")
    
    def _export_enhanced_csv_data(self, timestamp):
        """Export enhanced CSV data for further analysis"""
        try:
            analysis_folder = os.path.join(self.results_folder, 'enhanced_analysis')
            os.makedirs(analysis_folder, exist_ok=True)

            # Export strategies data
            strategies_data = []
            for test_config, strategies in self.strategies.items():
                for strategy_name, strategy_data in strategies.items():
                    row = {
                        'Type': 'Strategy',
                        'Configuration': test_config,
                        'Name': strategy_name,
                        'Total_Return_Pct': strategy_data.get('total_return', 0) * 100,
                        'Sharpe_Ratio': strategy_data.get('sharpe_ratio', 0),
                        'Win_Rate_Pct': self._normalize_win_rate(strategy_data.get('win_rate', 0)),
                        'Total_Trades': strategy_data.get('total_trades', 0),
                        'Max_Drawdown_Pct': strategy_data.get('max_drawdown', 0) * 100,
                        'Volatility_Pct': strategy_data.get('volatility', 0) * 100
                    }
                    strategies_data.append(row)
            
            # Export benchmarks data
            benchmarks_data = []
            for test_config, benchmarks in self.benchmarks.items():
                for benchmark_name, benchmark_data in benchmarks.items():
                    row = {
                        'Type': 'Benchmark',
                        'Configuration': test_config,
                        'Name': benchmark_name,
                        'Total_Return_Pct': benchmark_data.get('total_return', 0) * 100,
                        'Sharpe_Ratio': benchmark_data.get('sharpe_ratio', 0),
                        'Win_Rate_Pct': self._normalize_win_rate(benchmark_data.get('win_rate', 0)),
                        'Total_Trades': benchmark_data.get('total_trades', 0),
                        'Max_Drawdown_Pct': benchmark_data.get('max_drawdown', 0) * 100,
                        'Volatility_Pct': benchmark_data.get('volatility', 0) * 100
                    }
                    benchmarks_data.append(row)
            
            # Save to CSV files
            if strategies_data:
                strategies_df = pd.DataFrame(strategies_data)
                strategies_file = f'{analysis_folder}/enhanced_strategies_analysis_{timestamp}.csv'
                strategies_df.to_csv(strategies_file, index=False)
                print(f"   📊 Strategies data: {strategies_file}")
            
            if benchmarks_data:
                benchmarks_df = pd.DataFrame(benchmarks_data)
                benchmarks_file = f'{analysis_folder}/enhanced_benchmarks_analysis_{timestamp}.csv'
                benchmarks_df.to_csv(benchmarks_file, index=False)
                print(f"   📊 Benchmarks data: {benchmarks_file}")
            
            # Combined data
            if strategies_data and benchmarks_data:
                combined_data = strategies_data + benchmarks_data
                combined_df = pd.DataFrame(combined_data)
                combined_file = f'{analysis_folder}/enhanced_combined_analysis_{timestamp}.csv'
                combined_df.to_csv(combined_file, index=False)
                print(f"   📊 Combined analysis: {combined_file}")
                
        except Exception as e:
            print(f"❌ Error exporting CSV data: {e}")
    
    def run_enhanced_analysis(self):
        """Run complete enhanced hypothesis testing analysis with comprehensive validation"""
        print("🚀 Starting Enhanced Hypothesis Testing Analysis...")
        
        if not self.load_all_results():
            print("❌ No results to analyze")
            return
        
        # Comprehensive data validation
        print("\n🔍 Performing comprehensive data validation...")
        
        valid_strategies = 0
        valid_benchmarks = 0
        
        for test_config, strategies in self.strategies.items():
            for strategy_name, strategy_data in strategies.items():
                if 'total_return' in strategy_data:
                    valid_strategies += 1
        
        for test_config, benchmarks in self.benchmarks.items():
            for benchmark_name, benchmark_data in benchmarks.items():
                if 'total_return' in benchmark_data:
                    valid_benchmarks += 1
        
        print(f"📊 Validation Results:")
        print(f"   ✅ Valid Strategies: {valid_strategies}")
        print(f"   ✅ Valid Benchmarks: {valid_benchmarks}")
        
        if valid_strategies == 0 and valid_benchmarks == 0:
            print("\n❌ CRITICAL ERROR: No valid performance data found!")
            print("🔧 Check that result files contain 'total_return' metrics")
            return
        
        print(f"\n✅ Proceeding with analysis of {valid_strategies + valid_benchmarks} valid entries")
        
        # Proceed with analysis
        self.create_enhanced_comprehensive_analysis()
        print("\n🎉 Enhanced hypothesis testing analysis complete!")
        print(f"📁 Check {self.results_folder}/ for comprehensive results")


def main():
    """Run enhanced comprehensive hypothesis testing analysis"""
    analyzer = EnhancedHypothesisAnalyzer()
    analyzer.run_enhanced_analysis()


if __name__ == "__main__":
    main()