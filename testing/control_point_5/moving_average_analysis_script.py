"""
MOVING AVERAGE ANALYSIS SCRIPT (CP1 Enhancement)
20-day vs 50-day moving average comparison with proper delays
Non-overlapping line analysis and long-term decision support system
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

try:
    from src.data.fetcher import get_data_api
    from src.indicators.technical import TechnicalIndicators
    PROJECT_AVAILABLE = True
    print("✅ Your project modules loaded successfully")
except ImportError as e:
    print(f"❌ Project modules not available: {e}")
    PROJECT_AVAILABLE = False
    # Fallback imports
    import yfinance as yf


class MovingAverageAnalyzer:
    """
    Enhanced Moving Average Analysis with proper delays and decision support
    """
    
    def __init__(self, symbol="MSFT", period_days=500):
        self.symbol = symbol
        self.period_days = period_days
        self.data = None
        self.ma_analysis = {}
        
    def load_data(self):
        """Load historical data using project's data fetcher or fallback"""
        
        print(f"📊 Loading data for {self.symbol}...")
        
        if PROJECT_AVAILABLE:
            try:
                # Use your project's data fetcher
                api = get_data_api("yahoo_finance")
                self.data = api.fetch_historical_data(self.symbol, "1d")
                
                if self.data is None or self.data.empty:
                    print("⚠️ Yahoo Finance failed, trying Alpha Vantage...")
                    api = get_data_api("alpha_vantage")
                    self.data = api.fetch_historical_data(self.symbol, "1d")
                
                self.data = self.data.tail(self.period_days)
                print(f"✅ Loaded {len(self.data)} days using project methods")
                
            except Exception as e:
                print(f"⚠️ Project data fetcher failed: {e}")
                self._load_data_fallback()
        else:
            self._load_data_fallback()
    
    def _load_data_fallback(self):
        """Fallback data loading using yfinance"""
        
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period="2y")  # Get 2 years of data
            
            if not self.data.empty:
                # Rename columns to match project format
                self.data.columns = [col.lower() for col in self.data.columns]
                self.data = self.data.tail(self.period_days)
                print(f"✅ Loaded {len(self.data)} days using Yahoo Finance fallback")
            else:
                raise Exception("No data retrieved")
                
        except Exception as e:
            print(f"❌ Fallback data loading failed: {e}")
            return False
        
        return True
    
    def calculate_moving_averages(self):
        """Calculate 20-day and 50-day moving averages with proper delays"""
        
        if self.data is None:
            print("❌ No data available for MA calculation")
            return
        
        print("📈 Calculating Moving Averages with proper delays...")
        
        # Calculate moving averages
        self.data['sma_20'] = self.data['close'].rolling(window=20, min_periods=20).mean()
        self.data['sma_50'] = self.data['close'].rolling(window=50, min_periods=50).mean()
        
        # Create proper delays to avoid look-ahead bias
        # Shift signals by 1 day to simulate real-world trading
        self.data['sma_20_delayed'] = self.data['sma_20'].shift(1)
        self.data['sma_50_delayed'] = self.data['sma_50'].shift(1)
        
        # Calculate crossover signals with delays
        self.data['ma_signal'] = 0
        self.data.loc[self.data['sma_20_delayed'] > self.data['sma_50_delayed'], 'ma_signal'] = 1  # Bullish
        self.data.loc[self.data['sma_20_delayed'] < self.data['sma_50_delayed'], 'ma_signal'] = -1  # Bearish
        
        # Identify non-overlapping periods
        self.data['lines_overlapping'] = abs(self.data['sma_20'] - self.data['sma_50']) / self.data['close'] < 0.005  # Within 0.5%
        
        print("✅ Moving averages calculated with proper delays")
    
    def detect_crossover_events(self):
        """Detect crossover events and analyze their significance"""
        
        print("🔍 Detecting crossover events...")
        
        crossovers = []
        
        # Find crossover points (where signal changes)
        for i in range(1, len(self.data)):
            current_signal = self.data['ma_signal'].iloc[i]
            prev_signal = self.data['ma_signal'].iloc[i-1]
            
            if current_signal != prev_signal and current_signal != 0:
                
                # Calculate signal strength based on separation
                ma_separation = abs(self.data['sma_20'].iloc[i] - self.data['sma_50'].iloc[i]) / self.data['close'].iloc[i]
                volume_factor = self.data['volume'].iloc[i] / self.data['volume'].rolling(20).mean().iloc[i]
                
                crossover = {
                    'date': self.data.index[i],
                    'price': self.data['close'].iloc[i],
                    'signal': 'BULLISH' if current_signal == 1 else 'BEARISH',
                    'sma_20': self.data['sma_20'].iloc[i],
                    'sma_50': self.data['sma_50'].iloc[i],
                    'separation_pct': ma_separation * 100,
                    'volume_factor': volume_factor,
                    'signal_strength': self._calculate_signal_strength(ma_separation, volume_factor),
                    'lines_overlapping': self.data['lines_overlapping'].iloc[i]
                }
                
                crossovers.append(crossover)
        
        self.ma_analysis['crossovers'] = crossovers
        print(f"✅ Found {len(crossovers)} crossover events")
        
        return crossovers
    
    def _calculate_signal_strength(self, separation, volume_factor):
        """Calculate signal strength based on MA separation and volume"""
        
        # Base strength from separation
        separation_score = min(separation * 200, 1.0)  # Cap at 1.0
        
        # Volume confirmation
        volume_score = min(volume_factor / 2, 1.0)  # Cap at 1.0
        
        # Combined strength (weighted average)
        strength = (separation_score * 0.7) + (volume_score * 0.3)
        
        if strength > 0.8:
            return 'STRONG'
        elif strength > 0.5:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def analyze_trend_periods(self):
        """Analyze different trend periods and their characteristics"""
        
        print("📊 Analyzing trend periods...")
        
        # Identify trend periods
        trend_periods = []
        current_trend = None
        trend_start = None
        
        for i, row in self.data.iterrows():
            signal = row['ma_signal']
            
            if signal != current_trend:
                # End previous trend
                if current_trend is not None and trend_start is not None:
                    trend_periods.append({
                        'start_date': trend_start,
                        'end_date': i,
                        'trend': 'BULLISH' if current_trend == 1 else 'BEARISH',
                        'duration_days': (i - trend_start).days,
                        'start_price': self.data.loc[trend_start, 'close'],
                        'end_price': row['close'],
                        'return': (row['close'] - self.data.loc[trend_start, 'close']) / self.data.loc[trend_start, 'close']
                    })
                
                # Start new trend
                current_trend = signal
                trend_start = i
        
        # Add final trend period
        if current_trend is not None and trend_start is not None:
            trend_periods.append({
                'start_date': trend_start,
                'end_date': self.data.index[-1],
                'trend': 'BULLISH' if current_trend == 1 else 'BEARISH',
                'duration_days': (self.data.index[-1] - trend_start).days,
                'start_price': self.data.loc[trend_start, 'close'],
                'end_price': self.data['close'].iloc[-1],
                'return': (self.data['close'].iloc[-1] - self.data.loc[trend_start, 'close']) / self.data.loc[trend_start, 'close']
            })
        
        self.ma_analysis['trend_periods'] = trend_periods
        
        # Calculate trend statistics
        bullish_periods = [t for t in trend_periods if t['trend'] == 'BULLISH']
        bearish_periods = [t for t in trend_periods if t['trend'] == 'BEARISH']
        
        self.ma_analysis['trend_stats'] = {
            'total_periods': len(trend_periods),
            'bullish_periods': len(bullish_periods),
            'bearish_periods': len(bearish_periods),
            'avg_bullish_duration': np.mean([t['duration_days'] for t in bullish_periods]) if bullish_periods else 0,
            'avg_bearish_duration': np.mean([t['duration_days'] for t in bearish_periods]) if bearish_periods else 0,
            'avg_bullish_return': np.mean([t['return'] for t in bullish_periods]) if bullish_periods else 0,
            'avg_bearish_return': np.mean([t['return'] for t in bearish_periods]) if bearish_periods else 0,
            'bullish_success_rate': len([t for t in bullish_periods if t['return'] > 0]) / len(bullish_periods) if bullish_periods else 0,
            'bearish_success_rate': len([t for t in bearish_periods if t['return'] < 0]) / len(bearish_periods) if bearish_periods else 0
        }
        
        print(f"✅ Analyzed {len(trend_periods)} trend periods")
    
    def generate_long_term_decision_support(self):
        """Generate long-term decision support based on MA analysis"""
        
        print("🎯 Generating long-term decision support...")
        
        current_signal = self.data['ma_signal'].iloc[-1]
        current_price = self.data['close'].iloc[-1]
        sma_20_current = self.data['sma_20'].iloc[-1]
        sma_50_current = self.data['sma_50'].iloc[-1]
        
        # Current trend analysis
        current_trend = 'BULLISH' if current_signal == 1 else 'BEARISH' if current_signal == -1 else 'NEUTRAL'
        
        # Price position relative to MAs
        price_vs_sma20 = (current_price - sma_20_current) / sma_20_current
        price_vs_sma50 = (current_price - sma_50_current) / sma_50_current
        
        # Recent trend strength
        recent_data = self.data.tail(20)
        trend_consistency = abs(recent_data['ma_signal'].mean())
        
        # Support/Resistance levels
        support_level = min(sma_20_current, sma_50_current)
        resistance_level = max(sma_20_current, sma_50_current)
        
        # Decision recommendations
        recommendations = []
        confidence_score = 0
        
        if current_trend == 'BULLISH':
            if price_vs_sma20 > 0 and price_vs_sma50 > 0:
                recommendations.append("Strong bullish trend - price above both MAs")
                confidence_score += 0.3
            
            if trend_consistency > 0.7:
                recommendations.append("Consistent trend - high probability continuation")
                confidence_score += 0.2
            
            recommendations.append(f"Long-term support at ${support_level:.2f}")
            recommendations.append("Consider: Buy on dips to MA support")
            
        elif current_trend == 'BEARISH':
            if price_vs_sma20 < 0 and price_vs_sma50 < 0:
                recommendations.append("Strong bearish trend - price below both MAs")
                confidence_score += 0.3
            
            if trend_consistency > 0.7:
                recommendations.append("Consistent downtrend - exercise caution")
                confidence_score += 0.2
            
            recommendations.append(f"Long-term resistance at ${resistance_level:.2f}")
            recommendations.append("Consider: Sell on rallies to MA resistance")
        
        else:
            recommendations.append("Neutral/Consolidation phase")
            recommendations.append("Wait for clear breakout above/below MAs")
            confidence_score += 0.1
        
        # Add timing considerations
        if self.data['lines_overlapping'].tail(5).sum() > 2:
            recommendations.append("WARNING: MAs converging - trend change possible")
            confidence_score -= 0.1
        
        # Risk management
        stop_loss_level = support_level * 0.95 if current_trend == 'BULLISH' else resistance_level * 1.05
        recommendations.append(f"Suggested stop-loss level: ${stop_loss_level:.2f}")
        
        decision_support = {
            'current_trend': current_trend,
            'trend_strength': trend_consistency,
            'confidence_score': max(0, min(1, confidence_score)),
            'price_position': {
                'vs_sma20': price_vs_sma20,
                'vs_sma50': price_vs_sma50
            },
            'key_levels': {
                'support': support_level,
                'resistance': resistance_level,
                'stop_loss': stop_loss_level
            },
            'recommendations': recommendations,
            'lines_overlapping': self.data['lines_overlapping'].iloc[-1]
        }
        
        self.ma_analysis['decision_support'] = decision_support
        print("✅ Long-term decision support generated")
        
        return decision_support
    
    def create_visualizations(self):
        """Create comprehensive MA analysis visualizations"""
        
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Moving Average Analysis - {self.symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data['close'], label='Price', linewidth=1, alpha=0.8)
        ax1.plot(self.data.index, self.data['sma_20'], label='SMA 20', linewidth=2, color='orange')
        ax1.plot(self.data.index, self.data['sma_50'], label='SMA 50', linewidth=2, color='blue')
        
        # Highlight crossover points
        crossovers = self.ma_analysis.get('crossovers', [])
        for crossover in crossovers:
            color = 'green' if crossover['signal'] == 'BULLISH' else 'red'
            marker = '^' if crossover['signal'] == 'BULLISH' else 'v'
            ax1.scatter(crossover['date'], crossover['price'], color=color, marker=marker, s=100, alpha=0.8)
        
        ax1.set_title('Price with Moving Averages & Crossovers')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MA Separation
        ax2 = axes[0, 1]
        ma_spread = ((self.data['sma_20'] - self.data['sma_50']) / self.data['close'] * 100).dropna()
        ax2.plot(ma_spread.index, ma_spread, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(ma_spread.index, ma_spread, 0, 
                        where=(ma_spread > 0), color='green', alpha=0.3, label='Bullish')
        ax2.fill_between(ma_spread.index, ma_spread, 0, 
                        where=(ma_spread < 0), color='red', alpha=0.3, label='Bearish')
        ax2.set_title('MA Separation (SMA20 - SMA50) %')
        ax2.set_ylabel('Separation %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trend Periods
        ax3 = axes[1, 0]
        trend_periods = self.ma_analysis.get('trend_periods', [])
        y_pos = 1
        
        for period in trend_periods:
            color = 'green' if period['trend'] == 'BULLISH' else 'red'
            ax3.barh(y_pos, period['duration_days'], left=0, color=color, alpha=0.6)
            ax3.text(period['duration_days']/2, y_pos, 
                    f"{period['trend'][:4]}\n{period['return']:.1%}", 
                    ha='center', va='center', fontsize=8)
            y_pos += 1
        
        ax3.set_title('Trend Periods Duration & Returns')
        ax3.set_xlabel('Duration (Days)')
        ax3.set_ylabel('Trend Period')
        
        # Plot 4: Signal Strength Distribution
        ax4 = axes[1, 1]
        if crossovers:
            strengths = [c['signal_strength'] for c in crossovers]
            strength_counts = {s: strengths.count(s) for s in ['WEAK', 'MODERATE', 'STRONG']}
            
            bars = ax4.bar(strength_counts.keys(), strength_counts.values(), 
                          color=['red', 'orange', 'green'], alpha=0.7)
            ax4.set_title('Crossover Signal Strength Distribution')
            ax4.set_ylabel('Count')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Plot 5: Recent MA Analysis (Last 60 days)
        ax5 = axes[2, 0]
        recent_data = self.data.tail(60)
        ax5.plot(recent_data.index, recent_data['close'], label='Price', linewidth=2)
        ax5.plot(recent_data.index, recent_data['sma_20'], label='SMA 20', linewidth=2)
        ax5.plot(recent_data.index, recent_data['sma_50'], label='SMA 50', linewidth=2)
        
        # Highlight overlapping periods
        overlapping_periods = recent_data[recent_data['lines_overlapping']]
        if not overlapping_periods.empty:
            ax5.scatter(overlapping_periods.index, overlapping_periods['close'], 
                       color='yellow', marker='o', s=30, alpha=0.7, label='Lines Overlapping')
        
        ax5.set_title('Recent Analysis (Last 60 Days)')
        ax5.set_ylabel('Price ($)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Decision Support Summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        decision_support = self.ma_analysis.get('decision_support', {})
        if decision_support:
            summary_text = f"""
DECISION SUPPORT SUMMARY

Current Trend: {decision_support['current_trend']}
Confidence: {decision_support['confidence_score']:.2f}

Key Levels:
- Support: ${decision_support['key_levels']['support']:.2f}
- Resistance: ${decision_support['key_levels']['resistance']:.2f}
- Stop Loss: ${decision_support['key_levels']['stop_loss']:.2f}

Top Recommendations:
"""
            for i, rec in enumerate(decision_support['recommendations'][:3]):
                summary_text += f"• {rec}\n"
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        results_dir = os.path.join(os.path.dirname(__file__), 'moving_average_results')
        os.makedirs(results_dir, exist_ok=True)
        
        plot_file = os.path.join(results_dir, f'ma_analysis_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
    
    def generate_analysis_report(self):
        """Generate comprehensive MA analysis report"""
        
        print("📄 Generating analysis report...")
        
        results_dir = os.path.join(os.path.dirname(__file__), 'moving_average_results')
        os.makedirs(results_dir, exist_ok=True)
        
        report_file = os.path.join(results_dir, f'ma_analysis_report_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_file, 'w') as f:
            f.write("MOVING AVERAGE ANALYSIS REPORT (CP1 Enhancement)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Analysis Period: {self.data.index[0].date()} to {self.data.index[-1].date()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Crossover Analysis
            f.write("CROSSOVER EVENTS ANALYSIS\n")
            f.write("-" * 40 + "\n")
            crossovers = self.ma_analysis.get('crossovers', [])
            f.write(f"Total Crossovers: {len(crossovers)}\n\n")
            
            for i, crossover in enumerate(crossovers, 1):
                f.write(f"{i}. {crossover['signal']} CROSSOVER\n")
                f.write(f"   Date: {crossover['date'].strftime('%Y-%m-%d')}\n")
                f.write(f"   Price: ${crossover['price']:.2f}\n")
                f.write(f"   SMA 20: ${crossover['sma_20']:.2f}\n")
                f.write(f"   SMA 50: ${crossover['sma_50']:.2f}\n")
                f.write(f"   Separation: {crossover['separation_pct']:.2f}%\n")
                f.write(f"   Signal Strength: {crossover['signal_strength']}\n")
                f.write(f"   Volume Factor: {crossover['volume_factor']:.2f}\n")
                f.write(f"   Lines Overlapping: {crossover['lines_overlapping']}\n\n")
            
            # Trend Statistics
            f.write("TREND PERIOD STATISTICS\n")
            f.write("-" * 40 + "\n")
            trend_stats = self.ma_analysis.get('trend_stats', {})
            f.write(f"Total Trend Periods: {trend_stats.get('total_periods', 0)}\n")
            f.write(f"Bullish Periods: {trend_stats.get('bullish_periods', 0)}\n")
            f.write(f"Bearish Periods: {trend_stats.get('bearish_periods', 0)}\n")
            f.write(f"Average Bullish Duration: {trend_stats.get('avg_bullish_duration', 0):.1f} days\n")
            f.write(f"Average Bearish Duration: {trend_stats.get('avg_bearish_duration', 0):.1f} days\n")
            f.write(f"Average Bullish Return: {trend_stats.get('avg_bullish_return', 0):.2%}\n")
            f.write(f"Average Bearish Return: {trend_stats.get('avg_bearish_return', 0):.2%}\n")
            f.write(f"Bullish Success Rate: {trend_stats.get('bullish_success_rate', 0):.2%}\n")
            f.write(f"Bearish Success Rate: {trend_stats.get('bearish_success_rate', 0):.2%}\n\n")
            
            # Decision Support
            f.write("LONG-TERM DECISION SUPPORT\n")
            f.write("-" * 40 + "\n")
            decision_support = self.ma_analysis.get('decision_support', {})
            if decision_support:
                f.write(f"Current Trend: {decision_support['current_trend']}\n")
                f.write(f"Trend Strength: {decision_support['trend_strength']:.2f}\n")
                f.write(f"Confidence Score: {decision_support['confidence_score']:.2f}\n\n")
                
                f.write("KEY LEVELS:\n")
                f.write(f"  Support: ${decision_support['key_levels']['support']:.2f}\n")
                f.write(f"  Resistance: ${decision_support['key_levels']['resistance']:.2f}\n")
                f.write(f"  Stop Loss: ${decision_support['key_levels']['stop_loss']:.2f}\n\n")
                
                f.write("RECOMMENDATIONS:\n")
                for i, rec in enumerate(decision_support['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")
                f.write("\n")
            
            # Technical Summary
            f.write("TECHNICAL SUMMARY\n")
            f.write("-" * 40 + "\n")
            current_price = self.data['close'].iloc[-1]
            sma_20_current = self.data['sma_20'].iloc[-1]
            sma_50_current = self.data['sma_50'].iloc[-1]
            
            f.write(f"Current Price: ${current_price:.2f}\n")
            f.write(f"Current SMA 20: ${sma_20_current:.2f}\n")
            f.write(f"Current SMA 50: ${sma_50_current:.2f}\n")
            f.write(f"Price vs SMA 20: {((current_price - sma_20_current) / sma_20_current):.2%}\n")
            f.write(f"Price vs SMA 50: {((current_price - sma_50_current) / sma_50_current):.2%}\n")
            f.write(f"SMA 20 vs SMA 50: {((sma_20_current - sma_50_current) / sma_50_current):.2%}\n")
        
        print(f"📄 Report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run complete moving average analysis"""
        
        print("🚀 MOVING AVERAGE ANALYSIS (CP1 Enhancement)")
        print("=" * 60)
        print("20-day vs 50-day analysis with proper delays and decision support")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return
        
        # Run analysis steps
        self.calculate_moving_averages()
        self.detect_crossover_events()
        self.analyze_trend_periods()
        self.generate_long_term_decision_support()
        
        # Generate outputs
        self.create_visualizations()
        self.generate_analysis_report()
        
        print("\n✅ Moving Average Analysis completed!")
        print("📁 Check 'moving_average_results' folder for outputs")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print analysis summary"""
        
        print("\n" + "=" * 60)
        print("📊 MOVING AVERAGE ANALYSIS SUMMARY")
        print("=" * 60)
        
        crossovers = self.ma_analysis.get('crossovers', [])
        trend_stats = self.ma_analysis.get('trend_stats', {})
        decision_support = self.ma_analysis.get('decision_support', {})
        
        print(f"📈 Total Crossovers Detected: {len(crossovers)}")
        print(f"📊 Trend Periods Analyzed: {trend_stats.get('total_periods', 0)}")
        print(f"🎯 Current Trend: {decision_support.get('current_trend', 'Unknown')}")
        print(f"📊 Confidence Score: {decision_support.get('confidence_score', 0):.2f}")
        
        if crossovers:
            recent_crossover = crossovers[-1]
            days_since = (datetime.now().date() - recent_crossover['date'].date()).days
            print(f"🔄 Last Crossover: {recent_crossover['signal']} ({days_since} days ago)")
            print(f"💪 Signal Strength: {recent_crossover['signal_strength']}")
        
        print(f"✅ Analysis complete - check results folder for detailed outputs")


def main():
    """Main function to run moving average analysis"""
    
    # You can change the symbol here
    symbol = input("Enter symbol to analyze (default: MSFT): ").strip().upper() or "MSFT"
    
    # Initialize analyzer
    analyzer = MovingAverageAnalyzer(symbol=symbol, period_days=500)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()