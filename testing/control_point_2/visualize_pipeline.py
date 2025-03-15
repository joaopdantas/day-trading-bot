"""
Data Pipeline Visualization for Control Point 2.

This script creates a visual representation of the entire data pipeline.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def create_pipeline_visualization():
    """Create a visual representation of the MakesALot data pipeline."""
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    
    # Title
    fig.suptitle('MakesALot Trading Bot - Data Pipeline (Control Point 2)', fontsize=16, fontweight='bold')
    
    # Create grid for components
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1.5, 1])
    
    # Define colors
    colors = {
        'data': '#4285F4',      # Google Blue
        'process': '#34A853',   # Google Green
        'analysis': '#FBBC05',  # Google Yellow
        'storage': '#EA4335',   # Google Red
        'arrow': '#777777',     # Arrow gray
        'background': '#F6F6F6' # Light gray background
    }
    
    # Set background color
    fig.patch.set_facecolor('white')
    
    # Data Sources - First row, first column
    ax1 = fig.add_subplot(gs[0, 0])
    draw_component(ax1, 'Data Sources', [
        'Alpha Vantage API',
        'Yahoo Finance API',
        'Future: Direct Exchange APIs'
    ], colors['data'])
    
    # Data Fetcher - First row, second column
    ax2 = fig.add_subplot(gs[0, 1])
    draw_component(ax2, 'Data Fetcher', [
        'Historical Data Retrieval',
        'Interval Selection',
        'Date Range Filtering',
        'Error Handling'
    ], colors['data'])
    
    # Data Preprocessor - First row, third column
    ax3 = fig.add_subplot(gs[0, 2])
    draw_component(ax3, 'Data Preprocessor', [
        'Cleaning',
        'Normalization',
        'Feature Engineering',
        'Time Features'
    ], colors['process'])
    
    # Technical Indicators - Second row, first column
    ax4 = fig.add_subplot(gs[1, 0])
    draw_component(ax4, 'Technical Indicators', [
        'RSI',
        'MACD',
        'Bollinger Bands',
        'Moving Averages',
        'Volume Indicators',
        'Volatility Metrics'
    ], colors['analysis'])
    
    # Pattern Recognition - Second row, second column
    ax5 = fig.add_subplot(gs[1, 1])
    draw_component(ax5, 'Pattern Recognition', [
        'Candlestick Patterns',
        'Support & Resistance',
        'Trend Analysis',
        'Chart Patterns'
    ], colors['analysis'])
    
    # Signal Generation - Second row, third column
    ax6 = fig.add_subplot(gs[1, 2])
    draw_component(ax6, 'Signal Generation', [
        'Buy/Sell Signals',
        'Strategy Combination',
        'Risk Analysis',
        'Confidence Levels'
    ], colors['analysis'])
    
    # Data Storage - Third row, span all columns
    ax7 = fig.add_subplot(gs[2, :])
    draw_component(ax7, 'MongoDB Storage', [
        'Historical Data Collection',
        'Pattern Storage',
        'Signal Archive',
        'Performance Metrics',
        'Query Optimization',
        'Data Versioning'
    ], colors['storage'], wider=True)
    
    # Add arrows connecting components
    add_arrow(fig, ax1, ax2)
    add_arrow(fig, ax2, ax3)
    add_arrow(fig, ax3, ax4)
    add_arrow(fig, ax4, ax5)
    add_arrow(fig, ax5, ax6)
    
    # Add arrows to storage
    add_arrow(fig, ax3, ax7, vertical=True)
    add_arrow(fig, ax6, ax7, vertical=True)
    
    # Remove axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.axis('off')
    
    # Add legend for progress
    add_legend(fig)
    
    # Add watermark
    fig.text(0.5, 0.02, 'MakesALot Trading Bot - Control Point 2 - April 2025', 
             ha='center', color='gray', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create directory if it doesn't exist
    os.makedirs('testing/control_point_2', exist_ok=True)
    
    # Save figure
    plt.savefig('testing/control_point_2/data_pipeline_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Pipeline visualization saved to testing/control_point_2/data_pipeline_visualization.png")


def draw_component(ax, title, features, color, wider=False):
    """Draw a component box with title and features."""
    # Component box
    width = 0.8 if not wider else 0.9
    height = 0.8
    rect = Rectangle((0.1, 0.1), width, height, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    
    # Title
    ax.text(0.5, 0.85, title, horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Features
    y_pos = 0.75
    step = 0.1 if len(features) <= 5 else 0.08
    
    for feature in features:
        ax.text(0.15, y_pos, '• ' + feature, fontsize=9)
        y_pos -= step


def add_arrow(fig, from_ax, to_ax, vertical=False):
    """Add an arrow connecting two components."""
    # Get axes positions
    from_pos = from_ax.get_position()
    to_pos = to_ax.get_position()
    
    if vertical:
        # Vertical arrow (top to bottom)
        x_from = from_pos.x0 + from_pos.width / 2
        y_from = from_pos.y0
        x_to = to_pos.x0 + to_pos.width / 2
        y_to = to_pos.y0 + to_pos.height
    else:
        # Horizontal arrow (left to right)
        x_from = from_pos.x0 + from_pos.width
        y_from = from_pos.y0 + from_pos.height / 2
        x_to = to_pos.x0
        y_to = to_pos.y0 + to_pos.height / 2
    
    # Create arrow
    arrow = FancyArrowPatch((x_from, y_from), (x_to, y_to),
                          arrowstyle='-|>', color='#555555',
                          linewidth=1.5, mutation_scale=15,
                          connectionstyle='arc3,rad=0.1')
    
    # Add arrow to figure
    fig.add_artist(arrow)


def add_legend(fig):
    """Add a legend showing the different component types."""
    # Legend items
    legend_items = [
        ('Data Acquisition', '#4285F4'),
        ('Data Processing', '#34A853'),
        ('Analysis Engine', '#FBBC05'),
        ('Storage System', '#EA4335')
    ]
    
    # Position and size of legend
    x, y = 0.01, 0.01
    width, height = 0.15, 0.05
    
    # Add legend items
    for i, (label, color) in enumerate(legend_items):
        # Calculate position
        xi = x + i * width
        
        # Draw rectangle
        rect = Rectangle((xi, y), 0.03, height, facecolor=color, alpha=0.3, edgecolor=color)
        fig.add_artist(rect)
        
        # Add label
        fig.text(xi + 0.035, y + height/2, label, fontsize=8, va='center')


if __name__ == "__main__":
    create_pipeline_visualization()