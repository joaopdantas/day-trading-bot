"""
Final Data Pipeline Visualization for Control Point 2.

This script creates a professional visual representation of the MakesALot
data pipeline with proper text spacing and no overlaps.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

def create_pipeline_visualization():
    """Create a clean, professional visualization of the data pipeline."""
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Set up the plot
    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 600)
    ax.axis('off')
    
    # Define colors
    colors = {
        'data': '#A8D5FF',      # Light blue
        'process': '#A8E6B8',   # Light green
        'analysis': '#FFE699',  # Light gold
        'storage': '#FFBF86',   # Light orange
        'main_flow': '#333333', # Dark gray for main flow
        'storage_flow': '#777777', # Medium gray for storage
        'shadow': '#00000022'   # Shadow color
    }
    
    # Define component positions and sizes
    components = [
        # Row 1 - Data acquisition and preprocessing
        {'name': 'Data Sources', 'x': 30, 'y': 450, 'width': 250, 'height': 125, 'color': colors['data'],
         'features': [
             '• Alpha Vantage API',
             '• Yahoo Finance API',
             '• Future: Direct Exchange APIs'
         ]},
        
        {'name': 'Data Fetcher', 'x': 380, 'y': 450, 'width': 250, 'height': 125, 'color': colors['data'],
         'features': [
             '• Historical Data Retrieval',
             '• Interval Selection',
             '• Date Range Filtering',
             '• Error Handling'
         ]},
        
        {'name': 'Data Preprocessor', 'x': 730, 'y': 450, 'width': 250, 'height': 125, 'color': colors['process'],
         'features': [
             '• Cleaning',
             '• Normalization',
             '• Feature Engineering',
             '• Time Features'
         ]},
        
        # Row 2 - Analysis components
        {'name': 'Technical Indicators', 'x': 20, 'y': 250, 'width': 320, 'height': 125, 'color': colors['analysis'],
         'features_col1': [
             '• RSI',
             '• MACD',
             '• Bollinger Bands'
         ],
         'features_col2': [
             '• Moving Averages',
             '• Volume Indicators',
             '• Volatility Metrics'
         ]},
        
        {'name': 'Pattern Recognition', 'x': 380, 'y': 250, 'width': 250, 'height': 125, 'color': colors['analysis'],
         'features': [
             '• Candlestick Patterns',
             '• Support & Resistance',
             '• Trend Analysis',
             '• Chart Patterns'
         ]},
        
        {'name': 'Signal Generation', 'x': 730, 'y': 250, 'width': 250, 'height': 125, 'color': colors['analysis'],
         'features': [
             '• Buy/Sell Signals',
             '• Strategy Combination',
             '• Risk Analysis',
             '• Confidence Levels'
         ]},
        
        # Row 3 - Storage component (wider)
        {'name': 'MongoDB Storage', 'x': 280, 'y': 50, 'width': 450, 'height': 125, 'color': colors['storage'],
         'features_col1': [
             '• Historical Data Collection',
             '• Pattern Storage',
             '• Signal Archive'
         ],
         'features_col2': [
             '• Performance Metrics',
             '• Query Optimization',
             '• Data Versioning'
         ]}
    ]
    
    # Draw all components
    for component in components:
        draw_component(ax, component)
    
    # Define connections between components
    connections = [
        # Main flow connections
        {'start': 'Data Sources', 'end': 'Data Fetcher', 'type': 'main'},
        {'start': 'Data Fetcher', 'end': 'Data Preprocessor', 'type': 'main'},
        {'start': 'Data Preprocessor', 'end': 'Technical Indicators', 'type': 'main'},
        {'start': 'Technical Indicators', 'end': 'Pattern Recognition', 'type': 'main'},
        {'start': 'Pattern Recognition', 'end': 'Signal Generation', 'type': 'main'},
        
        # Storage connections
        {'start': 'Data Preprocessor', 'end': 'MongoDB Storage', 'type': 'storage'},
        {'start': 'Technical Indicators', 'end': 'MongoDB Storage', 'type': 'storage'},
        {'start': 'Pattern Recognition', 'end': 'MongoDB Storage', 'type': 'storage'},
        {'start': 'Signal Generation', 'end': 'MongoDB Storage', 'type': 'storage'}
    ]
    
    # Draw all connections
    for connection in connections:
        draw_connection(ax, connection, components, colors)
    
    # Add title and description
    plt.figtext(0.5, 0.95, "MakesALot Trading Bot - Data Pipeline (Control Point 2)", 
              fontsize=20, fontweight='bold', ha='center')
    
    plt.figtext(0.5, 0.88, "Data Flow: Market data is fetched from APIs, preprocessed, analyzed using technical indicators,\n"
                "patterns are identified, trading signals are generated, and all information is stored in MongoDB.", 
                fontsize=12, ha='center')
    
    # Add legend
    add_legend(ax, colors)
    
    # Add watermark
    plt.figtext(0.5, 0.01, "MakesALot Trading Bot - Control Point 2 - April 2025", 
                ha='center', fontsize=9, color='gray')
    
    # Save the visualization
    os.makedirs('testing/control_point_2', exist_ok=True)
    plt.savefig('testing/control_point_2/data_pipeline_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Final pipeline visualization saved to testing/control_point_2/data_pipeline_visualization.png")


def draw_component(ax, component):
    """Draw a component box with title and features."""
    # Extract component properties
    name = component['name']
    x = component['x']
    y = component['y']
    width = component['width']
    height = component['height']
    color = component['color']
    
    # Draw shadow for 3D effect
    shadow = Rectangle((x+5, y-5), width, height, facecolor=colors['shadow'], zorder=1)
    ax.add_patch(shadow)
    
    # Draw main rectangle
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='#333333', 
                    linewidth=1.5, alpha=0.7, zorder=2)
    ax.add_patch(rect)
    
    # Add component title
    plt.text(x + width/2, y + height - 25, name, 
             fontsize=14, fontweight='bold', ha='center', va='center', zorder=3)
    
    # Add features
    if 'features' in component:
        # Single column layout
        features = component['features']
        y_start = y + height - 50
        spacing = 20
        
        for i, feature in enumerate(features):
            plt.text(x + 20, y_start - i * spacing, feature, 
                     fontsize=11, ha='left', va='center', zorder=3)
    else:
        # Two column layout
        col1 = component.get('features_col1', [])
        col2 = component.get('features_col2', [])
        y_start = y + height - 50
        spacing = 20
        
        # Draw first column
        for i, feature in enumerate(col1):
            plt.text(x + 20, y_start - i * spacing, feature, 
                     fontsize=11, ha='left', va='center', zorder=3)
        
        # Draw second column
        for i, feature in enumerate(col2):
            plt.text(x + width/2 + 10, y_start - i * spacing, feature, 
                     fontsize=11, ha='left', va='center', zorder=3)


def draw_connection(ax, connection, components, colors):
    """Draw a connection arrow between components."""
    # Find the source and target components
    source = next(c for c in components if c['name'] == connection['start'])
    target = next(c for c in components if c['name'] == connection['end'])
    
    # Determine connection type
    conn_type = connection['type']
    
    # Get component center positions
    source_center_x = source['x'] + source['width'] / 2
    source_center_y = source['y'] + source['height'] / 2
    target_center_x = target['x'] + target['width'] / 2
    target_center_y = target['y'] + target['height'] / 2
    
    # Calculate connection points based on relative positions
    # For horizontal connections on the same row
    if abs(source_center_y - target_center_y) < 10:
        start_x = source['x'] + source['width']
        start_y = source_center_y
        end_x = target['x']
        end_y = target_center_y
        curve = 0  # Straight line
    
    # For vertical or diagonal connections
    else:
        # For main flow from Data Preprocessor to Technical Indicators
        if source['name'] == 'Data Preprocessor' and target['name'] == 'Technical Indicators':
            start_x = source['x'] + source['width']/2 - 40
            start_y = source['y']
            end_x = target['x'] + target['width']/2 + 40
            end_y = target['y'] + target['height']
            curve = 0.05  
        
        # For storage connections
        elif target['name'] == 'MongoDB Storage':
            # Different connection points based on source position
            if source['x'] < target['x']:  # Left side components
                start_x = source['x'] + 3*source['width']/4
                start_y = source['y']
                end_x = target['x'] + target['width']/4
                end_y = target['y'] + target['height']
                curve = -0.1
            elif source['x'] > target['x']:  # Right side components
                start_x = source['x'] + source['width']/4
                start_y = source['y']
                end_x = target['x'] + 3*target['width']/4
                end_y = target['y'] + target['height']
                curve = 0.15
            else:  # Center aligned components
                start_x = source['x'] + source['width']/2
                start_y = source['y']
                end_x = target['x'] + target['width']/2
                end_y = target['y'] + target['height']
                curve = 0
        else:
            # Default connection points for other diagonal connections
            start_x = source_center_x
            start_y = source_center_y
            end_x = target_center_x
            end_y = target_center_y
            curve = 0.2
    
    # Set arrow style based on connection type
    if conn_type == 'main':
        color = colors['main_flow']
        width = 2.5
        arrow_style = '-|>'
        mutation_scale = 20
    else:  # storage
        color = colors['storage_flow']
        width = 1.5
        arrow_style = '-|>'
        mutation_scale = 15
    
    # Draw the arrow
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                           connectionstyle=f'arc3,rad={curve}',
                           arrowstyle=arrow_style,
                           mutation_scale=mutation_scale,
                           linewidth=width,
                           color=color,
                           zorder=1)
    ax.add_patch(arrow)


def add_legend(ax, colors):
    """Add a legend for component types and connections."""
    legend_elements = [
        Patch(facecolor=colors['data'], edgecolor='#333333', label='Data Acquisition'),
        Patch(facecolor=colors['process'], edgecolor='#333333', label='Data Processing'),
        Patch(facecolor=colors['analysis'], edgecolor='#333333', label='Analysis Engine'),
        Patch(facecolor=colors['storage'], edgecolor='#333333', label='Storage System'),
        Line2D([0], [0], color=colors['main_flow'], lw=2.5, label='Main Data Flow'),
        Line2D([0], [0], color=colors['storage_flow'], lw=1.5, label='Storage Connection')
    ]
    
    # Create legend box
    legend = ax.legend(handles=legend_elements, 
               loc='upper center',
               bbox_to_anchor=(0.5, 0.05),
               title='Component Types and Connections',
               ncol=3,
               frameon=True, 
               fontsize=11)
    
    # Make the legend title bold
    legend.get_title().set_fontweight('bold')


# Define colors globally for consistent use
colors = {
    'data': '#A8D5FF',      # Light blue
    'process': '#A8E6B8',   # Light green
    'analysis': '#FFE699',  # Light gold
    'storage': '#FFBF86',   # Light orange
    'main_flow': '#333333', # Dark gray for main flow
    'storage_flow': '#777777', # Medium gray for storage
    'shadow': '#00000022'   # Shadow color
}


if __name__ == "__main__":
    create_pipeline_visualization()