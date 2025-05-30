# Cross-Platform Day Trading Analysis & Automation Bot

## Project Overview
A sophisticated algorithmic trading system implemented as a Chrome extension that works across multiple trading platforms (XTB, Binance, etc.). The system analyzes stock price patterns, recognizes potential trading opportunities, and either provides probability-based recommendations or executes trades automatically based on user preferences.

## Key Features
- **Technical Analysis Engine**: Implements standard indicators (MACD, RSI, Bollinger Bands) and pattern recognition
- **Machine Learning Integration**: Uses historical data to train predictive models for price movement forecasting
- **Cross-Platform Compatibility**: Functions on multiple trading platforms through a Chrome extension
- **Dual Operation Modes**: 
  - Advisory mode: Displays probability analysis and trading suggestions
  - Automated mode: Executes trades based on algorithm-determined opportunities
- **Backtesting Framework**: Allows strategy validation against historical market data
- **Performance Dashboard**: Tracks success rate, ROI, and other key trading metrics

## Project Structure
```
day-trading-bot/
├── src/                    # Main source code directory
│   ├── data/               # Data handling components
│   │   ├── __init__.py
│   │   ├── fetcher.py      # Code for fetching market data from APIs
│   │   ├── preprocessor.py # Data cleaning and preparation functions
│   │   └── storage.py      # MongoDB storage implementation
│   ├── indicators/         # Technical analysis indicators
│   │   ├── __init__.py
│   │   └── technical.py    # Implementation of technical indicators, patterns, signals
│   ├── models/             # Machine learning models
│   │   ├── __init__.py
│   │   └── ml.py           # ML model definitions and training code
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── helpers.py      # Helper functions used across the project
├── tests/                  # Unit tests directory
│   ├── __init__.py
│   ├── test_data.py
│   └── test_indicators.py
├── notebooks/              # Jupyter notebooks for exploration
│   └── prototyping.ipynb
├── config/                 # Configuration files
│   └── settings.py
├── testing/                # Testing and visualization scripts
│   ├── control_point_1/    # Contains tests for CP1
│   │   ├── enhanced_test_implementation.py
│   │   └── test_implementation.py
│   ├── control_point_2/    # Contains tests for CP2
│   │   ├── test_data_pipeline.py
│   │   ├── visualize_pipeline.py
│   │   └── easier_explain_visualization.py
│   └── control_point_3/    # Contains tests for CP3
│       ├── test_dataset_preparation.py
│       └── beginner_friendly_technical_visualization3.py
├── .env                    # Environment variables and API keys
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── setup.py                # Package installation script           
```

## Development Timeline
The project follows a phased approach with five control points:

1. **Foundation Phase** (by March 26, 2025): Environment setup, architecture, and data pipeline
2. **Algorithm Development** (April-May): Technical indicators, pattern recognition, and ML models
3. **Chrome Extension Development** (May-June): UI and platform integration
4. **Integration & Deployment** (June): Connecting components and final delivery

See `project-timeline.md` for detailed schedule information.

## Getting Started

### Prerequisites
- Python 3.11+
- Visual Studio Code
- Chrome Browser (for extension development and testing)

### Installation

1. Clone the repository
```bash
git clone https://github.com/joaopdantas/day-trading-bot.git
cd day-trading-bot
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure API Keys
Create a `.env` file in the root directory with your API keys:
```
MARKET_DATA_API_KEY=your_api_key_here
```

### Running Tests
```bash
pytest
```

## Technical Requirements
- Python backend for algorithms, data processing, and ML modeling
- JavaScript/React frontend for Chrome extension
- REST API connecting frontend and backend components
- Cross-platform DOM manipulation for market data extraction
- Secure authentication and data handling

## Initial Scope
- Focus on stock markets (expand to other assets later)
- Begin with paper trading to validate strategies
- Start with technical analysis before implementing ML components

## License

## Contributors
Rui Alves
João Dantas
