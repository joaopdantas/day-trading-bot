# MakesALot Trading Assistant

An AI-powered trading analysis and prediction tool that combines technical analysis with machine learning for enhanced trading decisions.

## Features

- Real-time technical analysis with multiple indicators
- Machine learning-based price movement predictions
- Support and resistance level detection
- Interactive trading charts
- Multi-platform compatibility (TradingView, ThinkOrSwim, Webull)
- User feedback and continuous improvement system
- Performance analytics and tracking

## Installation

### Backend API

1. Clone the repository:

```bash
git clone https://github.com/yourusername/day-trading-bot.git
cd day-trading-bot/makesalot-api
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the API:

```bash
uvicorn main:app --reload
```

### Chrome Extension

1. Navigate to the extension directory:

```bash
cd chrome-extension
```

2. Install dependencies:

```bash
npm install
```

3. Build the extension:

```bash
npm run build
```

4. Load the extension in Chrome:

- Open Chrome and go to `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked"
- Select the `dist` directory

## Usage

1. Click the extension icon to open the trading assistant
2. Enter a symbol or use auto-detection
3. View technical analysis and predictions
4. Use the interactive charts for visualization
5. Submit feedback to help improve the system

## Configuration

### API Settings

Edit `.env` file:

```env
DEBUG=true
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./makesalot.db
SECRET_KEY=your-secret-key
CHROME_EXTENSION_URL=chrome-extension://your-extension-id
SENTRY_DSN=your-sentry-dsn
```

### Extension Settings

Edit `src/config.ts`:

```typescript
export const config = {
  apiUrl:
    process.env.NODE_ENV === "production"
      ? "https://api.makesalot.trading"
      : "http://localhost:8000",
  defaultTimeframe: "1d",
  defaultIndicators: ["RSI", "MACD", "BB"],
  // ...
};
```

## Development

### Running Tests

Backend:

```bash
pytest tests/
```

Extension:

```bash
npm test
```

### Building for Production

Backend:

```bash
docker-compose build
docker-compose up -d
```

Extension:

```bash
npm run build:prod
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue or contact support@makesalot.trading
