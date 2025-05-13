import React, { useState, useEffect } from "react";
import "./App.css";

interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
}

interface TradingSignal {
  type: "BUY" | "SELL";
  confidence: number;
  timestamp: number;
}

const App: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [activeSymbols, setActiveSymbols] = useState<string[]>([]);

  useEffect(() => {
    // Load initial state from storage
    chrome.storage.local.get(["activeSymbols", "lastMarketData"], (result) => {
      if (result.activeSymbols) {
        setActiveSymbols(result.activeSymbols);
      }
      if (result.lastMarketData) {
        setMarketData(result.lastMarketData);
      }
    });

    // Listen for market data updates
    chrome.runtime.onMessage.addListener((message) => {
      if (message.type === "FETCH_MARKET_DATA") {
        setMarketData(message.data);
      }
      if (message.type === "ANALYZE_SIGNALS") {
        setSignals(message.data);
      }
    });
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Trading Assistant</h1>
      </header>

      <main>
        {marketData ? (
          <div className="market-data">
            <h2>{marketData.symbol}</h2>
            <div className="price">
              <span>Price:</span> ${marketData.price.toFixed(2)}
            </div>
            <div className="volume">
              <span>Volume:</span> {marketData.volume.toLocaleString()}
            </div>
            <div className="timestamp">
              Last updated:{" "}
              {new Date(marketData.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ) : (
          <div className="no-data">No market data available</div>
        )}

        <div className="signals">
          <h2>Trading Signals</h2>
          {signals.length > 0 ? (
            signals.map((signal, index) => (
              <div
                key={index}
                className={`signal-card ${signal.type.toLowerCase()}`}
              >
                <div className="signal-type">{signal.type}</div>
                <div className="confidence">
                  Confidence: {(signal.confidence * 100).toFixed(1)}%
                </div>
                <div className="timestamp">
                  {new Date(signal.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))
          ) : (
            <div className="no-signals">No active signals</div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
