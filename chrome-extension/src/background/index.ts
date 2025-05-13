export interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
}

export interface TradingSignal {
  type: "BUY" | "SELL";
  confidence: number;
  timestamp: number;
}

export enum MessageType {
  FETCH_MARKET_DATA = "FETCH_MARKET_DATA",
  ANALYZE_SIGNALS = "ANALYZE_SIGNALS",
}

chrome.runtime.onInstalled.addListener(() => {
  console.log("Trading Assistant Extension installed");

  // Initialize storage with default settings
  chrome.storage.local.set({
    activeSymbols: ["MSFT", "AAPL", "GOOGL"],
    refreshInterval: 60000, // 1 minute
    notificationsEnabled: true,
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case MessageType.FETCH_MARKET_DATA:
      handleMarketData(message.data);
      break;
    case MessageType.ANALYZE_SIGNALS:
      analyzeSignals(message.data);
      break;
    default:
      console.log("Unknown message type:", message.type);
  }
  return true;
});

async function handleMarketData(data: MarketData) {
  try {
    await chrome.storage.local.set({ lastMarketData: data });
    console.log("Market data updated:", data);
  } catch (error) {
    console.error("Error handling market data:", error);
  }
}

async function analyzeSignals(
  marketData: MarketData
): Promise<TradingSignal[]> {
  const signals: TradingSignal[] = [];
  if (marketData.price > 0) {
    signals.push({
      type: "BUY",
      confidence: 0.85,
      timestamp: Date.now(),
    });
  }
  return signals;
}
