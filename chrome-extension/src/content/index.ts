import { MarketData, MessageType } from "../background";

class MarketDataExtractor {
  private observer: MutationObserver;
  private lastUpdate: number = 0;
  private updateInterval: number = 5000;

  constructor() {
    this.observer = new MutationObserver(this.handleDOMChanges.bind(this));
  }

  start() {
    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
    this.extractInitialData();
  }

  private handleDOMChanges(mutations: MutationRecord[]) {
    const now = Date.now();
    if (now - this.lastUpdate > this.updateInterval) {
      this.extractInitialData();
      this.lastUpdate = now;
    }
  }

  private extractInitialData() {
    const data = this.extractMarketData();
    if (data) {
      chrome.runtime.sendMessage({
        type: MessageType.FETCH_MARKET_DATA,
        data,
      });
    }
  }

  private extractMarketData(): MarketData | null {
    try {
      if (window.location.hostname.includes("tradingview.com")) {
        return this.extractTradingViewData();
      }
      if (window.location.hostname.includes("yahoo.com")) {
        return this.extractYahooFinanceData();
      }
      return null;
    } catch (error) {
      console.error("Error extracting market data:", error);
      return null;
    }
  }

  private extractTradingViewData(): MarketData | null {
    const priceElement = document.querySelector('[data-name="last"]');
    const volumeElement = document.querySelector('[data-name="volume"]');
    const symbolElement = document.querySelector(
      ".chart-container [data-symbol]"
    );

    if (!priceElement || !volumeElement || !symbolElement) return null;

    return {
      symbol: symbolElement.getAttribute("data-symbol") || "",
      price: parseFloat(priceElement.textContent || "0"),
      volume: parseFloat(volumeElement.textContent || "0"),
      timestamp: Date.now(),
    };
  }

  private extractYahooFinanceData(): MarketData | null {
    const priceElement = document.querySelector('[data-test="qsp-price"]');
    const volumeElement = document.querySelector('[data-test="qsp-volume"]');
    const symbolElement = document.querySelector('[data-test="qsp-symbol"]');

    if (!priceElement || !volumeElement || !symbolElement) return null;

    return {
      symbol: symbolElement.textContent || "",
      price: parseFloat(priceElement.textContent || "0"),
      volume: parseFloat(volumeElement.textContent || "0"),
      timestamp: Date.now(),
    };
  }
}

// Initialize content script
export const extractor = new MarketDataExtractor();
extractor.start();
