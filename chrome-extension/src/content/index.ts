import { MarketData } from '../types';

class MarketDataExtractor {
  private observer: MutationObserver;
  private lastUpdate: number = 0;
  private updateInterval: number = 5000;

  constructor() {
    this.observer = new MutationObserver(this.handleDOMChanges.bind(this));
    console.log('MarketDataExtractor initialized');
  }

  start() {
    // Extract initial data
    this.extractInitialData();
    
    // Start observing DOM changes
    this.observer.observe(document.body, {
      childList: true,
      subtree: true
    });    // Set up message listener for popup requests
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.type === 'GET_MARKET_DATA') {
        console.log('Received request for market data');
        console.log('Current data:', this.latestData);
        sendResponse(this.latestData);
        return true; // Keep the message channel open for asynchronous response
      }
    });
  }

  private handleDOMChanges(mutations: MutationRecord[]) {
    const now = Date.now();
    if (now - this.lastUpdate >= this.updateInterval) {
      this.extractInitialData();
      this.lastUpdate = now;
    }
  }

  private extractInitialData() {
    const data = this.extractMarketData();
    if (data) {
      this.latestData = data;
      // Send the data to any interested listeners
      chrome.runtime.sendMessage({ type: 'MARKET_DATA_UPDATE', data });
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

// Initialize and start the data extractor when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    const extractor = new MarketDataExtractor();
    extractor.start();
    console.log('Market data extractor started');
  });
} else {
  const extractor = new MarketDataExtractor();
  extractor.start();
  console.log('Market data extractor started');
}
