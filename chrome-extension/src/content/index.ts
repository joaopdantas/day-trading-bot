import { MarketData } from "../types";

class MarketDataExtractor {
  private observer: MutationObserver;
  private lastUpdate: number = 0;
  private updateInterval: number = 5000;
  private isInitialized: boolean = false;

  constructor() {
    this.observer = new MutationObserver(this.handleDOMChanges.bind(this));
    console.log("MarketDataExtractor initialized");
  }

  start() {
    console.log("Starting MarketDataExtractor");

    // Set up message listener for popup requests
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      console.log("Received message:", request);

      if (request.type === "PING") {
        console.log("Received ping, sending pong");
        sendResponse({ type: "PONG" });
        return true;
      }

      if (request.type === "GET_MARKET_DATA") {
        console.log("Extracting market data on demand");
        try {
          const data = this.extractMarketData();
          if (!data) {
            console.log("No market data found");
            sendResponse({ error: "No market data found on this page" });
          } else {
            console.log("Extracted data:", data);
            sendResponse(data);
          }
        } catch (error) {
          console.error("Error extracting market data:", error);
          sendResponse({ error: "Failed to extract market data" });
        }
        return true;
      }
    });

    // Start observing DOM changes
    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    // Wait for the page to be fully loaded
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.initializeExtractor());
    } else {
      this.initializeExtractor();
    }

    console.log("DOM observer started");
  }

  private initializeExtractor() {
    console.log("Initializing market data extractor");
    this.isInitialized = true;
    // Try initial market data extraction
    this.extractMarketData();
  }

  private extractMarketData(): MarketData | null {
    try {
      console.log("Attempting to extract market data");
      const host = window.location.hostname;

      if (host.includes("finance.yahoo.com")) {
        return this.extractYahooFinanceData();
      } else if (host.includes("tradingview.com")) {
        return this.extractTradingViewData();
      }

      console.log("Not on a supported financial page");
      return null;
    } catch (error) {
      console.error("Error in extractMarketData:", error);
      return null;
    }
  }

  private extractYahooFinanceData(): MarketData | null {
    try {
      // Find the main price element
      const priceEl = document.querySelector('[data-test="qsp-price"]');
      const symbolEl = document.querySelector('[data-test="quote-header"] [class*="C($primaryColor)"]');
      const volumeEl = document.querySelector('[data-test="VOLUME-value"]');

      if (!priceEl || !symbolEl || !volumeEl) {
        console.log("Missing required elements for Yahoo Finance data");
        return null;
      }

      const price = parseFloat(priceEl.textContent?.replace(/[^0-9.-]/g, "") || "0");
      const symbol = symbolEl.textContent?.trim() || "";
      const volume = parseInt(volumeEl.textContent?.replace(/[^0-9]/g, "") || "0");

      if (!price || !symbol || !volume) {
        console.log("Invalid data extracted from Yahoo Finance");
        return null;
      }

      return {
        symbol,
        price,
        volume,
        timestamp: Date.now(),
      };
    } catch (error) {
      console.error("Error extracting Yahoo Finance data:", error);
      return null;
    }
  }

  private extractTradingViewData(): MarketData | null {
    try {
      // Find the main price element (TradingView specific selectors)
      const priceEl = document.querySelector(".js-symbol-last");
      const symbolEl = document.querySelector(".js-symbol-header-symbol");
      const volumeEl = document.querySelector(".js-symbol-volume");

      if (!priceEl || !symbolEl || !volumeEl) {
        console.log("Missing required elements for TradingView data");
        return null;
      }

      const price = parseFloat(priceEl.textContent?.replace(/[^0-9.-]/g, "") || "0");
      const symbol = symbolEl.textContent?.trim() || "";
      const volume = parseInt(volumeEl.textContent?.replace(/[^0-9]/g, "") || "0");

      if (!price || !symbol || !volume) {
        console.log("Invalid data extracted from TradingView");
        return null;
      }

      return {
        symbol,
        price,
        volume,
        timestamp: Date.now(),
      };
    } catch (error) {
      console.error("Error extracting TradingView data:", error);
      return null;
    }
  }

  private handleDOMChanges(mutations: MutationRecord[]) {
    const now = Date.now();
    if (now - this.lastUpdate >= this.updateInterval) {
      console.log("DOM changed, checking for market data updates");
      this.extractMarketData();
      this.lastUpdate = now;
    }
  }
}

// Initialize and start the data extractor
console.log("Content script loaded, checking DOM state");

const initializeExtractor = () => {
  console.log("Initializing MarketDataExtractor");
  const extractor = new MarketDataExtractor();
  extractor.start();
};

if (document.readyState === "loading") {
  console.log("DOM still loading, waiting for DOMContentLoaded");
  document.addEventListener("DOMContentLoaded", initializeExtractor);
} else {
  console.log("DOM already ready, initializing immediately");
  initializeExtractor();
}
