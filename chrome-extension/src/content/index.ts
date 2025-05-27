import { MarketData } from "../types";

class MarketDataExtractor {
  private observer: MutationObserver;
  private lastUpdate: number = 0;
  private updateInterval: number = 5000;

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
        const data = this.extractMarketData();
        console.log("Extracted data:", data);
        sendResponse(data);
        return true;
      }
    });

    // Start observing DOM changes
    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    console.log("DOM observer started");
  }

  private handleDOMChanges(mutations: MutationRecord[]) {
    const now = Date.now();
    if (now - this.lastUpdate >= this.updateInterval) {
      console.log("DOM changed, checking for market data updates");
      this.extractMarketData();
      this.lastUpdate = now;
    }
  }

  private extractMarketData(): MarketData | null {
    try {
      if (window.location.hostname.includes("finance.yahoo.com")) {
        return this.extractYahooFinanceData();
      }
      if (window.location.hostname.includes("tradingview.com")) {
        return this.extractTradingViewData();
      }
      console.log("Not on a supported financial page");
      return null;
    } catch (error) {
      console.error("Error extracting market data:", error);
      return null;
    }
  }

  private extractYahooFinanceData(): MarketData | null {
    console.log("Attempting to extract Yahoo Finance data");

    try {
        // Log all potential elements we find
        const possiblePriceSelectors = [
            '[data-test="qsp-price"]',
            'fin-streamer[data-field="regularMarketPrice"]',
            '[data-symbol-price]',
            '#quote-header-info fin-streamer[value]'
        ];

        const possibleVolumeSelectors = [
            '[data-test="qsp-volume"]',
            'fin-streamer[data-field="regularMarketVolume"]',
            '#quote-summary [data-test="VOLUME-value"]'
        ];

        const possibleSymbolSelectors = [
            '[data-test="qsp-symbol"]',
            'h1 [data-symbol]',
            '[data-symbol]',
            '[class*="symbol"]'
        ];

        // Try each selector and log what we find
        console.log("Searching for price elements...");
        possiblePriceSelectors.forEach(selector => {
            const element = document.querySelector(selector);
            console.log(`Selector ${selector}:`, {
                found: !!element,
                value: element?.textContent,
                attributes: element ? Array.from(element.attributes).map(attr => `${attr.name}=${attr.value}`).join(', ') : null
            });
        });

        // Get first working selector
        const priceElement = possiblePriceSelectors.map(selector => document.querySelector(selector)).find(el => el);
        const volumeElement = possibleVolumeSelectors.map(selector => document.querySelector(selector)).find(el => el);
        const symbolElement = possibleSymbolSelectors.map(selector => document.querySelector(selector)).find(el => el);

        console.log("Final elements found:", {
            price: priceElement?.textContent,
            priceAttributes: priceElement ? Array.from(priceElement.attributes).map(attr => `${attr.name}=${attr.value}`) : null,
            volume: volumeElement?.textContent,
            symbol: symbolElement?.textContent
        });

        if (!priceElement || !volumeElement || !symbolElement) {
            console.log("Missing required elements on Yahoo Finance page");
            // Log the entire relevant HTML section for debugging
            console.log("Quote header HTML:", document.getElementById('quote-header-info')?.outerHTML);
            return null;
        }

        const price = parseFloat(priceElement.textContent?.replace(/[^0-9.-]/g, "") || "0");
        const volume = parseFloat(volumeElement.textContent?.replace(/[^0-9.]/g, "") || "0");
        const symbol = symbolElement.textContent?.trim() || "";

        if (!price || !volume || !symbol) {
            console.log("Invalid data values found:", { price, volume, symbol });
            return null;
        }

        const data = {
            symbol,
            price,
            volume,
            timestamp: Date.now(),
        };

        console.log("Successfully extracted Yahoo Finance data:", data);
        return data;
    } catch (error) {
        console.error("Error extracting Yahoo Finance data:", error);
        return null;
    }
}

  private extractTradingViewData(): MarketData | null {
    console.log("Attempting to extract TradingView data");

    try {
      const priceElement = document.querySelector('[data-name="last"]');
      const volumeElement = document.querySelector('[data-name="volume"]');
      const symbolElement = document.querySelector(
        ".chart-container [data-symbol]"
      );

      console.log("Found elements:", {
        price: priceElement?.textContent,
        volume: volumeElement?.textContent,
        symbol: symbolElement?.getAttribute("data-symbol"),
      });

      if (!priceElement || !volumeElement || !symbolElement) {
        console.log("Missing required elements on TradingView page");
        return null;
      }

      const data = {
        symbol: symbolElement.getAttribute("data-symbol") || "",
        price: parseFloat(
          priceElement.textContent?.replace(/[^0-9.-]/g, "") || "0"
        ),
        volume: parseFloat(
          volumeElement.textContent?.replace(/[^0-9.]/g, "") || "0"
        ),
        timestamp: Date.now(),
      };

      console.log("Successfully extracted TradingView data:", data);
      return data;
    } catch (error) {
      console.error("Error extracting TradingView data:", error);
      return null;
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
