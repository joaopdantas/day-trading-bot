import { MarketData, DataSource } from "../types";

class MarketDataExtractor {
  private static readonly POPULAR_SYMBOLS = [
    { symbol: "AAPL", name: "Apple Inc." },
    { symbol: "MSFT", name: "Microsoft Corporation" },
    { symbol: "GOOGL", name: "Alphabet Inc." },
    { symbol: "AMZN", name: "Amazon.com Inc." },
    { symbol: "META", name: "Meta Platforms Inc." },
    { symbol: "TSLA", name: "Tesla Inc." },
    { symbol: "NVDA", name: "NVIDIA Corporation" },
    { symbol: "JPM", name: "JPMorgan Chase & Co." },
  ];

  private static readonly CACHE_DURATION = 60 * 1000; // 1 minute cache duration
  private observer: MutationObserver;
  private lastUpdate: number = 0;
  private updateInterval: number = 5000;
  private isInitialized: boolean = false;
  private apiEndpoints = {
    alpha_vantage: "https://www.alphavantage.co/query",
    yahoo_finance: "https://query1.finance.yahoo.com/v8/finance/chart"
  };
  private cache: Map<string, { data: MarketData; timestamp: number }> = new Map();

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
        console.log("Extracting market data on demand:", request.data);
        (async () => {
          try {
            const data = request.data;
            if (!data || !data.symbol || !data.source) {
              sendResponse({ error: "Invalid request: missing symbol or data source" });
              return;
            }
            await this.fetchMarketData(data, sendResponse);
          } catch (error) {
            console.error("Error in GET_MARKET_DATA handler:", error);
            sendResponse({ error: error instanceof Error ? error.message : "Failed to fetch market data" });
          }
        })();
        return true;
      }

      if (request.type === "VALIDATE_API_KEY") {
        console.log("Validating API key:", request.data);
        (async () => {
          try {
            const data = request.data;
            if (!data || !data.source || !data.apiKey) {
              sendResponse({ isValid: false, error: "Invalid request: missing source or API key" });
              return;
            }
            const isValid = await this.validateApiKey(data.source, data.apiKey);
            sendResponse({ isValid });
          } catch (error) {
            console.error("Error in VALIDATE_API_KEY handler:", error);
            sendResponse({ isValid: false, error: error instanceof Error ? error.message : "Failed to validate API key" });
          }
        })();
        return true;
      }

      if (request.type === "SEARCH_SYMBOLS") {
        console.log("Searching symbols:", request.data);
        (async () => {
          try {
            const data = request.data;
            if (!data || !data.query) {
              sendResponse({ suggestions: MarketDataExtractor.POPULAR_SYMBOLS });
              return;
            }
            const suggestions = await this.searchSymbols(data.query);
            sendResponse({ suggestions });
          } catch (error) {
            console.error("Error in SEARCH_SYMBOLS handler:", error);
            sendResponse({ suggestions: [], error: error instanceof Error ? error.message : "Failed to search symbols" });
          }
        })();
        return true;
      }
    });

    console.log("Message listener set up");
  }

  private getCachedData(symbol: string, source: DataSource): MarketData | null {
    const cacheKey = `${symbol}:${source}`;
    const cached = this.cache.get(cacheKey);
    
    if (cached) {
      const now = Date.now();
      if (now - cached.timestamp < MarketDataExtractor.CACHE_DURATION) {
        console.log(`Cache hit for ${cacheKey}`);
        return cached.data;
      } else {
        console.log(`Cache expired for ${cacheKey}`);
        this.cache.delete(cacheKey);
      }
    }
    return null;
  }

  private setCachedData(data: MarketData): void {
    const cacheKey = `${data.symbol}:${data.source}`;
    this.cache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });
    console.log(`Cached data for ${cacheKey}`);
  }

  private async fetchMarketData(
    { symbol, source }: { symbol: string; source: DataSource },
    sendResponse: (response: any) => void
  ) {
    try {
      console.log(`Fetching data for ${symbol} from ${source}`);
      
      // Check cache first
      const cachedData = this.getCachedData(symbol, source);
      if (cachedData) {
        sendResponse(cachedData);
        return;
      }

      if (source === "alpha_vantage") {
        await this.fetchAlphaVantageData(symbol, (response) => {
          if (!("error" in response)) {
            this.setCachedData(response);
          }
          sendResponse(response);
        });
      } else {
        await this.fetchYahooFinanceData(symbol, (response) => {
          if (!("error" in response)) {
            this.setCachedData(response);
          }
          sendResponse(response);
        });
      }
    } catch (error) {
      console.error(`Error fetching data from ${source}:`, error);
      sendResponse({
        error: `Failed to fetch data from ${source}. ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    }
  }

  private async makeApiRequest(url: URL, options?: RequestInit): Promise<any> {
    const defaultOptions: RequestInit = {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Chrome Extension'
      },
      mode: 'cors',
      ...options
    };

    try {
      const response = await fetch(url.toString(), defaultOptions);
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => 'No error details available');
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      const data = await response.json();
      return data;
    } catch (error: unknown) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown error occurred while fetching data');
    }
  }

  private async fetchAlphaVantageData(symbol: string, sendResponse: (response: any) => void) {
    try {
      const apiKey = await this.getApiKey("alpha_vantage");
      if (!apiKey) {
        throw new Error("Alpha Vantage API key not configured. Please set your API key in the extension settings.");
      }

      const url = new URL(this.apiEndpoints.alpha_vantage);
      url.searchParams.append("function", "GLOBAL_QUOTE");
      url.searchParams.append("symbol", symbol);
      url.searchParams.append("apikey", apiKey);

      const data = await this.makeApiRequest(url);

      if ("Error Message" in data) {
        throw new Error(data["Error Message"]);
      }

      if (!data["Global Quote"] || Object.keys(data["Global Quote"]).length === 0) {
        throw new Error(`No data available for symbol: ${symbol}`);
      }

      const quote = data["Global Quote"];
      const price = parseFloat(quote["05. price"]);
      const volume = parseInt(quote["06. volume"]);

      if (isNaN(price) || isNaN(volume)) {
        throw new Error("Invalid price or volume data received");
      }

      const marketData: MarketData = {
        symbol: quote["01. symbol"] || symbol,
        price,
        volume,
        timestamp: Date.now(),
        source: "alpha_vantage"
      };

      console.log("Successfully fetched Alpha Vantage data:", marketData);
      sendResponse(marketData);
    } catch (error) {
      console.error("Alpha Vantage API error:", error);
      sendResponse({
        error: error instanceof Error ? error.message : "Failed to fetch Alpha Vantage data"
      });
    }
  }

  private async fetchYahooFinanceData(symbol: string, sendResponse: (response: any) => void) {
    try {
      if (!symbol || typeof symbol !== 'string') {
        throw new Error("Invalid symbol provided");
      }

      const end = Math.floor(Date.now() / 1000);
      const start = end - 86400; // last 24 hours

      const url = new URL(`${this.apiEndpoints.yahoo_finance}/${encodeURIComponent(symbol)}`);
      url.searchParams.append("period1", start.toString());
      url.searchParams.append("period2", end.toString());
      url.searchParams.append("interval", "1d");
      url.searchParams.append("includePrePost", "false");
      url.searchParams.append("events", "div,split");

      console.log("Requesting URL:", url.toString());
      
      const data = await this.makeApiRequest(url);
      console.log("Raw Yahoo Finance response:", data);

      if (data.error) {
        throw new Error(data.error.description || "Yahoo Finance API error");
      }

      if (!data?.chart?.result?.[0]?.indicators?.quote?.[0]) {
        throw new Error(`No data available for symbol: ${symbol}`);
      }

      const result = data.chart.result[0];
      const quote = result.indicators.quote[0];
      
      if (!result.timestamp?.length || !quote.close?.length || !quote.volume?.length) {
        throw new Error(`Incomplete data received for symbol: ${symbol}`);
      }

      let latestIndex = result.timestamp.length - 1;
      let price = quote.close[latestIndex];
      let volume = quote.volume[latestIndex];

      while (latestIndex >= 0 && (price === null || volume === null)) {
        latestIndex--;
        price = quote.close[latestIndex];
        volume = quote.volume[latestIndex];
      }

      if (latestIndex < 0 || typeof price !== 'number' || typeof volume !== 'number' || isNaN(price) || isNaN(volume)) {
        throw new Error(`No valid price data available for symbol: ${symbol}`);
      }

      const marketData: MarketData = {
        symbol: result.meta.symbol || symbol,
        price,
        volume,
        timestamp: result.timestamp[latestIndex] * 1000,
        source: "yahoo_finance"
      };

      console.log("Successfully fetched Yahoo Finance data:", marketData);
      sendResponse(marketData);
    } catch (error) {
      console.error("Yahoo Finance API error:", error);
      sendResponse({
        error: error instanceof Error ? error.message : "Failed to fetch Yahoo Finance data"
      });
    }
  }

  private async getApiKey(source: DataSource): Promise<string | null> {
    try {
      return new Promise((resolve) => {
        if (!chrome?.storage?.local) {
          console.error("Chrome storage is not available");
          resolve(null);
          return;
        }

        chrome.storage.local.get([`${source}_api_key`], (result) => {
          const apiKey = result[`${source}_api_key`];
          console.log(`API key ${apiKey ? "found" : "not found"} for ${source}`);
          resolve(apiKey || null);
        });
      });
    } catch (error) {
      console.error("Error accessing chrome.storage:", error);
      return null;
    }
  }

  private async validateApiKey(source: DataSource, apiKey: string): Promise<boolean> {
    try {
      if (source === "alpha_vantage") {
        const testUrl = new URL(this.apiEndpoints.alpha_vantage);
        testUrl.searchParams.append("function", "TIME_SERIES_INTRADAY");
        testUrl.searchParams.append("symbol", "MSFT"); // Using Microsoft as test symbol
        testUrl.searchParams.append("interval", "1min");
        testUrl.searchParams.append("apikey", apiKey);
        
        const data = await this.makeApiRequest(testUrl);
        return !("Error Message" in data) && !("Information" in data);
      }
      // Yahoo Finance doesn't require API key validation
      return true;
    } catch (error) {
      console.error(`API key validation failed for ${source}:`, error);
      return false;
    }
  }

  private async searchSymbols(query: string): Promise<Array<{ symbol: string; name: string }>> {
    try {
      if (query.length < 1) {
        return MarketDataExtractor.POPULAR_SYMBOLS;
      }

      const source = "alpha_vantage";
      const apiKey = await this.getApiKey(source);
      
      if (!apiKey) {
        // If no API key, return filtered popular symbols
        return MarketDataExtractor.POPULAR_SYMBOLS.filter(
          item => item.symbol.includes(query.toUpperCase()) || 
                 item.name.toLowerCase().includes(query.toLowerCase())
        );
      }

      const url = new URL(this.apiEndpoints.alpha_vantage);
      url.searchParams.append("function", "SYMBOL_SEARCH");
      url.searchParams.append("keywords", query);
      url.searchParams.append("apikey", apiKey);

      const data = await this.makeApiRequest(url);
      
      if ("Error Message" in data || !data.bestMatches) {
        return MarketDataExtractor.POPULAR_SYMBOLS.filter(
          item => item.symbol.includes(query.toUpperCase()) || 
                 item.name.toLowerCase().includes(query.toLowerCase())
        );
      }

      return data.bestMatches.map((match: any) => ({
        symbol: match["1. symbol"],
        name: match["2. name"],
      }));
    } catch (error) {
      console.error("Error searching symbols:", error);
      return MarketDataExtractor.POPULAR_SYMBOLS.filter(
        item => item.symbol.includes(query.toUpperCase()) || 
               item.name.toLowerCase().includes(query.toLowerCase())
      );
    }
  }

  private handleDOMChanges(mutations: MutationRecord[]) {
    const now = Date.now();
    if (now - this.lastUpdate >= this.updateInterval) {
      this.lastUpdate = now;
    }
  }
}

// Initialize and start the data extractor
console.log("Content script loaded");
const extractor = new MarketDataExtractor();
extractor.start();
