/// <reference types="chrome" />
import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  Paper,
  ThemeProvider,
  createTheme,
  CircularProgress,
  Alert,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Stack,
  IconButton,
  SelectChangeEvent,
  Autocomplete,
} from "@mui/material";
import { blue, grey } from "@mui/material/colors";
import { MarketData, DataSource } from "../types";
import MarketChart from "../components/MarketChart";

interface SymbolSuggestion {
  symbol: string;
  name: string;
}

const getCurrentTabId = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab?.id;
};

const theme = createTheme({
  palette: {
    mode: "light",
    primary: blue,
    background: {
      default: "#ffffff",
      paper: "#f5f5f5",
    },
  },
});

const isValidMarketData = (data: any): data is MarketData => {
  return (
    typeof data === "object" &&
    data !== null &&
    typeof data.symbol === "string" &&
    typeof data.price === "number" &&
    !isNaN(data.price) &&
    typeof data.volume === "number" &&
    !isNaN(data.volume) &&
    typeof data.timestamp === "number" &&
    !isNaN(data.timestamp)
  );
};

const Popup: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [historicalData, setHistoricalData] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [symbol, setSymbol] = useState<string>("");
  const [dataSource, setDataSource] = useState<DataSource>("yahoo_finance");
  const [apiKey, setApiKey] = useState<string>("");
  const [showSettings, setShowSettings] = useState(false);
  const [suggestions, setSuggestions] = useState<SymbolSuggestion[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] =
    useState<SymbolSuggestion | null>(null);  const injectContentScript = async (tabId: number) => {
    console.log("Injecting content script...");
    try {
      // Get the current tab info to check if we can inject
      const tab = await chrome.tabs.get(tabId);
      if (!tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('edge://') || tab.url.startsWith('about:')) {
        throw new Error('Cannot inject script into browser system pages. Please open a regular webpage.');
      }

      // Try to inject the initialization function first
      await chrome.scripting.executeScript({
        target: { tabId },
        func: () => {
          // This sets up a flag to indicate the script is initializing
          window.postMessage({ type: "CONTENT_SCRIPT_INITIALIZING" }, "*");
        }
      });

      // Inject the actual content script
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"]
      });

      // Wait for initialization
      console.log("Waiting for content script to initialize...");
      let attempts = 0;
      const maxAttempts = 10;
      
      while (attempts < maxAttempts) {
        try {
          const response = await chrome.tabs.sendMessage(tabId, { type: "PING" });
          if (response?.type === "PONG") {
            console.log("Content script initialized successfully");
            return true;
          }
        } catch (e) {
          // Ignore errors during initialization check
        }
        
        await new Promise(resolve => setTimeout(resolve, 500));
        attempts++;
      }

      throw new Error("Content script initialization timed out");
    } catch (error) {
      console.error("Failed to inject content script:", error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Script injection failed: ${message}`);
    }
  };
  const connectToDataService = async (): Promise<boolean> => {
    try {
      const tabId = await getCurrentTabId();
      if (!tabId) {
        throw new Error("No active tab found");
      }

      // Try to ping the content script
      try {
        console.log("Attempting to ping existing content script...");
        const response = await chrome.tabs.sendMessage(tabId, { type: "PING" });
        if (response?.type === "PONG") {
          console.log("Content script is already running");
          return true;
        }
      } catch (error) {
        console.log("Content script not running, will try to inject");
      }

      try {
        // Try to inject the content script
        await injectContentScript(tabId);
        
        // Wait for the script to initialize
        console.log("Waiting for script initialization...");
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Verify the script is running with another ping
        console.log("Verifying script injection...");
        const response = await chrome.tabs.sendMessage(tabId, { type: "PING" });
        
        if (response?.type === "PONG") {
          console.log("Content script successfully initialized");
          return true;
        } else {
          throw new Error("Content script did not respond correctly after injection");
        }
      } catch (error) {
        console.error("Failed during script injection process:", error);
        throw new Error(error instanceof Error ? error.message : "Script injection failed");
      }
    } catch (error) {
      console.error("Failed to connect to data service:", error);
      throw error;
    }
  };

  const fetchData = async (symbol: string, source: DataSource) => {
    if (!symbol) {
      setError("Please enter a symbol");
      return;
    }    setLoading(true);
    setError(null);
    setMarketData(null);
    setHistoricalData([]); // Clear historical data

    try {
      const connected = await connectToDataService();
      if (!connected) {
        throw new Error("Failed to connect to data service. Please refresh the page and try again.");
      }

      const tabId = await getCurrentTabId();
      if (!tabId) {
        throw new Error("No active tab found");
      }

      // Get historical data
      console.log("Fetching historical data...");
      const historicalResponse = await chrome.tabs.sendMessage(tabId, {
        type: "GET_MARKET_DATA",
        data: {
          symbol,
          source,
          interval: "1d",
          limit: 30
        },
      });

      console.log("Historical data response:", historicalResponse);

      if ("error" in historicalResponse) {
        throw new Error(historicalResponse.error);
      }

      // Update historical data
      if (Array.isArray(historicalResponse)) {
        console.log("Setting historical data:", historicalResponse);
        setHistoricalData(historicalResponse);
      } else if (isValidMarketData(historicalResponse)) {
        console.log("Setting single data point as historical:", historicalResponse);
        setHistoricalData([historicalResponse]);
      }

      // Get current data
      console.log("Fetching current data...");
      const response = await chrome.tabs.sendMessage(tabId, {
        type: "GET_MARKET_DATA",
        data: { symbol, source }
      });

      console.log("Current market data response:", response);

      if ("error" in response) {
        throw new Error(response.error);
      }

      if (isValidMarketData(response)) {
        setMarketData(response);
      } else {
        throw new Error("Invalid market data received");
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      setError(error instanceof Error ? error.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  const searchSymbols = async (query: string) => {
    try {
      const tabId = await getCurrentTabId();
      if (!tabId) {
        throw new Error("No active tab found");
      }

      const response = await chrome.tabs.sendMessage(tabId, {
        type: "SEARCH_SYMBOLS",
        data: { query },
      });

      if (response?.suggestions) {
        setSuggestions(response.suggestions);
      } else if (response?.error) {
        console.error("Symbol search error:", response.error);
      }
    } catch (error) {
      console.error("Error searching symbols:", error);
    }
  };

  const handleSymbolChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = event.target.value.toUpperCase();
    setSymbol(value);
    if (value.length >= 1) {
      await searchSymbols(value);
    } else {
      setSuggestions([]);
    }
  };

  const handleSourceChange = (event: SelectChangeEvent) => {
    setDataSource(event.target.value as DataSource);
  };

  const handleSubmit = () => {
    if (!symbol) {
      setError("Please enter a symbol");
      return;
    }
    fetchData(symbol, dataSource);
  };
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) {
      handleSubmit();
    }
  };

  const handleApiKeyChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const key = e.target.value.trim();
    setApiKey(key);
    if (!key) return;

    setLoading(true);
    try {
      // Test the API key with a simple request
      const tabId = await getCurrentTabId();
      if (!tabId) {
        throw new Error("No active tab found");
      }

      // Validate the API key with the content script
      const response = await chrome.tabs.sendMessage(tabId, {
        type: "VALIDATE_API_KEY",
        data: { source: dataSource, apiKey: key },
      });

      if (response?.isValid) {
        // Save API key to storage only if valid
        chrome.storage.local.set({ [`${dataSource}_api_key`]: key });
        setError(null);
      } else {
        setError("Invalid API key. Please check and try again.");
      }
    } catch (err) {
      console.error("Error validating API key:", err);
      setError(
        err instanceof Error ? err.message : "Failed to validate API key"
      );
    } finally {
      setLoading(false);
    }
  };

  // Load saved API key when data source changes
  useEffect(() => {
    chrome.storage.local.get(`${dataSource}_api_key`, (result) => {
      const savedKey = result[`${dataSource}_api_key`];
      if (savedKey) {
        setApiKey(savedKey);
      } else {
        setApiKey("");
      }
    });
  }, [dataSource]);

  // Load initial suggestions
  useEffect(() => {
    searchSymbols("");
  }, []);

  useEffect(() => {
    if (marketData) {
      // Add new market data to historical data
      setHistoricalData((prev) => {
        const newData = [...prev, marketData];
        // Keep only last 100 data points
        return newData.slice(-100);
      });
    }
  }, [marketData]);
  const renderMarketData = () => {
    if (!marketData) {
      return null;
    }

    // Debug logs
    console.log('MarketData:', marketData);
    console.log('HistoricalData:', historicalData);

    return (
      <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Market Data
        </Typography>
        <Stack spacing={2}>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Symbol
            </Typography>
            <Typography variant="body1">{marketData.symbol}</Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Price
            </Typography>
            <Typography variant="body1">
              ${marketData.price.toFixed(2)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Volume
            </Typography>
            <Typography variant="body1">
              {marketData.volume.toLocaleString()}
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Time
            </Typography>
            <Typography variant="body1">
              {new Date(marketData.timestamp).toLocaleString()}
            </Typography>
          </Box>          {historicalData.length > 0 && (
            <Box sx={{ mt: 2, height: 400 }}>
              <MarketChart data={historicalData} height={400} />
            </Box>
          )}
        </Stack>
      </Paper>
    );
  };

  // Debug historical data changes
  useEffect(() => {
    console.log('Historical data updated:', {
      length: historicalData.length,
      firstItem: historicalData[0],
      lastItem: historicalData[historicalData.length - 1]
    });
  }, [historicalData]);

  // Track historical data state changes
  useEffect(() => {
    if (historicalData.length > 0) {
      console.group('Historical Data Update');
      console.log('Number of data points:', historicalData.length);
      console.log('Time range:', {
        start: new Date(historicalData[0].timestamp).toLocaleString(),
        end: new Date(historicalData[historicalData.length - 1].timestamp).toLocaleString()
      });
      console.log('Price range:', {
        min: Math.min(...historicalData.map(d => d.low || d.price)),
        max: Math.max(...historicalData.map(d => d.high || d.price))
      });
      console.groupEnd();
    }
  }, [historicalData]);

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{ width: 350, p: 2, bgcolor: "background.default", minHeight: 200 }}
      >
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          mb={2}
        >
          <Typography variant="h6">Market Data</Typography>
          <Button size="small" onClick={() => setShowSettings(!showSettings)}>
            {showSettings ? "Hide Settings" : "Settings"}
          </Button>
        </Box>

        <Stack spacing={2} sx={{ mb: 2 }}>
          <FormControl fullWidth>
            <InputLabel id="data-source-label">Data Source</InputLabel>
            <Select
              labelId="data-source-label"
              value={dataSource}
              label="Data Source"
              onChange={handleSourceChange}
              disabled={loading}
            >
              <MenuItem value="yahoo_finance">Yahoo Finance</MenuItem>
              <MenuItem value="alpha_vantage">Alpha Vantage</MenuItem>
            </Select>
          </FormControl>

          {showSettings && (
            <TextField
              fullWidth
              label={`${
                dataSource === "alpha_vantage"
                  ? "Alpha Vantage"
                  : "Yahoo Finance"
              } API Key`}
              value={apiKey}
              onChange={handleApiKeyChange}
              type="password"
              placeholder="Enter your API key"
              helperText={
                dataSource === "alpha_vantage"
                  ? "Required for Alpha Vantage"
                  : "Optional for Yahoo Finance"
              }
            />
          )}

          <Autocomplete
            freeSolo
            options={suggestions}
            getOptionLabel={(option) =>
              typeof option === "string"
                ? option
                : `${option.symbol} - ${option.name}`
            }
            value={selectedSuggestion}
            onChange={(_, newValue) => {
              if (typeof newValue === "string") {
                setSymbol(newValue);
                setSelectedSuggestion(null);
              } else if (newValue) {
                setSymbol(newValue.symbol);
                setSelectedSuggestion(newValue);
              } else {
                setSymbol("");
                setSelectedSuggestion(null);
              }
            }}
            onInputChange={(_, value) => {
              setSymbol(value.toUpperCase());
              searchSymbols(value);
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                fullWidth
                label="Symbol"
                placeholder="Enter stock symbol (e.g. MSFT)"
                disabled={loading}
                error={Boolean(error && !loading)}
                helperText={error && !loading ? error : null}
                onKeyPress={handleKeyPress}
              />
            )}
            renderOption={(props, option) => (
              <li {...props}>
                <Typography variant="body1">{option.symbol}</Typography>
                <Typography
                  variant="caption"
                  sx={{ ml: 1, color: "text.secondary" }}
                >
                  {option.name}
                </Typography>
              </li>
            )}
            loading={loading}
            disabled={loading}
          />

          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleSubmit}
            disabled={loading || !symbol}
          >
            {loading ? "Fetching..." : "Get Data"}
          </Button>
        </Stack>

        {loading && (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            height={140}
          >
            <CircularProgress />
          </Box>
        )}

        {renderMarketData()}

        {!loading && error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Box>
    </ThemeProvider>
  );
};

export default Popup;
