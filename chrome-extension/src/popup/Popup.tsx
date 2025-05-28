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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [symbol, setSymbol] = useState<string>("");
  const [dataSource, setDataSource] = useState<DataSource>("yahoo_finance");
  const [apiKey, setApiKey] = useState<string>("");
  const [showSettings, setShowSettings] = useState(false);
  const [suggestions, setSuggestions] = useState<SymbolSuggestion[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState<SymbolSuggestion | null>(null);

  const fetchData = async (symbol: string, source: DataSource) => {
    if (!symbol) {
      setError("Please enter a symbol");
      return;
    }

    setLoading(true);
    setError(null);
    setMarketData(null);

    try {
      const tabId = await getCurrentTabId();
      if (!tabId) {
        throw new Error("No active tab found");
      }

      // First check if we can establish connection
      try {
        console.log("Attempting to ping content script...");
        const response = await chrome.tabs.sendMessage(tabId, {
          type: "PING",
        });
        console.log("Ping response:", response);
        if (!response || response.type !== "PONG") {
          throw new Error("Failed to connect to data service");
        }
      } catch (error) {
        // If ping fails, reinject the content script
        console.log("Content script not ready, reinjecting...");
        await chrome.scripting.executeScript({
          target: { tabId },
          files: ["dist/content.js"],
        });
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }

      console.log("Requesting market data for:", symbol);
      const response = await chrome.tabs.sendMessage(tabId, {
        type: "GET_MARKET_DATA",
        data: { symbol, source },
      });

      console.log("API response:", response);

      if (!response) {
        throw new Error("No response received from the data service");
      }

      if ("error" in response) {
        throw new Error(response.error);
      }

      if (!isValidMarketData(response)) {
        console.error("Invalid market data structure:", response);
        throw new Error("Received invalid data format from the API");
      }

      setMarketData(response);
      setError(null);
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
      setMarketData(null);
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

  const handleSymbolChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
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
      setError(err instanceof Error ? err.message : "Failed to validate API key");
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

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ width: 350, p: 2, bgcolor: "background.default", minHeight: 200 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
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
              label={`${dataSource === 'alpha_vantage' ? 'Alpha Vantage' : 'Yahoo Finance'} API Key`}
              value={apiKey}
              onChange={handleApiKeyChange}
              type="password"
              placeholder="Enter your API key"
              helperText={dataSource === 'alpha_vantage' ? "Required for Alpha Vantage" : "Optional for Yahoo Finance"}
            />
          )}

          <Autocomplete
            freeSolo
            options={suggestions}
            getOptionLabel={(option) => 
              typeof option === 'string' ? option : `${option.symbol} - ${option.name}`
            }
            value={selectedSuggestion}
            onChange={(_, newValue) => {
              if (typeof newValue === 'string') {
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
                <Typography variant="caption" sx={{ ml: 1, color: "text.secondary" }}>
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
          <Box display="flex" justifyContent="center" alignItems="center" height={140}>
            <CircularProgress />
          </Box>
        )}

        {!loading && marketData && (
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>
              {marketData.symbol}
            </Typography>
            <Typography variant="body1">
              Price: ${marketData.price.toFixed(2)}
            </Typography>
            <Typography variant="body1">
              Volume: {marketData.volume.toLocaleString()}
            </Typography>
            <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
              Last updated: {new Date(marketData.timestamp).toLocaleTimeString()}
            </Typography>
            <Typography variant="caption" color="textSecondary" display="block">
              Source: {marketData.source || dataSource}
            </Typography>
          </Paper>
        )}

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
