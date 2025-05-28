/// <reference types="chrome" />
import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  ThemeProvider, 
  createTheme,
  CircularProgress,
  Alert
} from '@mui/material';
import { blue, grey } from '@mui/material/colors';
import { MarketData } from '../types';

const getCurrentTabId = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab.id;
};

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: blue,
    background: {
      default: grey[900],
      paper: grey[800],
    }
  }
});

const isValidMarketData = (data: any): data is MarketData => {
  return (
    typeof data === 'object' &&
    data !== null &&
    typeof data.symbol === 'string' &&
    typeof data.price === 'number' &&
    typeof data.volume === 'number' &&
    typeof data.timestamp === 'number'
  );
};

const Popup: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const tabId = await getCurrentTabId();
        if (!tabId) {
          throw new Error('No active tab found');
        }

        // Check if we're on a supported page
        const tab = await chrome.tabs.get(tabId);
        const isFinancePage = tab.url?.includes('finance.yahoo.com') || 
                             tab.url?.includes('tradingview.com');
        
        if (!isFinancePage) {
          throw new Error('Please navigate to Yahoo Finance or TradingView');
        }

        // First check if we can establish connection
        try {
          console.log("Attempting to ping content script...");
          const response = await chrome.tabs.sendMessage(tabId, { type: "PING" });
          console.log("Ping response:", response);
          if (!response || response.type !== "PONG") {
            throw new Error('Invalid ping response');
          }
        } catch (error) {
          // If ping fails, reinject the content script and wait for it to load
          console.log("Content script not ready, reinjecting...");
          try {
            await chrome.scripting.executeScript({
              target: { tabId },
              files: ['dist/content.js']
            });
            console.log("Content script reinjected successfully");
          } catch (injectionError) {
            console.error("Failed to inject content script:", injectionError);
            throw new Error('Failed to inject content script. Please refresh the page.');
          }
          
          // Wait longer for script to initialize (1000ms instead of 500ms)
          console.log("Waiting for content script to initialize...");
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log("Requesting market data...");
        // Now try to get market data with timeout
        const response = await Promise.race([
          chrome.tabs.sendMessage(tabId, { type: "GET_MARKET_DATA" }),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Data fetch timeout - please try refreshing the page')), 5000)
          )
        ]);

        console.log("Raw response from content script:", response);

        if (!response) {
          throw new Error("No data received from page. Please make sure you're on a stock details page.");
        }

        if ('error' in response) {
          throw new Error(response.error);
        }

        if (!isValidMarketData(response)) {
          console.error("Invalid market data structure:", response);
          throw new Error("Received malformed data from page. Expected price, symbol, and volume.");
        }

        console.log("Received valid market data:", response);
        setMarketData(response);
        setError(null);
      } catch (err) {
        console.error("Error:", err);
        setError(err instanceof Error ? err.message : "An unexpected error occurred");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ width: 350, p: 2, bgcolor: 'background.default', minHeight: 200 }}>
        <Typography variant="h6" gutterBottom>
          Market Data
        </Typography>
        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={140}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        ) : marketData ? (
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
          </Paper>
        ) : (
          <Alert severity="info" sx={{ mt: 2 }}>
            No market data available. Please navigate to a supported financial page.
          </Alert>
        )}
      </Box>
    </ThemeProvider>
  );
};

export default Popup;
