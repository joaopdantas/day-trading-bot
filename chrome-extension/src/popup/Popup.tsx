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

const getMarketData = async () => {
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
      const response = await chrome.tabs.sendMessage(tabId, { type: "PING" });
      if (!response || response.type !== "PONG") {
        throw new Error('No response');
      }
    } catch (error) {
      // If ping fails, reinject the content script and wait for it to load
      console.log("Content script not ready, reinjecting...");
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ['content.js']
      });
      
      // Wait for script to initialize (500ms)
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Now try to get market data with timeout
    const data = await Promise.race([
      chrome.tabs.sendMessage(tabId, { type: "GET_MARKET_DATA" }),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Data fetch timeout')), 5000)
      )
    ]);

    if (!data) {
      throw new Error("No data received from page");
    }

    return data;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('An unexpected error occurred');
  }
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

const Popup: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getMarketData();
        setMarketData(data);
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
