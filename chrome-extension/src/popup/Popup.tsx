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
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (!tabs[0]?.id) {
          setError('No active tab found');
          setLoading(false);
          return;
        }

        const isFinancePage = tabs[0].url?.includes('finance.yahoo.com') || 
                            tabs[0].url?.includes('tradingview.com');
        
        if (!isFinancePage) {
          setError('Please navigate to Yahoo Finance or TradingView');
          setLoading(false);
          return;
        }

        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: 'GET_MARKET_DATA' },
          (response) => {
            if (chrome.runtime.lastError) {
              console.error('Runtime error:', chrome.runtime.lastError);
              setError('Could not connect to page. Please refresh and try again.');
              setLoading(false);
              return;
            }

            if (!response) {
              setError('No data received from page');
              setLoading(false);
              return;
            }

            setMarketData(response);
            setError(null);
            setLoading(false);
          }
        );
      } catch (err) {
        console.error('Error:', err);
        setError('An unexpected error occurred');
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
