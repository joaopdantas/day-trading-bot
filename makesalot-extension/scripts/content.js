// Content Script for MakesALot Trading Assistant
// This script runs on trading platform pages to extract symbols and inject signals

console.log('MakesALot Trading Assistant content script loaded');

// Function to extract stock symbol from current page
function extractSymbolFromPage() {
  const url = window.location.href;
  let symbol = "";

  if (url.includes("finance.yahoo.com")) {
    const match = url.match(/quote\/([A-Z.]+)/);
    if (match) {
      symbol = match[1];
      console.log('Yahoo Finance symbol detected:', symbol);
    }
  } else if (url.includes("tradingview.com/symbols/")) {
    const match = url.match(/symbols\/([^\/]+)/);
    if (match) {
      symbol = match[1].toUpperCase();
      console.log('TradingView symbol detected:', symbol);
    }
  } else if (url.includes("investing.com")) {
    const match = url.match(/equities\/([^\/]+)/);
    if (match) {
      symbol = match[1].replace("-", "_").toUpperCase();
      console.log('Investing.com symbol detected:', symbol);
    }
  }

  return symbol;
}

// Function to inject trading signal overlay (future feature)
function injectTradingSignal(signal) {
  // This will be implemented when we add signal display on pages
  console.log('Would inject signal:', signal);
}

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getSymbol') {
    const symbol = extractSymbolFromPage();
    sendResponse({ symbol: symbol });
  }
  
  if (request.action === 'displaySignal') {
    injectTradingSignal(request.signal);
    sendResponse({ success: true });
  }
});

// Auto-detect symbol when page loads
document.addEventListener('DOMContentLoaded', () => {
  const symbol = extractSymbolFromPage();
  if (symbol) {
    // Send symbol to background script
    chrome.runtime.sendMessage({
      type: 'SYMBOL_DETECTED',
      symbol: symbol,
      url: window.location.href
    });
  }
});