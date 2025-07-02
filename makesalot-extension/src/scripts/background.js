// background.js - Service Worker for MakesALot Extension

chrome.runtime.onInstalled.addListener((details) => {
    console.log('MakesALot Extension installed:', details.reason);
});

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background received message:', message);
    
    switch (message.type) {
        case 'SYMBOL_DETECTED':
            handleSymbolDetection(message.symbol, sender.tab);
            break;
            
        case 'GET_SYMBOL':
            getDetectedSymbol(sender.tab.id, sendResponse);
            return true; // Keep message channel open
            
        default:
            console.log('Unknown message type:', message.type);
    }
});

// Handle symbol detection from content scripts
function handleSymbolDetection(symbol, tab) {
    console.log(`Symbol detected: ${symbol} on ${tab.url}`);
    
    // Store symbol for this tab
    chrome.storage.local.set({
        [`symbol_${tab.id}`]: symbol,
        [`detected_at_${tab.id}`]: Date.now()
    });
    
    // Update extension badge
    chrome.action.setBadgeText({
        text: symbol.substring(0, 4),
        tabId: tab.id
    });
    
    chrome.action.setBadgeBackgroundColor({
        color: '#667eea',
        tabId: tab.id
    });
}

// Get detected symbol for a tab
function getDetectedSymbol(tabId, sendResponse) {
    chrome.storage.local.get([`symbol_${tabId}`], (result) => {
        const symbol = result[`symbol_${tabId}`] || null;
        sendResponse({ symbol: symbol });
    });
}

// Clean up when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
    chrome.storage.local.remove([
        `symbol_${tabId}`,
        `detected_at_${tabId}`
    ]);
});

// Clear badge when navigating away
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'loading') {
        chrome.action.setBadgeText({ text: '', tabId });
    }
});