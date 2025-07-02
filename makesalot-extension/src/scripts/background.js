// background.js - Service Worker for MakesALot Extension

console.log('MakesALot Background Script Loading...');

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
    console.log('MakesALot Extension installed:', details.reason);
    
    // Set initial badge color
    try {
        chrome.action.setBadgeBackgroundColor({ color: '#667eea' });
    } catch (error) {
        console.log('Badge API error:', error);
    }
});

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background received message:', message);
    
    try {
        switch (message.type) {
            case 'SYMBOL_DETECTED':
                handleSymbolDetection(message, sender.tab);
                sendResponse({ success: true });
                break;
                
            case 'GET_SYMBOL':
                getDetectedSymbol(sender.tab?.id, sendResponse);
                return true; // Keep message channel open for async response
                
            case 'CLEAR_SYMBOL':
                clearSymbolForTab(sender.tab?.id);
                sendResponse({ success: true });
                break;
                
            default:
                console.log('Unknown message type:', message.type);
                sendResponse({ error: 'Unknown message type' });
        }
    } catch (error) {
        console.error('Error handling message:', error);
        sendResponse({ error: error.message });
    }
});

// Handle symbol detection from content scripts
function handleSymbolDetection(message, tab) {
    if (!tab || !tab.id) {
        console.error('Invalid tab information');
        return;
    }

    const { symbol, source, url, hostname } = message;
    
    console.log(`Symbol detected: ${symbol} from ${source} on ${hostname}`);
    
    try {
        // Store symbol data
        const storageData = {
            [`symbol_${tab.id}`]: symbol,
            [`source_${tab.id}`]: source,
            [`url_${tab.id}`]: url,
            [`detected_at_${tab.id}`]: Date.now(),
            // Global storage for popup access
            'detected_symbol': symbol,
            'detected_source': source,
            'detected_url': url,
            'detected_at': Date.now()
        };
        
        chrome.storage.local.set(storageData, () => {
            if (chrome.runtime.lastError) {
                console.error('Storage error:', chrome.runtime.lastError);
            } else {
                console.log('Symbol stored successfully');
                updateBadgeForSymbol(symbol, source, tab.id);
            }
        });
        
    } catch (error) {
        console.error('Error storing symbol:', error);
    }
}

// Update badge with detected symbol
function updateBadgeForSymbol(symbol, source, tabId) {
    try {
        // Truncate symbol for badge (max 4 characters)
        const badgeText = symbol && symbol.length > 4 ? symbol.substring(0, 4) : symbol || '';
        
        chrome.action.setBadgeText({
            text: badgeText,
            tabId: tabId
        });
        
        // Color based on source
        const sourceColors = {
            'yahoo': '#7C3AED',
            'tradingview': '#2563EB', 
            'investing': '#059669',
            'marketwatch': '#DC2626',
            'unknown': '#6B7280'
        };
        
        chrome.action.setBadgeBackgroundColor({
            color: sourceColors[source] || sourceColors.unknown,
            tabId: tabId
        });
        
        // Update title with detected info
        const title = symbol ? 
            `MakesALot: ${symbol} detected from ${source}` : 
            'MakesALot Trading Assistant';
            
        chrome.action.setTitle({
            title: title,
            tabId: tabId
        });
        
        console.log(`Badge updated for tab ${tabId}: ${badgeText}`);
        
    } catch (error) {
        console.error('Error updating badge:', error);
    }
}

// Get detected symbol for a specific tab
function getDetectedSymbol(tabId, sendResponse) {
    if (!tabId) {
        sendResponse({ error: 'No tab ID provided' });
        return;
    }

    const keys = [
        `symbol_${tabId}`,
        `source_${tabId}`,
        `detected_at_${tabId}`,
        // Also get global storage
        'detected_symbol',
        'detected_source',
        'detected_at'
    ];
    
    chrome.storage.local.get(keys, (result) => {
        if (chrome.runtime.lastError) {
            console.error('Storage retrieval error:', chrome.runtime.lastError);
            sendResponse({ error: chrome.runtime.lastError.message });
            return;
        }
        
        // Try tab-specific first, then global
        const symbol = result[`symbol_${tabId}`] || result.detected_symbol || null;
        const source = result[`source_${tabId}`] || result.detected_source || null;
        const detectedAt = result[`detected_at_${tabId}`] || result.detected_at || 0;
        
        sendResponse({ 
            symbol: symbol,
            source: source,
            detectedAt: detectedAt
        });
    });
}

// Clear symbol data for a tab
function clearSymbolForTab(tabId) {
    if (!tabId) return;
    
    const keysToRemove = [
        `symbol_${tabId}`,
        `source_${tabId}`,
        `url_${tabId}`,
        `detected_at_${tabId}`
    ];
    
    chrome.storage.local.remove(keysToRemove, () => {
        if (chrome.runtime.lastError) {
            console.error('Error clearing storage:', chrome.runtime.lastError);
        } else {
            console.log(`Cleared data for tab ${tabId}`);
        }
    });
    
    // Clear badge
    try {
        chrome.action.setBadgeText({ text: '', tabId });
        chrome.action.setTitle({ title: 'MakesALot Trading Assistant', tabId });
    } catch (error) {
        console.error('Error clearing badge:', error);
    }
}

// Clean up when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
    console.log(`Tab ${tabId} closed, cleaning up`);
    clearSymbolForTab(tabId);
});

// Handle tab updates (navigation)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'loading' && changeInfo.url) {
        try {
            const url = new URL(changeInfo.url);
            const isFinancialSite = [
                'finance.yahoo.com',
                'tradingview.com', 
                'investing.com',
                'marketwatch.com'
            ].some(site => url.hostname.includes(site));
            
            if (!isFinancialSite) {
                // Clear badge when leaving financial sites
                console.log(`Left financial site, clearing badge for tab ${tabId}`);
                chrome.action.setBadgeText({ text: '', tabId });
                chrome.action.setTitle({ title: 'MakesALot Trading Assistant', tabId });
            }
        } catch (error) {
            console.error('Error processing URL change:', error);
        }
    }
});

// Simple cleanup function that runs periodically
function cleanupOldData() {
    chrome.storage.local.get(null, (items) => {
        if (chrome.runtime.lastError) {
            console.error('Cleanup storage error:', chrome.runtime.lastError);
            return;
        }
        
        const now = Date.now();
        const sixHours = 6 * 60 * 60 * 1000; // 6 hours
        const keysToRemove = [];
        
        for (const [key, value] of Object.entries(items)) {
            if (key.startsWith('detected_at_') && now - value > sixHours) {
                const tabId = key.replace('detected_at_', '');
                keysToRemove.push(
                    `symbol_${tabId}`,
                    `source_${tabId}`,
                    `url_${tabId}`,
                    key
                );
            }
        }
        
        if (keysToRemove.length > 0) {
            chrome.storage.local.remove(keysToRemove, () => {
                if (chrome.runtime.lastError) {
                    console.error('Cleanup removal error:', chrome.runtime.lastError);
                } else {
                    console.log(`Cleaned up ${keysToRemove.length} old symbol detections`);
                }
            });
        }
    });
}

// Run cleanup every 30 minutes
setInterval(cleanupOldData, 30 * 60 * 1000);

console.log('MakesALot Background Script Loaded Successfully');