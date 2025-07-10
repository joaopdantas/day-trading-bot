// background.js - Complete Service Worker for MakesALot Extension v2.1

console.log('🚀 MakesALot Background Script v2.1 Loading...');

// ===== CONFIGURATION =====
const CONFIG = {
    API_BASE_URL: 'https://makesalot-backend.onrender.com',
    API_TIMEOUT: 15000,
    CACHE_DURATION: 300000, // 5 minutes
    MAX_CACHE_SIZE: 50,
    RETRY_ATTEMPTS: 2,
    RETRY_DELAY: 2000,
    CLEANUP_INTERVAL: 1800000, // 30 minutes
    HEALTH_CHECK_INTERVAL: 300000, // 5 minutes
    SESSION_TIMEOUT: 3600000 // 1 hour
};

// ===== GLOBAL STATE =====
const state = {
    apiCache: new Map(),
    symbolData: new Map(),
    lastApiCheck: 0,
    apiStatus: 'unknown',
    sessionStart: Date.now(),
    analytics: {
        symbolDetections: 0,
        apiRequests: 0,
        successfulAnalyses: 0,
        errors: 0,
        cacheHits: 0
    }
};

// ===== INSTALLATION & STARTUP =====
chrome.runtime.onInstalled.addListener((details) => {
    console.log('📦 MakesALot Extension installed:', details.reason);
    
    try {
        // Set initial badge configuration
        chrome.action.setBadgeBackgroundColor({ color: '#667eea' });
        chrome.action.setBadgeText({ text: '' });
        chrome.action.setTitle({ title: 'MakesALot Trading Assistant' });
        
        // Initialize context menus
        createContextMenus();
        
        // Setup alarms for periodic tasks
        setupPeriodicTasks();
        
        // Initial cleanup
        setTimeout(cleanupOldData, 5000);
        
        console.log('✅ Extension initialized successfully');
    } catch (error) {
        console.error('❌ Initialization error:', error);
        trackEvent('error', { context: 'initialization', error: error.message });
    }
});

// ===== MESSAGE HANDLING =====
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('📨 Background received message:', message.type, 'from tab:', sender.tab?.id);
    
    try {
        switch (message.type) {
            case 'SYMBOL_DETECTED':
                handleSymbolDetection(message, sender.tab);
                sendResponse({ success: true });
                break;
                
            case 'GET_SYMBOL':
                getDetectedSymbol(sender.tab?.id, sendResponse);
                return true; // Keep channel open for async response
                
            case 'CLEAR_SYMBOL':
                clearSymbolForTab(sender.tab?.id);
                sendResponse({ success: true });
                break;
                
            case 'TEST_API':
                testAPIConnection(sendResponse);
                return true; // Keep channel open
                
            case 'FETCH_ANALYSIS':
                fetchAnalysisFromAPI(message.symbol, message.options || {}, sendResponse);
                return true; // Keep channel open
                
            case 'FETCH_QUOTE':
                fetchQuoteFromAPI(message.symbol, sendResponse);
                return true; // Keep channel open
                
            case 'VALIDATE_SYMBOL':
                validateSymbolWithAPI(message.symbol, sendResponse);
                return true; // Keep channel open
                
            case 'GET_ANALYTICS':
                sendResponse(getAnalytics());
                break;
                
            case 'CLEAR_CACHE':
                state.apiCache.clear();
                sendResponse({ success: true, message: 'Cache cleared' });
                break;
                
            default:
                console.log('❓ Unknown message type:', message.type);
                sendResponse({ error: 'Unknown message type', type: message.type });
        }
    } catch (error) {
        console.error('❌ Error handling message:', error);
        trackEvent('error', { context: 'message_handling', error: error.message });
        sendResponse({ error: error.message });
    }
});

// ===== SYMBOL DETECTION HANDLING =====
function handleSymbolDetection(message, tab) {
    if (!tab || !tab.id) {
        console.error('❌ Invalid tab information');
        return;
    }

    const { symbol, source, url, hostname, confidence } = message;
    
    console.log(`📊 Symbol detected: ${symbol} from ${source} on ${hostname} (confidence: ${confidence || 'unknown'})`);
    
    try {
        // Track analytics
        trackEvent('symbol_detected', { symbol, source, confidence });
        
        // Create symbol info object
        const symbolInfo = {
            symbol,
            source,
            url,
            hostname,
            confidence: confidence || 0.8,
            detected_at: Date.now(),
            tab_id: tab.id
        };
        
        // Store in memory
        state.symbolData.set(tab.id, symbolInfo);
        
        // Store in chrome storage
        const storageData = {
            [`symbol_${tab.id}`]: symbol,
            [`source_${tab.id}`]: source,
            [`url_${tab.id}`]: url,
            [`detected_at_${tab.id}`]: Date.now(),
            [`confidence_${tab.id}`]: confidence || 0.8,
            // Global storage for popup access
            'detected_symbol': symbol,
            'detected_source': source,
            'detected_url': url,
            'detected_at': Date.now(),
            'detected_confidence': confidence || 0.8
        };
        
        chrome.storage.local.set(storageData).then(() => {
            console.log('💾 Symbol stored successfully');
            updateBadgeForSymbol(symbol, source, tab.id);
            
            // Preload analysis for high confidence detections
            if (confidence && confidence > 0.9) {
                preloadAnalysis(symbol);
            }
        }).catch(error => {
            console.error('❌ Storage error:', error);
            trackEvent('error', { context: 'storage', error: error.message });
        });
        
    } catch (error) {
        console.error('❌ Error storing symbol:', error);
        trackEvent('error', { context: 'symbol_detection', error: error.message });
    }
}

function updateBadgeForSymbol(symbol, source, tabId) {
    try {
        // Truncate symbol for badge (max 4 characters)
        const badgeText = symbol && symbol.length > 4 ? symbol.substring(0, 4) : symbol || '';
        
        chrome.action.setBadgeText({
            text: badgeText,
            tabId: tabId
        });
        
        // Color based on source reliability
        const sourceColors = {
            'yahoo': '#7C3AED',
            'tradingview': '#2563EB', 
            'investing': '#059669',
            'marketwatch': '#DC2626',
            'bloomberg': '#1F2937',
            'cnbc': '#EF4444',
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
        
        console.log(`🏷️ Badge updated for tab ${tabId}: ${badgeText}`);
        
    } catch (error) {
        console.error('❌ Error updating badge:', error);
        trackEvent('error', { context: 'badge_update', error: error.message });
    }
}

// ===== API COMMUNICATION =====
async function makeAPIRequest(endpoint, options = {}) {
    const url = `${CONFIG.API_BASE_URL}${endpoint}`;
    const cacheKey = `${url}_${JSON.stringify(options)}`;
    
    // Check cache first
    if (options.useCache !== false) {
        const cached = state.apiCache.get(cacheKey);
        if (cached && (Date.now() - cached.timestamp) < CONFIG.CACHE_DURATION) {
            console.log('📋 Using cached API response for', endpoint);
            trackEvent('cache_hit', { endpoint });
            state.analytics.cacheHits++;
            return cached.data;
        }
    }

    const requestOptions = {
        method: options.method || 'GET',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'MakesALot-Extension/2.1',
            ...options.headers
        },
        signal: AbortSignal.timeout(CONFIG.API_TIMEOUT)
    };

    if (options.body && requestOptions.method !== 'GET') {
        requestOptions.body = JSON.stringify(options.body);
    }

    let lastError;
    
    // Retry logic with exponential backoff
    for (let attempt = 1; attempt <= CONFIG.RETRY_ATTEMPTS; attempt++) {
        try {
            console.log(`🌐 API Request (attempt ${attempt}): ${endpoint}`);
            trackEvent('api_request', { endpoint, attempt });
            state.analytics.apiRequests++;
            
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Cache successful responses
            if (options.useCache !== false) {
                state.apiCache.set(cacheKey, {
                    data,
                    timestamp: Date.now()
                });

                // Limit cache size
                if (state.apiCache.size > CONFIG.MAX_CACHE_SIZE) {
                    const firstKey = state.apiCache.keys().next().value;
                    state.apiCache.delete(firstKey);
                }
            }

            console.log(`✅ API Request successful: ${endpoint}`);
            return data;
            
        } catch (error) {
            lastError = error;
            console.warn(`⚠️ API Request failed (attempt ${attempt}): ${error.message}`);
            trackEvent('api_error', { endpoint, attempt, error: error.message });
            
            if (attempt < CONFIG.RETRY_ATTEMPTS) {
                const delay = CONFIG.RETRY_DELAY * Math.pow(2, attempt - 1); // Exponential backoff
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }
    
    state.analytics.errors++;
    throw lastError;
}

async function testAPIConnection(sendResponse) {
    try {
        console.log('🔍 Testing API connection...');
        
        // Test basic health endpoint
        const healthData = await makeAPIRequest('/health', { useCache: false });
        
        state.apiStatus = 'connected';
        state.lastApiCheck = Date.now();
        
        sendResponse({
            connected: true,
            info: {
                status: healthData.status,
                service: healthData.service || 'MakesALot Trading API',
                timestamp: healthData.timestamp
            }
        });
        
        console.log('✅ API connection test passed');
        
    } catch (error) {
        // Try fallback endpoints
        try {
            await makeAPIRequest('/api/v1/stats', { useCache: false });
            
            state.apiStatus = 'limited';
            sendResponse({
                connected: true,
                info: { message: 'Limited API access available' }
            });
            
            console.log('⚠️ API connection limited but working');
            
        } catch (fallbackError) {
            state.apiStatus = 'disconnected';
            console.error('❌ API connection failed:', error.message);
            
            sendResponse({
                connected: false,
                error: error.message
            });
        }
    }
}

async function fetchAnalysisFromAPI(symbol, options = {}, sendResponse) {
    try {
        console.log(`🔍 Fetching analysis for ${symbol}`);
        
        // Try advanced analysis first
        try {
            const analysisData = await makeAPIRequest('/api/v1/analyze', {
                method: 'POST',
                body: {
                    symbol: symbol,
                    strategy: options.strategy || 'ml_trading',
                    days: options.days || 100,
                    timeframe: options.timeframe || '1d',
                    include_predictions: options.include_predictions !== false
                }
            });

            state.analytics.successfulAnalyses++;
            sendResponse({
                success: true,
                data: analysisData,
                source: 'advanced_api'
            });
            
            console.log(`✅ Advanced analysis successful for ${symbol}`);
            
        } catch (advancedError) {
            console.log('🔄 Advanced API failed, trying simple analysis...');
            
            // Fallback to simple analysis
            const simpleData = await makeAPIRequest(
                `/api/v1/simple-analyze?symbol=${symbol}&days=${options.days || 100}`
            );

            state.analytics.successfulAnalyses++;
            sendResponse({
                success: true,
                data: simpleData,
                source: 'simple_api'
            });
            
            console.log(`✅ Simple analysis successful for ${symbol}`);
        }
        
    } catch (error) {
        console.error(`❌ Analysis failed for ${symbol}:`, error);
        trackEvent('error', { context: 'fetch_analysis', symbol, error: error.message });
        
        sendResponse({
            success: false,
            error: error.message,
            fallback: true
        });
    }
}

async function fetchQuoteFromAPI(symbol, sendResponse) {
    try {
        console.log(`💰 Fetching quote for ${symbol}`);
        
        // Try enhanced quote first
        try {
            const quoteData = await makeAPIRequest(`/api/v1/quote/${symbol}`);
            
            sendResponse({
                success: true,
                data: quoteData,
                source: 'enhanced_quote'
            });
            
            console.log(`✅ Enhanced quote successful for ${symbol}`);
            
        } catch (enhancedError) {
            // Fallback to quick quote
            const quickData = await makeAPIRequest(`/api/v1/quick-quote/${symbol}`);
            
            sendResponse({
                success: true,
                data: quickData,
                source: 'quick_quote'
            });
            
            console.log(`✅ Quick quote successful for ${symbol}`);
        }
        
    } catch (error) {
        console.error(`❌ Quote failed for ${symbol}:`, error);
        trackEvent('error', { context: 'fetch_quote', symbol, error: error.message });
        
        sendResponse({
            success: false,
            error: error.message
        });
    }
}

async function validateSymbolWithAPI(symbol, sendResponse) {
    try {
        console.log(`✓ Validating symbol ${symbol}`);
        
        const validationData = await makeAPIRequest(`/api/v1/utils/validate-symbol/${symbol}`);
        
        sendResponse({
            success: true,
            data: validationData
        });
        
        console.log(`✅ Symbol validation successful for ${symbol}`);
        
    } catch (error) {
        console.error(`❌ Validation failed for ${symbol}:`, error);
        
        // Fallback: try quick quote to check if symbol exists
        try {
            await makeAPIRequest(`/api/v1/quick-quote/${symbol}`);
            
            sendResponse({
                success: true,
                data: {
                    is_valid: true,
                    exists: true,
                    fallback_validation: true
                }
            });
            
            console.log(`✅ Fallback validation successful for ${symbol}`);
            
        } catch (fallbackError) {
            trackEvent('error', { context: 'validate_symbol', symbol, error: error.message });
            
            sendResponse({
                success: false,
                error: error.message
            });
        }
    }
}

async function preloadAnalysis(symbol) {
    try {
        console.log(`⚡ Preloading analysis for ${symbol}`);
        
        // Preload in background without blocking
        await makeAPIRequest(`/api/v1/simple-analyze?symbol=${symbol}&days=50`);
        console.log(`✅ Analysis preloaded for ${symbol}`);
        
    } catch (error) {
        console.log(`⚠️ Preload failed for ${symbol}:`, error.message);
    }
}

// ===== SYMBOL DATA MANAGEMENT =====
function getDetectedSymbol(tabId, sendResponse) {
    try {
        // Try memory first for fast access
        const memoryData = state.symbolData.get(tabId);
        if (memoryData && (Date.now() - memoryData.detected_at) < CONFIG.SESSION_TIMEOUT) {
            sendResponse({
                symbol: memoryData.symbol,
                source: memoryData.source,
                confidence: memoryData.confidence,
                detectedAt: memoryData.detected_at
            });
            return;
        }

        // Fallback to chrome storage
        const keys = [
            `symbol_${tabId}`,
            `source_${tabId}`,
            `detected_at_${tabId}`,
            `confidence_${tabId}`,
            // Global fallback
            'detected_symbol',
            'detected_source',
            'detected_at',
            'detected_confidence'
        ];
        
        chrome.storage.local.get(keys, (result) => {
            if (chrome.runtime.lastError) {
                console.error('❌ Storage retrieval error:', chrome.runtime.lastError);
                sendResponse({ error: chrome.runtime.lastError.message });
                return;
            }
            
            // Try tab-specific first, then global
            const symbol = result[`symbol_${tabId}`] || result.detected_symbol || null;
            const source = result[`source_${tabId}`] || result.detected_source || null;
            const detectedAt = result[`detected_at_${tabId}`] || result.detected_at || 0;
            const confidence = result[`confidence_${tabId}`] || result.detected_confidence || 0.5;
            
            sendResponse({ 
                symbol: symbol,
                source: source,
                confidence: confidence,
                detectedAt: detectedAt
            });
        });
        
    } catch (error) {
        console.error('❌ Error getting symbol data:', error);
        trackEvent('error', { context: 'get_symbol', error: error.message });
        sendResponse({ error: error.message });
    }
}

function clearSymbolForTab(tabId) {
    if (!tabId) return;
    
    console.log(`🧹 Clearing symbol data for tab ${tabId}`);
    
    // Clear from memory
    state.symbolData.delete(tabId);
    
    // Clear from storage
    const keysToRemove = [
        `symbol_${tabId}`,
        `source_${tabId}`,
        `url_${tabId}`,
        `detected_at_${tabId}`,
        `confidence_${tabId}`
    ];
    
    chrome.storage.local.remove(keysToRemove, () => {
        if (chrome.runtime.lastError) {
            console.error('❌ Error clearing storage:', chrome.runtime.lastError);
        } else {
            console.log(`✅ Cleared data for tab ${tabId}`);
        }
    });
    
    // Clear badge
    try {
        chrome.action.setBadgeText({ text: '', tabId });
        chrome.action.setTitle({ title: 'MakesALot Trading Assistant', tabId });
    } catch (error) {
        console.error('❌ Error clearing badge:', error);
    }
}

// ===== TAB MANAGEMENT =====
chrome.tabs.onRemoved.addListener((tabId) => {
    console.log(`🗑️ Tab ${tabId} closed, cleaning up`);
    clearSymbolForTab(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'loading' && changeInfo.url) {
        try {
            const url = new URL(changeInfo.url);
            const isFinancialSite = [
                'finance.yahoo.com',
                'tradingview.com', 
                'investing.com',
                'marketwatch.com',
                'bloomberg.com',
                'cnbc.com'
            ].some(site => url.hostname.includes(site));
            
            if (!isFinancialSite) {
                // Clear badge when leaving financial sites
                console.log(`🌐 Left financial site, clearing badge for tab ${tabId}`);
                chrome.action.setBadgeText({ text: '', tabId });
                chrome.action.setTitle({ title: 'MakesALot Trading Assistant', tabId });
                
                // Keep data for a short while in case user returns
                setTimeout(() => {
                    const data = state.symbolData.get(tabId);
                    if (data && Date.now() - data.detected_at > 600000) { // 10 minutes
                        clearSymbolForTab(tabId);
                    }
                }, 600000);
            }
        } catch (error) {
            console.error('❌ Error processing URL change:', error);
        }
    }
});

// ===== PERIODIC CLEANUP =====
function cleanupOldData() {
    console.log('🧹 Starting cleanup...');
    
    // Clean API cache
    const now = Date.now();
    let cacheCleared = 0;
    
    for (const [key, value] of state.apiCache.entries()) {
        if (now - value.timestamp > CONFIG.CACHE_DURATION) {
            state.apiCache.delete(key);
            cacheCleared++;
        }
    }
    
    // Clean memory symbol data
    let symbolsCleared = 0;
    for (const [tabId, data] of state.symbolData.entries()) {
        if (now - data.detected_at > CONFIG.SESSION_TIMEOUT) {
            state.symbolData.delete(tabId);
            symbolsCleared++;
        }
    }
    
    // Clean chrome storage
    chrome.storage.local.get(null, (items) => {
        if (chrome.runtime.lastError) {
            console.error('❌ Cleanup storage error:', chrome.runtime.lastError);
            return;
        }
        
        const sixHours = 6 * 60 * 60 * 1000;
        const keysToRemove = [];
        
        for (const [key, value] of Object.entries(items)) {
            if (key.startsWith('detected_at_') && now - value > sixHours) {
                const tabId = key.replace('detected_at_', '');
                keysToRemove.push(
                    `symbol_${tabId}`,
                    `source_${tabId}`,
                    `url_${tabId}`,
                    `confidence_${tabId}`,
                    key
                );
            }
        }
        
        if (keysToRemove.length > 0) {
            chrome.storage.local.remove(keysToRemove, () => {
                if (chrome.runtime.lastError) {
                    console.error('❌ Cleanup removal error:', chrome.runtime.lastError);
                } else {
                    console.log(`🗑️ Cleaned up ${keysToRemove.length} old storage entries`);
                }
            });
        }
    });
    
    console.log(`🧹 Cleanup completed: ${cacheCleared} cache entries, ${symbolsCleared} symbols cleared`);
}

// ===== ANALYTICS AND MONITORING =====
function trackEvent(event, data = {}) {
    switch (event) {
        case 'symbol_detected':
            state.analytics.symbolDetections++;
            break;
        case 'api_request':
            state.analytics.apiRequests++;
            break;
        case 'successful_analysis':
            state.analytics.successfulAnalyses++;
            break;
        case 'cache_hit':
            state.analytics.cacheHits++;
            break;
        case 'error':
        case 'api_error':
            state.analytics.errors++;
            break;
    }
    
    console.log(`📊 Event tracked: ${event}`, data);
}

function getAnalytics() {
    const sessionDuration = Date.now() - state.sessionStart;
    
    return {
        ...state.analytics,
        sessionDuration: Math.round(sessionDuration / 1000),
        sessionStart: state.sessionStart,
        apiStatus: state.apiStatus,
        cacheSize: state.apiCache.size,
        activeSymbols: state.symbolData.size,
        averageRequestsPerMinute: state.analytics.apiRequests > 0 ? 
            Math.round((state.analytics.apiRequests / sessionDuration) * 60000) : 0,
        successRate: state.analytics.apiRequests > 0 ? 
            Math.round((state.analytics.successfulAnalyses / state.analytics.apiRequests) * 100) : 0,
        cacheHitRate: state.analytics.apiRequests > 0 ?
            Math.round((state.analytics.cacheHits / state.analytics.apiRequests) * 100) : 0
    };
}

// ===== ALARMS AND PERIODIC TASKS =====
function setupPeriodicTasks() {
    // Clear existing alarms
    chrome.alarms.clearAll();
    
    // Setup new alarms
    chrome.alarms.create('cleanup', { periodInMinutes: 30 });
    chrome.alarms.create('api-health-check', { periodInMinutes: 5 });
    chrome.alarms.create('cache-optimization', { periodInMinutes: 10 });
    
    console.log('⏰ Periodic tasks scheduled');
}

chrome.alarms.onAlarm.addListener((alarm) => {
    console.log(`⏰ Alarm triggered: ${alarm.name}`);
    
    switch (alarm.name) {
        case 'cleanup':
            cleanupOldData();
            break;
            
        case 'api-health-check':
            performApiHealthCheck();
            break;
            
        case 'cache-optimization':
            optimizeCache();
            break;
    }
});

async function performApiHealthCheck() {
    try {
        if (Date.now() - state.lastApiCheck > CONFIG.HEALTH_CHECK_INTERVAL) {
            await makeAPIRequest('/health', { useCache: false });
            state.apiStatus = 'connected';
            state.lastApiCheck = Date.now();
            console.log('💚 API health check passed');
        }
    } catch (error) {
        state.apiStatus = 'disconnected';
        console.log('💔 API health check failed:', error.message);
    }
}

function optimizeCache() {
    const before = state.apiCache.size;
    
    // Remove expired entries
    const now = Date.now();
    for (const [key, value] of state.apiCache.entries()) {
        if (now - value.timestamp > CONFIG.CACHE_DURATION) {
            state.apiCache.delete(key);
        }
    }
    
    // If cache is still too large, remove oldest entries
    if (state.apiCache.size > CONFIG.MAX_CACHE_SIZE) {
        const entries = Array.from(state.apiCache.entries());
        entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
        
        const toRemove = entries.slice(0, state.apiCache.size - CONFIG.MAX_CACHE_SIZE);
        toRemove.forEach(([key]) => state.apiCache.delete(key));
    }
    
    const after = state.apiCache.size;
    if (before !== after) {
        console.log(`🗑️ Cache optimized: ${before} → ${after} entries`);
    }
}

// ===== KEYBOARD SHORTCUTS =====
chrome.commands.onCommand.addListener((command) => {
    console.log(`⌨️ Command received: ${command}`);
    
    switch (command) {
        case 'analyze-current-symbol':
            handleAnalyzeCurrentSymbol();
            break;
            
        case 'toggle-popup':
            handleTogglePopup();
            break;
    }
});

async function handleAnalyzeCurrentSymbol() {
    try {
        // Get current active tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) return;
        
        // Get detected symbol for this tab
        const symbolData = state.symbolData.get(tab.id);
        if (symbolData && symbolData.symbol) {
            console.log(`⚡ Quick analyzing: ${symbolData.symbol}`);
            
            // Trigger analysis via content script
            chrome.tabs.sendMessage(tab.id, {
                type: 'QUICK_ANALYZE',
                symbol: symbolData.symbol
            }).catch(() => {
                // If content script not available, open popup
                chrome.action.openPopup();
            });
        } else {
            // No symbol detected, open popup
            chrome.action.openPopup();
        }
    } catch (error) {
        console.error('❌ Error in analyze current symbol:', error);
    }
}

async function handleTogglePopup() {
    try {
        // This will open the popup
        chrome.action.openPopup();
    } catch (error) {
        console.error('❌ Error opening popup:', error);
    }
}

// ===== CONTEXT MENUS =====
function createContextMenus() {
    try {
        chrome.contextMenus.removeAll(() => {
            chrome.contextMenus.create({
                id: 'analyze-symbol',
                title: 'Analyze with MakesALot',
                contexts: ['selection'],
                documentUrlPatterns: [
                    'https://finance.yahoo.com/*',
                    'https://tradingview.com/*',
                    'https://www.investing.com/*',
                    'https://www.marketwatch.com/*',
                    'https://www.bloomberg.com/*',
                    'https://www.cnbc.com/*'
                ]
            });

            chrome.contextMenus.create({
                id: 'separator-1',
                type: 'separator',
                contexts: ['selection']
            });

            chrome.contextMenus.create({
                id: 'detect-symbol',
                title: 'Force Symbol Detection',
                contexts: ['page'],
                documentUrlPatterns: [
                    'https://finance.yahoo.com/*',
                    'https://tradingview.com/*',
                    'https://www.investing.com/*',
                    'https://www.marketwatch.com/*',
                    'https://www.bloomberg.com/*',
                    'https://www.cnbc.com/*'
                ]
            });
        });

        console.log('📋 Context menus created');
    } catch (error) {
        console.log('⚠️ Context menu creation failed:', error.message);
    }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
    try {
        switch (info.menuItemId) {
            case 'analyze-symbol':
                if (info.selectionText) {
                    const symbol = info.selectionText.trim().toUpperCase();
                    
                    // Validate basic symbol format
                    if (/^[A-Z0-9.-]{1,10}$/.test(symbol)) {
                        // Send to content script to trigger analysis
                        chrome.tabs.sendMessage(tab.id, {
                            type: 'ANALYZE_SYMBOL',
                            symbol: symbol
                        });
                        
                        console.log(`📊 Context menu analysis triggered for: ${symbol}`);
                        trackEvent('context_menu_analysis', { symbol });
                    }
                }
                break;
                
            case 'detect-symbol':
                // Force symbol detection
                chrome.tabs.sendMessage(tab.id, {
                    type: 'FORCE_DETECTION'
                });
                
                console.log('🔍 Forced symbol detection triggered');
                trackEvent('forced_detection', { tabId: tab.id });
                break;
        }
    } catch (error) {
        console.error('❌ Context menu click error:', error);
    }
});

// ===== WINDOW FOCUS TRACKING =====
chrome.windows.onFocusChanged.addListener((windowId) => {
    if (windowId === chrome.windows.WINDOW_ID_NONE) {
        // Browser lost focus
        console.log('👁️ Browser lost focus');
    } else {
        // Browser gained focus
        console.log('👁️ Browser gained focus');
        
        // Opportunity to refresh data if needed
        if (Date.now() - state.lastApiCheck > 600000) { // 10 minutes
            performApiHealthCheck();
        }
    }
});

// ===== STORAGE QUOTA MANAGEMENT =====
chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === 'local') {
        // Monitor storage usage
        chrome.storage.local.getBytesInUse(null, (bytesInUse) => {
            if (chrome.runtime.lastError) return;
            
            const maxBytes = chrome.storage.local.QUOTA_BYTES || 5242880; // 5MB default
            const usagePercent = Math.round((bytesInUse / maxBytes) * 100);
            
            if (usagePercent > 80) {
                console.warn(`⚠️ Storage usage high: ${usagePercent}% (${bytesInUse} bytes)`);
                // Trigger aggressive cleanup
                cleanupOldData();
            }
        });
    }
});

// ===== NOTIFICATION SYSTEM =====
async function showNotification(title, message, type = 'basic') {
    try {
        const notificationId = await chrome.notifications.create({
            type: type,
            iconUrl: 'icons/icon48.png',
            title: title,
            message: message
        });
        
        console.log(`🔔 Notification shown: ${notificationId}`);
        return notificationId;
    } catch (error) {
        console.log('⚠️ Notifications not available:', error.message);
        return null;
    }
}

// ===== ERROR HANDLING =====
self.addEventListener('error', (event) => {
    console.error('❌ Background script error:', event.error);
    trackEvent('error', { context: 'global', error: event.error?.message });
});

self.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Unhandled promise rejection:', event.reason);
    trackEvent('error', { context: 'promise_rejection', error: event.reason?.message });
});

// ===== EXTENSION UPDATE HANDLING =====
chrome.runtime.onUpdateAvailable.addListener((details) => {
    console.log('🔄 Extension update available:', details.version);
    
    // Show notification about update
    showNotification(
        'MakesALot Update Available',
        `Version ${details.version} is ready to install. Extension will restart automatically in 30 seconds.`,
        'basic'
    );
    
    // Auto-reload after a delay
    setTimeout(() => {
        chrome.runtime.reload();
    }, 30000); // 30 seconds delay
});

// ===== DEBUGGING AND DEVELOPMENT HELPERS =====
function setupDevelopmentHelpers() {
    if (chrome.runtime.getManifest().version_name?.includes('dev') || 
        chrome.runtime.getManifest().name.includes('Development')) {
        
        // Development mode helpers
        globalThis.makesALotDebug = {
            state,
            config: CONFIG,
            analytics: getAnalytics,
            clearCache: () => {
                state.apiCache.clear();
                console.log('🧹 Debug: Cache cleared');
            },
            clearSymbols: () => {
                state.symbolData.clear();
                console.log('🧹 Debug: Symbols cleared');
            },
            testSymbolDetection: (symbol, source = 'manual') => {
                handleSymbolDetection({
                    symbol,
                    source,
                    url: 'debug://test',
                    hostname: 'debug.test',
                    confidence: 0.9,
                    timestamp: Date.now()
                }, { id: 999 });
            },
            forceCleanup: cleanupOldData,
            getApiStatus: () => state.apiStatus,
            testApiRequest: async (endpoint) => {
                try {
                    const result = await makeAPIRequest(endpoint);
                    console.log('Debug API test result:', result);
                    return result;
                } catch (error) {
                    console.error('Debug API test error:', error);
                    throw error;
                }
            },
            triggerAlarm: (alarmName) => {
                chrome.alarms.create(alarmName, { when: Date.now() + 1000 });
            }
        };
        
        console.log('🔧 Development mode: Debug helpers available at globalThis.makesALotDebug');
        
        // Additional dev logging
        const originalConsoleLog = console.log;
        console.log = function(...args) {
            originalConsoleLog.apply(console, [new Date().toISOString(), ...args]);
        };
    }
}

// ===== FINAL INITIALIZATION =====
async function initializeExtension() {
    try {
        console.log('🔧 Initializing MakesALot Extension v2.1...');
        
        // Setup development helpers if in dev mode
        setupDevelopmentHelpers();
        
        // Check initial API status
        await performApiHealthCheck();
        
        // Clean up any orphaned data
        cleanupOldData();
        
        // Log initial analytics
        console.log('📊 Initial analytics:', getAnalytics());
        
        console.log('✅ Extension fully initialized');
        
    } catch (error) {
        console.error('❌ Initialization error:', error);
        trackEvent('error', { context: 'initialization', error: error.message });
    }
}

// ===== GRACEFUL SHUTDOWN =====
self.addEventListener('beforeunload', () => {
    console.log('👋 Background script shutting down...');
    
    // Log final analytics
    const finalAnalytics = getAnalytics();
    console.log('📊 Final session analytics:', finalAnalytics);
    
    // Clear sensitive data from memory
    state.apiCache.clear();
    state.symbolData.clear();
    
    console.log('✅ Shutdown complete');
});

// ===== STARTUP EXECUTION =====
console.log('✅ MakesALot Background Script Loaded Successfully');
console.log(`🔧 Configuration:`, {
    apiUrl: CONFIG.API_BASE_URL,
    cacheSize: CONFIG.MAX_CACHE_SIZE,
    cacheTTL: `${CONFIG.CACHE_DURATION / 1000}s`,
    retryAttempts: CONFIG.RETRY_ATTEMPTS,
    timeouts: `${CONFIG.API_TIMEOUT / 1000}s`
});

// Run initialization
initializeExtension();

// ===== EXPORT FOR TESTING =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        state,
        makeAPIRequest,
        handleSymbolDetection,
        cleanupOldData,
        getAnalytics,
        trackEvent,
        testAPIConnection,
        fetchAnalysisFromAPI
    };
}