// content-utils.js - Utilitários para Content Script MakesALot

/**
 * Configurações e constantes para detecção de símbolos
 */
const MAKESALOT_CONFIG = {
    // APIs
    API_BASE_URL: 'https://makesalot-backend.onrender.com/api/v1',
    API_TIMEOUT: 10000,
    
    // Cache
    CACHE_DURATION: 300000, // 5 minutes
    MAX_CACHE_SIZE: 100,
    
    // Validation
    MIN_CONFIDENCE: 0.7,
    VALIDATION_RETRY_DELAY: 5000,
    
    // Symbols
    SYMBOL_PATTERN: /^[A-Z0-9.-]{1,10}$/,
    CRYPTO_SUFFIXES: ['-USD', '-BTC', '-ETH'],
    
    // Detection
    DETECTION_DELAY: 1500,
    RECHECK_INTERVAL: 120000, // 2 minutes
    
    // Sites configuration
    SUPPORTED_SITES: {
        'finance.yahoo.com': {
            priority: 1,
            patterns: ['/quote/', '/screener/', '/watchlists/'],
            confidence_boost: 0.1
        },
        'tradingview.com': {
            priority: 2,
            patterns: ['/symbols/', '/chart/', '/ideas/'],
            confidence_boost: 0.05
        },
        'investing.com': {
            priority: 3,
            patterns: ['/equities/', '/currencies/', '/crypto/'],
            confidence_boost: 0.0
        },
        'marketwatch.com': {
            priority: 4,
            patterns: ['/investing/stock/', '/investing/fund/'],
            confidence_boost: 0.0
        },
        'bloomberg.com': {
            priority: 2,
            patterns: ['/quote/', '/stocks/', '/markets/'],
            confidence_boost: 0.05
        },
        'cnbc.com': {
            priority: 3,
            patterns: ['/quotes/', '/investing/'],
            confidence_boost: 0.0
        }
    }
};

/**
 * Utility class for symbol validation and formatting
 */
class SymbolUtils {
    /**
     * Validate if a string is a valid stock symbol
     */
    static isValidSymbol(symbol) {
        if (!symbol || typeof symbol !== 'string') return false;
        
        const cleanSymbol = symbol.trim().toUpperCase();
        
        // Basic pattern check
        if (!MAKESALOT_CONFIG.SYMBOL_PATTERN.test(cleanSymbol)) {
            return false;
        }
        
        // Length check
        if (cleanSymbol.length < 1 || cleanSymbol.length > 10) {
            return false;
        }
        
        // Exclude common false positives
        const excludePatterns = [
            /^USD$/,
            /^EUR$/,
            /^GBP$/,
            /^JPY$/,
            /^CAD$/,
            /^AUD$/,
            /^\d+$/,
            /^[.-]+$/
        ];
        
        return !excludePatterns.some(pattern => pattern.test(cleanSymbol));
    }

    /**
     * Clean and format symbol
     */
    static formatSymbol(symbol) {
        if (!symbol) return null;
        
        let cleaned = symbol.trim().toUpperCase();
        
        // Remove common prefixes/suffixes
        cleaned = cleaned.replace(/^(STOCK:|SYMBOL:|TICKER:)/i, '');
        cleaned = cleaned.replace(/\s+/g, '');
        
        // Handle crypto symbols
        if (cleaned.includes('USD') && !cleaned.endsWith('-USD')) {
            if (cleaned.length > 3 && cleaned.endsWith('USD')) {
                const base = cleaned.slice(0, -3);
                if (this.isValidSymbol(base)) {
                    cleaned = `${base}-USD`;
                }
            }
        }
        
        return this.isValidSymbol(cleaned) ? cleaned : null;
    }

    /**
     * Determine symbol type
     */
    static getSymbolType(symbol) {
        if (!symbol) return 'unknown';
        
        if (symbol.endsWith('-USD') || symbol.endsWith('-BTC') || symbol.endsWith('-ETH')) {
            return 'crypto';
        }
        
        if (symbol.includes('^')) {
            return 'index';
        }
        
        if (symbol.length <= 3 && /^[A-Z]+$/.test(symbol)) {
            return 'major_stock';
        }
        
        if (symbol.length <= 5) {
            return 'stock';
        }
        
        return 'unknown';
    }

    /**
     * Get confidence boost based on symbol characteristics
     */
    static getConfidenceBoost(symbol, source) {
        let boost = 0;
        
        // Symbol type boost
        const type = this.getSymbolType(symbol);
        switch (type) {
            case 'major_stock':
                boost += 0.1;
                break;
            case 'crypto':
                boost += 0.05;
                break;
            case 'index':
                boost += 0.05;
                break;
        }
        
        // Source boost
        const siteConfig = MAKESALOT_CONFIG.SUPPORTED_SITES[source];
        if (siteConfig) {
            boost += siteConfig.confidence_boost;
        }
        
        // Known symbols boost
        const knownSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'];
        if (knownSymbols.includes(symbol)) {
            boost += 0.15;
        }
        
        return Math.min(boost, 0.3); // Cap at 30% boost
    }
}

/**
 * API Communication utilities
 */
class APIClient {
    constructor(baseUrl = MAKESALOT_CONFIG.API_BASE_URL) {
        this.baseUrl = baseUrl;
        this.requestCache = new Map();
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const cacheKey = `${url}_${JSON.stringify(options)}`;
        
        // Check cache first
        const cached = this.requestCache.get(cacheKey);
        if (cached && (Date.now() - cached.timestamp) < MAKESALOT_CONFIG.CACHE_DURATION) {
            return cached.data;
        }

        try {
            const response = await fetch(url, {
                method: options.method || 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'User-Agent': 'MakesALot-Extension/2.0',
                    ...options.headers
                },
                body: options.body ? JSON.stringify(options.body) : undefined,
                timeout: MAKESALOT_CONFIG.API_TIMEOUT
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Cache successful responses
            this.requestCache.set(cacheKey, {
                data,
                timestamp: Date.now()
            });

            // Limit cache size
            if (this.requestCache.size > MAKESALOT_CONFIG.MAX_CACHE_SIZE) {
                const firstKey = this.requestCache.keys().next().value;
                this.requestCache.delete(firstKey);
            }

            return data;
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    async validateSymbol(symbol) {
        try {
            return await this.makeRequest(`/utils/validate-symbol/${symbol}`);
        } catch (error) {
            return { is_valid: false, exists: false, error: error.message };
        }
    }

    async getQuickQuote(symbol) {
        try {
            return await this.makeRequest(`/quick-quote/${symbol}`);
        } catch (error) {
            return null;
        }
    }

    async getSimpleAnalysis(symbol, days = 30) {
        try {
            return await this.makeRequest(`/simple-analyze?symbol=${symbol}&days=${days}`);
        } catch (error) {
            return null;
        }
    }

    clearCache() {
        this.requestCache.clear();
    }
}

/**
 * Performance monitoring utilities
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            detectionTime: [],
            validationTime: [],
            apiResponseTime: [],
            errors: []
        };
    }

    startTimer(operation) {
        return {
            operation,
            startTime: performance.now(),
            end: () => {
                const duration = performance.now() - performance.now();
                this.recordMetric(operation, duration);
                return duration;
            }
        };
    }

    recordMetric(operation, duration) {
        if (this.metrics[operation]) {
            this.metrics[operation].push({
                duration,
                timestamp: Date.now()
            });

            // Keep only last 100 measurements
            if (this.metrics[operation].length > 100) {
                this.metrics[operation].shift();
            }
        }
    }

    recordError(error, context = '') {
        this.metrics.errors.push({
            error: error.message || error,
            context,
            timestamp: Date.now()
        });

        // Keep only last 50 errors
        if (this.metrics.errors.length > 50) {
            this.metrics.errors.shift();
        }
    }

    getAverageTime(operation) {
        const measurements = this.metrics[operation];
        if (!measurements || measurements.length === 0) return 0;

        const sum = measurements.reduce((acc, m) => acc + m.duration, 0);
        return sum / measurements.length;
    }

    getStats() {
        return {
            avgDetectionTime: this.getAverageTime('detectionTime'),
            avgValidationTime: this.getAverageTime('validationTime'),
            avgApiResponseTime: this.getAverageTime('apiResponseTime'),
            totalErrors: this.metrics.errors.length,
            recentErrors: this.metrics.errors.slice(-5)
        };
    }
}

/**
 * Storage utilities for extension data
 */
class StorageManager {
    static async setSymbolData(data) {
        try {
            await chrome.storage.local.set({
                'detected_symbol': data.symbol,
                'detected_source': data.source,
                'detected_confidence': data.confidence,
                'detected_url': data.url,
                'detected_at': data.timestamp,
                'detection_context': data.context || {}
            });
        } catch (error) {
            console.error('Error saving symbol data:', error);
        }
    }

    static async getSymbolData() {
        try {
            return await chrome.storage.local.get([
                'detected_symbol',
                'detected_source', 
                'detected_confidence',
                'detected_url',
                'detected_at',
                'detection_context'
            ]);
        } catch (error) {
            console.error('Error getting symbol data:', error);
            return {};
        }
    }

    static async setAnalysisCache(symbol, analysis) {
        try {
            const key = `analysis_${symbol}`;
            await chrome.storage.local.set({
                [key]: {
                    ...analysis,
                    cached_at: Date.now()
                }
            });
        } catch (error) {
            console.error('Error caching analysis:', error);
        }
    }

    static async getAnalysisCache(symbol) {
        try {
            const key = `analysis_${symbol}`;
            const result = await chrome.storage.local.get([key]);
            const cached = result[key];
            
            if (cached && (Date.now() - cached.cached_at) < MAKESALOT_CONFIG.CACHE_DURATION) {
                return cached;
            }
            
            return null;
        } catch (error) {
            console.error('Error getting cached analysis:', error);
            return null;
        }
    }

    static async clearOldData() {
        try {
            const keys = await chrome.storage.local.get();
            const now = Date.now();
            const toRemove = [];

            for (const [key, value] of Object.entries(keys)) {
                if (key.startsWith('analysis_') && value.cached_at) {
                    if (now - value.cached_at > MAKESALOT_CONFIG.CACHE_DURATION * 2) {
                        toRemove.push(key);
                    }
                }
            }

            if (toRemove.length > 0) {
                await chrome.storage.local.remove(toRemove);
                console.log(`Cleaned up ${toRemove.length} old cache entries`);
            }
        } catch (error) {
            console.error('Error cleaning up storage:', error);
        }
    }
}

/**
 * Site-specific detection strategies
 */
class DetectionStrategies {
    static getStrategiesForSite(hostname) {
        const strategies = [];

        if (hostname.includes('finance.yahoo.com')) {
            strategies.push(...this.getYahooStrategies());
        } else if (hostname.includes('tradingview.com')) {
            strategies.push(...this.getTradingViewStrategies());
        } else if (hostname.includes('investing.com')) {
            strategies.push(...this.getInvestingStrategies());
        } else if (hostname.includes('marketwatch.com')) {
            strategies.push(...this.getMarketWatchStrategies());
        } else if (hostname.includes('bloomberg.com')) {
            strategies.push(...this.getBloombergStrategies());
        } else if (hostname.includes('cnbc.com')) {
            strategies.push(...this.getCNBCStrategies());
        }

        // Add generic strategies as fallback
        strategies.push(...this.getGenericStrategies());

        return strategies;
    }

    static getYahooStrategies() {
        return [
            {
                name: 'yahoo_url_pattern',
                execute: () => {
                    const match = window.location.pathname.match(/\/quote\/([A-Z.-]+)/i);
                    return match ? { symbol: match[1].toUpperCase(), confidence: 0.95 } : null;
                }
            },
            {
                name: 'yahoo_data_attributes',
                execute: () => {
                    const elements = document.querySelectorAll('[data-symbol]');
                    for (const el of elements) {
                        const symbol = el.getAttribute('data-symbol');
                        if (SymbolUtils.isValidSymbol(symbol)) {
                            return { symbol: symbol.toUpperCase(), confidence: 0.9 };
                        }
                    }
                    return null;
                }
            },
            {
                name: 'yahoo_title_extraction',
                execute: () => {
                    const match = document.title.match(/\(([A-Z.-]+)\)/);
                    if (match && SymbolUtils.isValidSymbol(match[1])) {
                        return { symbol: match[1], confidence: 0.8 };
                    }
                    return null;
                }
            }
        ];
    }

    static getTradingViewStrategies() {
        return [
            {
                name: 'tradingview_url_symbol',
                execute: () => {
                    const patterns = [
                        /\/symbols\/[^\/]*\/([A-Z.-]+)/i,
                        /\/chart\/([A-Z.-]+)/i
                    ];
                    
                    for (const pattern of patterns) {
                        const match = window.location.pathname.match(pattern);
                        if (match && SymbolUtils.isValidSymbol(match[1])) {
                            return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                        }
                    }
                    return null;
                }
            },
            {
                name: 'tradingview_legend',
                execute: () => {
                    const element = document.querySelector('[data-name="legend-source-title"]');
                    if (element) {
                        const match = element.textContent.match(/([A-Z.-]{1,10})/);
                        if (match && SymbolUtils.isValidSymbol(match[1])) {
                            return { symbol: match[1], confidence: 0.85 };
                        }
                    }
                    return null;
                }
            }
        ];
    }

    static getInvestingStrategies() {
        return [
            {
                name: 'investing_header',
                execute: () => {
                    const selectors = ['.instrumentHeader h1', '[data-test="instrument-header-title"]'];
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            const match = element.textContent.match(/\(([A-Z.-]+)\)/);
                            if (match && SymbolUtils.isValidSymbol(match[1])) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    }
                    return null;
                }
            }
        ];
    }

    static getMarketWatchStrategies() {
        return [
            {
                name: 'marketwatch_url',
                execute: () => {
                    const match = window.location.pathname.match(/\/investing\/stock\/([A-Z.-]+)/i);
                    if (match && SymbolUtils.isValidSymbol(match[1])) {
                        return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                    }
                    return null;
                }
            },
            {
                name: 'marketwatch_ticker',
                execute: () => {
                    const element = document.querySelector('.company__ticker');
                    if (element) {
                        const match = element.textContent.match(/([A-Z.-]{1,10})/);
                        if (match && SymbolUtils.isValidSymbol(match[1])) {
                            return { symbol: match[1], confidence: 0.85 };
                        }
                    }
                    return null;
                }
            }
        ];
    }

    static getBloombergStrategies() {
        return [
            {
                name: 'bloomberg_quote_url',
                execute: () => {
                    const match = window.location.pathname.match(/\/quote\/([A-Z.-]+)/i);
                    if (match && SymbolUtils.isValidSymbol(match[1])) {
                        return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                    }
                    return null;
                }
            }
        ];
    }

    static getCNBCStrategies() {
        return [
            {
                name: 'cnbc_quotes_url',
                execute: () => {
                    const match = window.location.pathname.match(/\/quotes\/([A-Z.-]+)/i);
                    if (match && SymbolUtils.isValidSymbol(match[1])) {
                        return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                    }
                    return null;
                }
            }
        ];
    }

    static getGenericStrategies() {
        return [
            {
                name: 'generic_meta_tags',
                execute: () => {
                    const metaTags = document.querySelectorAll('meta[property*="symbol"], meta[name*="symbol"]');
                    for (const tag of metaTags) {
                        const content = tag.getAttribute('content');
                        if (content && SymbolUtils.isValidSymbol(content)) {
                            return { symbol: content.toUpperCase(), confidence: 0.7 };
                        }
                    }
                    return null;
                }
            },
            {
                name: 'generic_json_ld',
                execute: () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    for (const script of scripts) {
                        try {
                            const data = JSON.parse(script.textContent);
                            if (data.tickerSymbol && SymbolUtils.isValidSymbol(data.tickerSymbol)) {
                                return { symbol: data.tickerSymbol.toUpperCase(), confidence: 0.8 };
                            }
                        } catch (e) {
                            continue;
                        }
                    }
                    return null;
                }
            }
        ];
    }
}

// Export utilities for use in content script
if (typeof window !== 'undefined') {
    window.MakesALotUtils = {
        MAKESALOT_CONFIG,
        SymbolUtils,
        APIClient,
        PerformanceMonitor,
        StorageManager,
        DetectionStrategies
    };
}