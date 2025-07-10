// content.js - Symbol Detection Integrado com MakesALot API

class MakesALotSymbolDetector {
    constructor() {
        this.detectedSymbol = null;
        this.detectedSource = null;
        this.apiUrl = 'https://makesalot-backend.onrender.com/api/v1';
        this.lastValidationTime = 0;
        this.validationCache = new Map();
        this.init();
    }

    init() {
        console.log('🚀 MakesALot Symbol Detector initialized on:', window.location.hostname);
        
        // Detectar símbolo imediatamente
        this.detectSymbol();
        
        // Observar mudanças na página
        this.observeChanges();
        
        // Re-detectar após carregamento completo
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => this.detectSymbol(), 2000);
            });
        } else {
            setTimeout(() => this.detectSymbol(), 2000);
        }

        // Adicionar listener para mensagens da extensão
        this.setupMessageListeners();
        
        // Auto-validação periódica
        this.startPeriodicValidation();
    }

    detectSymbol() {
        const hostname = window.location.hostname;
        let symbol = null;
        let source = 'unknown';
        let confidence = 0;

        try {
            if (hostname.includes('finance.yahoo.com')) {
                const result = this.detectYahooSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'yahoo';
            } else if (hostname.includes('tradingview.com')) {
                const result = this.detectTradingViewSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'tradingview';
            } else if (hostname.includes('investing.com')) {
                const result = this.detectInvestingSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'investing';
            } else if (hostname.includes('marketwatch.com')) {
                const result = this.detectMarketWatchSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'marketwatch';
            } else if (hostname.includes('bloomberg.com')) {
                const result = this.detectBloombergSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'bloomberg';
            } else if (hostname.includes('cnbc.com')) {
                const result = this.detectCNBCSymbol();
                symbol = result.symbol;
                confidence = result.confidence;
                source = 'cnbc';
            }

            if (symbol && symbol !== this.detectedSymbol && confidence > 0.7) {
                this.detectedSymbol = symbol;
                this.detectedSource = source;
                
                console.log(`📊 Symbol detected: ${symbol} from ${source} (confidence: ${confidence})`);
                
                // Validar símbolo com a API antes de notificar
                this.validateAndNotify(symbol, source, confidence);
            }
        } catch (error) {
            console.error('❌ Symbol detection error:', error);
        }
    }

    detectYahooSymbol() {
        // Múltiplas estratégias para Yahoo Finance
        const strategies = [
            // URL pattern: /quote/SYMBOL
            () => {
                const urlMatch = window.location.pathname.match(/\/quote\/([A-Z.-]+)/i);
                return urlMatch ? { symbol: urlMatch[1].toUpperCase(), confidence: 0.95 } : null;
            },
            
            // Meta tags
            () => {
                const metaTags = document.querySelectorAll('meta[property="og:title"], meta[name="description"]');
                for (const tag of metaTags) {
                    const content = tag.getAttribute('content') || '';
                    const match = content.match(/\(([A-Z.-]{1,10})\)/);
                    if (match) {
                        return { symbol: match[1], confidence: 0.85 };
                    }
                }
                return null;
            },
            
            // Data attributes
            () => {
                const elements = document.querySelectorAll('[data-symbol], [data-reactid*="symbol"]');
                for (const el of elements) {
                    const symbol = el.getAttribute('data-symbol') || el.textContent;
                    if (symbol && /^[A-Z.-]{1,10}$/.test(symbol)) {
                        return { symbol: symbol.toUpperCase(), confidence: 0.9 };
                    }
                }
                return null;
            },
            
            // Page title
            () => {
                const titleMatch = document.title.match(/\(([A-Z.-]+)\)/);
                return titleMatch ? { symbol: titleMatch[1], confidence: 0.8 } : null;
            },
            
            // CSS selectors específicos do Yahoo
            () => {
                const selectors = [
                    'h1[data-reactid] span',
                    '.D\\(ib\\).Fz\\(18px\\)',
                    '[data-test="qsp-price"] h1',
                    '.quote-header h1'
                ];
                
                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || '';
                            const match = text.match(/([A-Z.-]{1,10})/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            }
        ];

        // Tentar todas as estratégias
        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    detectTradingViewSymbol() {
        const strategies = [
            // URL patterns
            () => {
                const patterns = [
                    /\/symbols\/[^\/]*\/([A-Z.-]+)/i,
                    /\/chart\/([A-Z.-]+)/i,
                    /#([A-Z.-]+)/i
                ];
                
                for (const pattern of patterns) {
                    const match = window.location.href.match(pattern);
                    if (match) {
                        return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                    }
                }
                return null;
            },
            
            // TradingView specific selectors
            () => {
                const selectors = [
                    '[data-name="legend-source-title"]',
                    '.tv-symbol-header__first-line',
                    '.js-symbol-page-header-symbol',
                    '.tv-category-header__title',
                    '[class*="symbolName"]',
                    '[class*="symbol-info"]'
                ];

                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || element.innerText || '';
                            const match = text.match(/([A-Z.-]{1,10})/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            },
            
            // Chart widget detection
            () => {
                const chartElements = document.querySelectorAll('[id*="chart"], [class*="chart"]');
                for (const el of chartElements) {
                    const dataAttrs = [...el.attributes].find(attr => 
                        attr.name.includes('symbol') || attr.value.match(/^[A-Z.-]{1,10}$/)
                    );
                    if (dataAttrs) {
                        const symbol = dataAttrs.value.match(/([A-Z.-]{1,10})/);
                        if (symbol) {
                            return { symbol: symbol[1], confidence: 0.8 };
                        }
                    }
                }
                return null;
            }
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    detectInvestingSymbol() {
        const strategies = [
            // URL patterns
            () => {
                const urlMatch = window.location.pathname.match(/\/equities\/[^\/]*-([a-z-]+)/i);
                if (urlMatch) {
                    // Converter URL slug para símbolo (aproximação)
                    const slug = urlMatch[1].toUpperCase().replace(/-/g, '');
                    return { symbol: slug, confidence: 0.7 };
                }
                return null;
            },
            
            // Page selectors
            () => {
                const selectors = [
                    '.instrumentHeader h1',
                    '.float_lang_base_1 h1',
                    '[data-test="instrument-header-title"]',
                    '.instrument-metadata h1'
                ];

                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || '';
                            const match = text.match(/\(([A-Z.-]+)\)/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            }
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    detectMarketWatchSymbol() {
        const strategies = [
            // URL pattern
            () => {
                const urlMatch = window.location.pathname.match(/\/investing\/stock\/([A-Z.-]+)/i);
                return urlMatch ? { symbol: urlMatch[1].toUpperCase(), confidence: 0.9 } : null;
            },
            
            // Page selectors
            () => {
                const selectors = [
                    '.company__ticker',
                    '[data-module="Ticker"]',
                    '.symbol',
                    '.quote__ticker'
                ];

                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || '';
                            const match = text.match(/([A-Z.-]{1,10})/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            }
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    detectBloombergSymbol() {
        const strategies = [
            // URL patterns
            () => {
                const patterns = [
                    /\/quote\/([A-Z.-]+)/i,
                    /\/stocks\/([A-Z.-]+)/i
                ];
                
                for (const pattern of patterns) {
                    const match = window.location.pathname.match(pattern);
                    if (match) {
                        return { symbol: match[1].toUpperCase(), confidence: 0.9 };
                    }
                }
                return null;
            },
            
            // Bloomberg specific selectors
            () => {
                const selectors = [
                    '[data-module="QuoteHeader"] h1',
                    '.security-name',
                    '.quote-header__name'
                ];

                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || '';
                            const match = text.match(/([A-Z.-]{1,10})/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.8 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            }
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    detectCNBCSymbol() {
        const strategies = [
            // URL patterns
            () => {
                const urlMatch = window.location.pathname.match(/\/quotes\/([A-Z.-]+)/i);
                return urlMatch ? { symbol: urlMatch[1].toUpperCase(), confidence: 0.9 } : null;
            },
            
            // CNBC specific selectors
            () => {
                const selectors = [
                    '.QuoteStrip-name',
                    '.QuoteHeader-name',
                    '.quote-strip__name'
                ];

                for (const selector of selectors) {
                    try {
                        const element = document.querySelector(selector);
                        if (element) {
                            const text = element.textContent || '';
                            const match = text.match(/\(([A-Z.-]+)\)/);
                            if (match) {
                                return { symbol: match[1], confidence: 0.85 };
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
                return null;
            }
        ];

        for (const strategy of strategies) {
            const result = strategy();
            if (result && result.symbol) {
                return result;
            }
        }

        return { symbol: null, confidence: 0 };
    }

    async validateAndNotify(symbol, source, confidence) {
        try {
            // Check cache first
            const cacheKey = `${symbol}_${source}`;
            const cached = this.validationCache.get(cacheKey);
            const now = Date.now();
            
            if (cached && (now - cached.timestamp) < 300000) { // 5 min cache
                if (cached.isValid) {
                    this.notifyDetection(symbol, source, confidence, cached.data);
                }
                return;
            }

            // Validate with MakesALot API
            console.log(`🔍 Validating ${symbol} with MakesALot API...`);
            
            const isValid = await this.validateSymbolWithAPI(symbol);
            
            // Cache result
            this.validationCache.set(cacheKey, {
                isValid,
                timestamp: now,
                data: isValid ? { symbol, source, confidence } : null
            });

            if (isValid) {
                console.log(`✅ ${symbol} validated successfully`);
                this.notifyDetection(symbol, source, confidence);
                
                // Optional: Get quick analysis
                this.getQuickAnalysis(symbol);
            } else {
                console.log(`❌ ${symbol} validation failed`);
            }

        } catch (error) {
            console.error('❌ Validation error:', error);
            // If validation fails, still notify (might be API issue)
            this.notifyDetection(symbol, source, confidence * 0.8);
        }
    }

    async validateSymbolWithAPI(symbol) {
        try {
            // Try validation endpoint first
            const validationUrl = `${this.apiUrl}/utils/validate-symbol/${symbol}`;
            const response = await fetch(validationUrl, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'User-Agent': 'MakesALot-Extension/2.0'
                },
                timeout: 5000
            });

            if (response.ok) {
                const data = await response.json();
                return data.is_valid && data.exists;
            }

            // Fallback: try quick quote
            const quoteUrl = `${this.apiUrl}/quick-quote/${symbol}`;
            const quoteResponse = await fetch(quoteUrl, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'User-Agent': 'MakesALot-Extension/2.0'
                },
                timeout: 5000
            });

            return quoteResponse.ok;

        } catch (error) {
            console.log(`⚠️ API validation failed for ${symbol}:`, error.message);
            return true; // Assume valid if API is down
        }
    }

    async getQuickAnalysis(symbol) {
        try {
            const analysisUrl = `${this.apiUrl}/simple-analyze?symbol=${symbol}&days=30`;
            const response = await fetch(analysisUrl, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                },
                timeout: 10000
            });

            if (response.ok) {
                const analysis = await response.json();
                console.log(`📊 Quick analysis for ${symbol}:`, {
                    price: analysis.current_price,
                    change: analysis.change_percent,
                    recommendation: analysis.recommendation,
                    rsi: analysis.rsi
                });

                // Store analysis for popup
                chrome.storage.local.set({
                    [`analysis_${symbol}`]: {
                        ...analysis,
                        timestamp: Date.now()
                    }
                });
            }
        } catch (error) {
            console.log(`⚠️ Quick analysis failed for ${symbol}:`, error.message);
        }
    }

    notifyDetection(symbol, source, confidence, additionalData = {}) {
        const notificationData = {
            type: 'SYMBOL_DETECTED',
            symbol: symbol,
            source: source,
            confidence: confidence,
            url: window.location.href,
            hostname: window.location.hostname,
            timestamp: Date.now(),
            ...additionalData
        };

        // Send to background script
        chrome.runtime.sendMessage(notificationData);

        // Store in local storage for popup access
        chrome.storage.local.set({
            'detected_symbol': symbol,
            'detected_source': source,
            'detected_confidence': confidence,
            'detected_url': window.location.href,
            'detected_at': Date.now()
        });

        // Dispatch custom event for other scripts
        window.dispatchEvent(new CustomEvent('makesalot:symbol-detected', {
            detail: notificationData
        }));
    }

    setupMessageListeners() {
        // Listen for messages from popup or background
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            switch (message.type) {
                case 'GET_CURRENT_SYMBOL':
                    sendResponse({
                        symbol: this.detectedSymbol,
                        source: this.detectedSource,
                        url: window.location.href
                    });
                    break;
                    
                case 'FORCE_DETECTION':
                    this.detectSymbol();
                    sendResponse({ status: 'detection_triggered' });
                    break;
                    
                case 'VALIDATE_SYMBOL':
                    this.validateAndNotify(message.symbol, 'manual', 1.0);
                    sendResponse({ status: 'validation_started' });
                    break;
            }
        });
    }

    startPeriodicValidation() {
        // Re-validate every 2 minutes if symbol detected
        setInterval(() => {
            if (this.detectedSymbol && this.detectedSource) {
                const timeSinceDetection = Date.now() - this.lastValidationTime;
                if (timeSinceDetection > 120000) { // 2 minutes
                    this.validateAndNotify(this.detectedSymbol, this.detectedSource, 0.9);
                    this.lastValidationTime = Date.now();
                }
            }
        }, 120000);
    }

    observeChanges() {
        let currentUrl = window.location.href;
        
        // Create a more efficient observer
        const observer = new MutationObserver((mutations) => {
            // Check for URL changes
            if (window.location.href !== currentUrl) {
                currentUrl = window.location.href;
                console.log('🔄 URL changed, re-detecting symbol');
                
                // Reset state
                this.detectedSymbol = null;
                this.detectedSource = null;
                
                // Re-detect after a short delay
                setTimeout(() => this.detectSymbol(), 1500);
            }
            
            // Check for significant DOM changes that might indicate new content
            let significantChange = false;
            for (const mutation of mutations) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            // Check if added node contains financial data indicators
                            const hasFinancialContent = node.textContent && (
                                /\$[\d,]+\.?\d*/.test(node.textContent) || // Price patterns
                                /[+-]?\d+\.?\d*%/.test(node.textContent) || // Percentage patterns
                                /[A-Z]{2,5}/.test(node.textContent) // Potential symbols
                            );
                            
                            if (hasFinancialContent || 
                                node.querySelector && (
                                    node.querySelector('[data-symbol]') ||
                                    node.querySelector('[class*="symbol"]') ||
                                    node.querySelector('[class*="ticker"]') ||
                                    node.querySelector('[class*="quote"]')
                                )) {
                                significantChange = true;
                                break;
                            }
                        }
                    }
                }
                if (significantChange) break;
            }
            
            if (significantChange) {
                console.log('📊 Financial content detected, re-checking symbol');
                setTimeout(() => this.detectSymbol(), 1000);
            }
        });

        // Start observing with optimized config
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false,
            characterData: false
        });

        // Listen for browser navigation events
        window.addEventListener('popstate', () => {
            console.log('🔙 Navigation event detected');
            setTimeout(() => this.detectSymbol(), 1500);
        });

        // Override pushState and replaceState for SPA detection
        const originalPushState = history.pushState;
        const originalReplaceState = history.replaceState;

        history.pushState = function() {
            originalPushState.apply(history, arguments);
            setTimeout(() => window.symbolDetector?.detectSymbol(), 1500);
        };

        history.replaceState = function() {
            originalReplaceState.apply(history, arguments);
            setTimeout(() => window.symbolDetector?.detectSymbol(), 1500);
        };
    }

    // Utility method to clean up
    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        
        // Clear cache
        this.validationCache.clear();
        
        // Reset state
        this.detectedSymbol = null;
        this.detectedSource = null;
    }
}

// Initialize when page loads
function initializeDetector() {
    // Only initialize on financial websites
    const financialSites = [
        'finance.yahoo.com',
        'tradingview.com',
        'investing.com',
        'marketwatch.com',
        'bloomberg.com',
        'cnbc.com'
    ];
    
    const isFinancialSite = financialSites.some(site => 
        window.location.hostname.includes(site)
    );
    
    if (isFinancialSite) {
        console.log('🌐 Initializing MakesALot detector on financial site');
        window.symbolDetector = new MakesALotSymbolDetector();
    } else {
        console.log('ℹ️ Not a financial site, skipping symbol detection');
    }
}

// Initialize based on document state
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDetector);
} else {
    initializeDetector();
}

// Global error handler
window.addEventListener('error', (event) => {
    if (event.error && event.error.message.includes('MakesALot')) {
        console.error('❌ MakesALot Symbol Detector Error:', event.error);
    }
});

// Export for potential external access
window.MakesALotSymbolDetector = MakesALotSymbolDetector;