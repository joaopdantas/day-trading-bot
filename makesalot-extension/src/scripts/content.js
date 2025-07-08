// content.js - Symbol Detection for MakesALot Extension

class SymbolDetector {
    constructor() {
        this.detectedSymbol = null;
        this.detectedSource = null;
        this.init();
    }

    init() {
        console.log('MakesALot Symbol Detector initialized on:', window.location.hostname);
        this.detectSymbol();
        
        // Watch for page changes
        this.observeChanges();
        
        // Initial detection after page load
        setTimeout(() => this.detectSymbol(), 2000);
    }

    detectSymbol() {
        const hostname = window.location.hostname;
        let symbol = null;
        let source = 'unknown';

        try {
            if (hostname.includes('finance.yahoo.com')) {
                symbol = this.detectYahooSymbol();
                source = 'yahoo';
            } else if (hostname.includes('tradingview.com')) {
                symbol = this.detectTradingViewSymbol();
                source = 'tradingview';
            } else if (hostname.includes('investing.com')) {
                symbol = this.detectInvestingSymbol();
                source = 'investing';
            } else if (hostname.includes('marketwatch.com')) {
                symbol = this.detectMarketWatchSymbol();
                source = 'marketwatch';
            }

            if (symbol && symbol !== this.detectedSymbol) {
                this.detectedSymbol = symbol;
                this.detectedSource = source;
                console.log(`Symbol detected: ${symbol} from ${source}`);
                this.notifyBackground(symbol, source);
            }
        } catch (error) {
            console.error('Symbol detection error:', error);
        }
    }

    detectYahooSymbol() {
        // Try URL first - Yahoo Finance URLs: /quote/SYMBOL
        const urlMatch = window.location.pathname.match(/\/quote\/([A-Z.-]+)/i);
        if (urlMatch) {
            return urlMatch[1].toUpperCase();
        }

        // Try page selectors
        const selectors = [
            '[data-symbol]',
            'h1[data-reactid] span',
            '.D\\(ib\\).Fz\\(18px\\)',
            '[data-test="qsp-price-header"] h1',
            '.quote-header-section h1',
            '.companyName'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const symbol = element.getAttribute('data-symbol') || element.textContent;
                    if (symbol) {
                        const match = symbol.match(/([A-Z.-]{1,10})/);
                        if (match) {
                            return match[1];
                        }
                    }
                }
            } catch (e) {
                continue;
            }
        }

        // Try to find symbol in page title
        const titleMatch = document.title.match(/\(([A-Z.-]+)\)/);
        if (titleMatch) {
            return titleMatch[1];
        }

        return null;
    }

    detectTradingViewSymbol() {
        // Try URL first - TradingView URLs: /symbols/SYMBOL or /chart/SYMBOL
        const urlMatches = [
            window.location.pathname.match(/\/symbols\/[^\/]*\/([A-Z.-]+)/i),
            window.location.pathname.match(/\/chart\/([A-Z.-]+)/i),
            window.location.hash.match(/#([A-Z.-]+)/i)
        ];

        for (const match of urlMatches) {
            if (match) {
                return match[1].toUpperCase();
            }
        }

        // Try page selectors
        const selectors = [
            '[data-name="legend-source-title"]',
            '.tv-symbol-header__first-line',
            '.js-symbol-page-header-symbol',
            '.tv-category-header__title',
            '[class*="symbolName"]'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const text = element.textContent || element.innerText;
                    if (text) {
                        const match = text.match(/([A-Z.-]{1,10})/);
                        if (match) {
                            return match[1];
                        }
                    }
                }
            } catch (e) {
                continue;
            }
        }

        return null;
    }

    detectInvestingSymbol() {
        // Try URL and selectors for Investing.com
        const urlMatch = window.location.pathname.match(/\/equities\/[^\/]*-([a-z-]+)/i);
        if (urlMatch) {
            // Convert URL format to symbol (rough approximation)
            return urlMatch[1].toUpperCase().replace(/-/g, '');
        }

        const selectors = [
            '.instrumentHeader h1',
            '.float_lang_base_1 h1',
            '[data-test="instrument-header-title"]'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const text = element.textContent;
                    const match = text.match(/\(([A-Z.-]+)\)/);
                    if (match) {
                        return match[1];
                    }
                }
            } catch (e) {
                continue;
            }
        }

        return null;
    }

    detectMarketWatchSymbol() {
        // Try URL first
        const urlMatch = window.location.pathname.match(/\/investing\/stock\/([A-Z.-]+)/i);
        if (urlMatch) {
            return urlMatch[1].toUpperCase();
        }

        // Try page selectors
        const selectors = [
            '.company__ticker',
            '[data-module="Ticker"]',
            '.symbol'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const text = element.textContent;
                    const match = text.match(/([A-Z.-]{1,10})/);
                    if (match) {
                        return match[1];
                    }
                }
            } catch (e) {
                continue;
            }
        }

        return null;
    }

    notifyBackground(symbol, source) {
        // Send to background script
        chrome.runtime.sendMessage({
            type: 'SYMBOL_DETECTED',
            symbol: symbol,
            source: source,
            url: window.location.href,
            hostname: window.location.hostname,
            timestamp: Date.now()
        });

        // Also store in local storage for popup access
        chrome.storage.local.set({
            'detected_symbol': symbol,
            'detected_source': source,
            'detected_url': window.location.href,
            'detected_at': Date.now()
        });
    }

    observeChanges() {
        // Watch for URL changes in SPAs
        let currentUrl = window.location.href;
        
        const observer = new MutationObserver(() => {
            if (window.location.href !== currentUrl) {
                currentUrl = window.location.href;
                console.log('URL changed, re-detecting symbol');
                setTimeout(() => this.detectSymbol(), 1500);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Also listen for popstate events
        window.addEventListener('popstate', () => {
            console.log('Popstate event, re-detecting symbol');
            setTimeout(() => this.detectSymbol(), 1500);
        });

        // Listen for pushstate/replacestate (for SPAs)
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
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.symbolDetector = new SymbolDetector();
    });
} else {
    window.symbolDetector = new SymbolDetector();
}