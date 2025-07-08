// content.js - Symbol Detection for MakesALot Extension

class SymbolDetector {
    constructor() {
        this.detectedSymbol = null;
        this.init();
    }

    init() {
        console.log('MakesALot Symbol Detector initialized on:', window.location.hostname);
        this.detectSymbol();
        
        // Watch for page changes
        this.observeChanges();
    }

    detectSymbol() {
        const hostname = window.location.hostname;
        let symbol = null;

        try {
            if (hostname.includes('finance.yahoo.com')) {
                symbol = this.detectYahooSymbol();
            } else if (hostname.includes('tradingview.com')) {
                symbol = this.detectTradingViewSymbol();
            }

            if (symbol && symbol !== this.detectedSymbol) {
                this.detectedSymbol = symbol;
                this.notifyBackground(symbol);
            }
        } catch (error) {
            console.error('Symbol detection error:', error);
        }
    }

    detectYahooSymbol() {
        // Try URL first
        const urlMatch = window.location.pathname.match(/\/quote\/([A-Z]+)/);
        if (urlMatch) {
            return urlMatch[1];
        }

        // Try page selectors
        const selectors = [
            'h1[data-reactid]',
            '.D\\(ib\\).Fz\\(18px\\)',
            '[data-symbol]'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const text = element.textContent || element.getAttribute('data-symbol');
                    const match = text.match(/([A-Z]{1,5})/);
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

    detectTradingViewSymbol() {
        // Try URL first
        const urlMatch = window.location.pathname.match(/\/symbols\/([A-Z]+)/);
        if (urlMatch) {
            return urlMatch[1];
        }

        // Try page selectors
        const selectors = [
            '[data-name="legend-source-title"]',
            '.tv-symbol-header__first-line'
        ];

        for (const selector of selectors) {
            try {
                const element = document.querySelector(selector);
                if (element) {
                    const text = element.textContent;
                    const match = text.match(/([A-Z]{1,5})/);
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

    notifyBackground(symbol) {
        chrome.runtime.sendMessage({
            type: 'SYMBOL_DETECTED',
            symbol: symbol,
            url: window.location.href,
            timestamp: Date.now()
        });
    }

    observeChanges() {
        // Watch for URL changes in SPAs
        let currentUrl = window.location.href;
        
        const observer = new MutationObserver(() => {
            if (window.location.href !== currentUrl) {
                currentUrl = window.location.href;
                setTimeout(() => this.detectSymbol(), 1000);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Also listen for popstate events
        window.addEventListener('popstate', () => {
            setTimeout(() => this.detectSymbol(), 1000);
        });
    }
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new SymbolDetector();
    });
} else {
    new SymbolDetector();
}