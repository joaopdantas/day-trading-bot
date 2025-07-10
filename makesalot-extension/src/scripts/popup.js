// MakesALot Trading Extension - API Integrated Version
class TradingAssistant {
    constructor() {
        this.apiConfig = {
            baseUrl: 'https://makesalot-backend.onrender.com',
            timeout: 15000,
            retryAttempts: 2,
            endpoints: {
                health: '/health',
                analyze: '/api/v1/analyze',
                quote: '/api/v1/quote',
                simpleAnalyze: '/api/v1/simple-analyze',
                quickQuote: '/api/v1/quick-quote',
                chartData: '/api/v1/chart/data',
                strategies: '/api/v1/strategies',
                validateSymbol: '/api/v1/utils/validate-symbol'
            }
        };
        
        this.state = {
            currentPeriod: '3m',
            currentSymbol: 'MSFT',
            currentDataType: 'price',
            apiStatus: 'unknown',
            chartData: null,
            lastAnalysis: null,
            analysisCache: new Map()
        };
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiConnection();
        this.loadDetectedSymbol();
        this.initializeUI();
    }

    bindEvents() {
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyzeStock());
        
        // Symbol input events
        const symbolInput = document.getElementById('symbol');
        symbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.analyzeStock();
        });
        
        symbolInput.addEventListener('input', (e) => {
            e.target.value = e.target.value.toUpperCase();
            this.state.currentSymbol = e.target.value;
            this.validateSymbolInput(e.target.value);
        });
        
        // API type change
        document.getElementById('apiType').addEventListener('change', () => {
            this.checkApiConnection();
        });
        
        // Data type change
        document.getElementById('dataType').addEventListener('change', (e) => {
            this.state.currentDataType = e.target.value;
            this.updateChartDisplay();
        });
        
        // Time period buttons
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.state.currentPeriod = e.target.dataset.period;
                this.updateChart();
            });
        });

        // Refresh button (if exists)
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }
    }

    initializeUI() {
        // Set initial UI state
        document.getElementById('symbol').value = this.state.currentSymbol;
        
        // Set default time period
        document.querySelector(`[data-period="${this.state.currentPeriod}"]`)?.classList.add('active');
        
        // Initialize tooltips if any
        this.initializeTooltips();
    }

    initializeTooltips() {
        // Add hover tooltips for indicators
        const indicators = document.querySelectorAll('.indicator');
        indicators.forEach(indicator => {
            const name = indicator.querySelector('.indicator-name')?.textContent;
            if (name) {
                const tooltips = {
                    'RSI (14)': 'Relative Strength Index: <30 oversold, >70 overbought',
                    'MACD': 'Moving Average Convergence Divergence: momentum indicator',
                    'Bollinger': 'Bollinger Bands: volatility and support/resistance levels'
                };
                
                if (tooltips[name]) {
                    indicator.title = tooltips[name];
                }
            }
        });
    }

    async loadDetectedSymbol() {
        try {
            // Get detected symbol from background script
            const response = await this.sendMessageToBackground({
                type: 'GET_SYMBOL'
            });

            if (response && response.symbol) {
                const detectedAt = response.detectedAt || 0;
                const now = Date.now();
                const fiveMinutes = 5 * 60 * 1000;

                // Only use if detected within last 5 minutes
                if (now - detectedAt < fiveMinutes) {
                    console.log(`✨ Using detected symbol: ${response.symbol} from ${response.source}`);
                    
                    // Set the symbol in input
                    document.getElementById('symbol').value = response.symbol;
                    this.state.currentSymbol = response.symbol;
                    
                    // Show detection notification
                    this.showDetectedSymbolStatus(response.symbol, response.source);
                    
                    // Auto-analyze if detected with high confidence
                    if (response.confidence && response.confidence > 0.8) {
                        setTimeout(() => this.analyzeStock(), 1000);
                    }
                }
            }

            // Listen for new symbol detection messages
            this.listenForSymbolUpdates();
        } catch (error) {
            console.error('❌ Error loading detected symbol:', error);
        }
    }

    showDetectedSymbolStatus(symbol, source) {
        const status = document.getElementById('apiStatus');
        const originalText = status.textContent;
        const originalClass = status.className;
        
        status.textContent = `✨ Auto-detected ${symbol} from ${source}`;
        status.className = 'api-status connected';
        
        // Add animation class if available
        status.classList.add('detection-flash');
        
        // Revert to original status after 3 seconds
        setTimeout(() => {
            status.textContent = originalText;
            status.className = originalClass;
            status.classList.remove('detection-flash');
        }, 3000);
    }

    listenForSymbolUpdates() {
        // Listen for storage changes (when new symbol is detected)
        chrome.storage.onChanged.addListener((changes, areaName) => {
            if (areaName === 'local' && changes.detected_symbol) {
                const newSymbol = changes.detected_symbol.newValue;
                const newSource = changes.detected_source?.newValue;
                const newConfidence = changes.detected_confidence?.newValue || 0.5;
                
                if (newSymbol && newSymbol !== this.state.currentSymbol) {
                    console.log(`🔄 New symbol detected: ${newSymbol} (confidence: ${newConfidence})`);
                    
                    // Update input
                    document.getElementById('symbol').value = newSymbol;
                    this.state.currentSymbol = newSymbol;
                    
                    // Show notification
                    this.showDetectedSymbolStatus(newSymbol, newSource || 'website');
                    
                    // Auto-analyze high confidence detections
                    if (newConfidence > 0.8) {
                        setTimeout(() => this.analyzeStock(), 1500);
                    }
                }
            }
        });
    }

    async checkApiConnection() {
        const status = document.getElementById('apiStatus');
        const apiType = document.getElementById('apiType').value;
        
        status.textContent = 'Checking connection...';
        status.className = 'api-status';
        
        try {
            if (apiType === 'makesalot') {
                const response = await this.testMakesALotAPI();
                if (response.connected) {
                    this.state.apiStatus = 'connected';
                    status.textContent = '✓ MakesALot API Connected';
                    status.classList.add('connected');
                    
                    // Show API info if available
                    if (response.info) {
                        console.log('📊 API Info:', response.info);
                    }
                } else {
                    throw new Error(response.error || 'API not responding');
                }
            } else {
                this.state.apiStatus = 'fallback';
                status.textContent = '✓ Fallback Mode Ready';
                status.classList.add('connected');
            }
        } catch (error) {
            this.state.apiStatus = 'disconnected';
            status.textContent = '⚠ API Unavailable - Using Fallback';
            status.classList.add('disconnected');
            console.warn('⚠️ API connection failed:', error.message);
        }
    }

    async testMakesALotAPI() {
        try {
            const response = await this.makeApiRequest('/health');
            return {
                connected: true,
                info: response
            };
        } catch (error) {
            // Try simple endpoint as fallback
            try {
                await this.makeApiRequest('/api/v1/stats');
                return { connected: true, info: { message: 'Basic connection OK' } };
            } catch (fallbackError) {
                return {
                    connected: false,
                    error: error.message
                };
            }
        }
    }

    async validateSymbolInput(symbol) {
        if (!symbol || symbol.length < 1) return;
        
        // Simple validation patterns
        const validPattern = /^[A-Z0-9.-]{1,10}$/;
        const symbolInput = document.getElementById('symbol');
        
        if (!validPattern.test(symbol)) {
            symbolInput.style.borderColor = '#ef4444';
            return false;
        } else {
            symbolInput.style.borderColor = '';
            return true;
        }
    }

    async analyzeStock() {
        const symbol = document.getElementById('symbol').value.trim();
        
        if (!symbol) {
            this.showError('Please enter a stock symbol');
            return;
        }

        if (!this.validateSymbolInput(symbol)) {
            this.showError('Invalid symbol format');
            return;
        }

        this.state.currentSymbol = symbol;
        this.showLoading(true);
        this.hideError();

        try {
            // Check cache first
            const cachedAnalysis = this.state.analysisCache.get(symbol);
            if (cachedAnalysis && (Date.now() - cachedAnalysis.timestamp) < 300000) { // 5 min cache
                console.log('📋 Using cached analysis for', symbol);
                await this.displayResults(cachedAnalysis.data);
                this.showLoading(false);
                return;
            }

            // Try MakesALot API first
            if (this.state.apiStatus === 'connected') {
                try {
                    const apiData = await this.fetchFromMakesALotAPI(symbol);
                    if (apiData) {
                        // Cache successful result
                        this.state.analysisCache.set(symbol, {
                            data: apiData,
                            timestamp: Date.now()
                        });
                        
                        await this.displayResults(apiData);
                        this.state.lastAnalysis = apiData;
                        return;
                    }
                } catch (apiError) {
                    console.warn('🔄 API failed, trying fallback:', apiError.message);
                }
            }

            // Fallback to mock data
            console.log('📊 Using fallback analysis for', symbol);
            const mockData = await this.generateEnhancedMockAnalysis(symbol);
            await this.displayResults(mockData);

        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.showError(`Failed to analyze ${symbol}. Please try again.`);
        } finally {
            this.showLoading(false);
        }
    }

    async fetchFromMakesALotAPI(symbol) {
        try {
            // Try advanced analysis first
            const analysisData = await this.makeApiRequest('/api/v1/analyze', {
                method: 'POST',
                body: {
                    symbol: symbol,
                    strategy: 'ml_trading',
                    days: 100,
                    timeframe: '1d',
                    include_predictions: true
                }
            });

            if (analysisData) {
                return this.formatAdvancedAPIResponse(analysisData);
            }
        } catch (error) {
            console.log('Advanced API failed, trying simple:', error.message);
        }

        // Fallback to simple analysis
        try {
            const simpleData = await this.makeApiRequest(
                `/api/v1/simple-analyze?symbol=${symbol}&days=100`
            );
            return this.formatSimpleAPIResponse(simpleData);
        } catch (error) {
            console.error('Both API endpoints failed:', error);
            throw error;
        }
    }

    async makeApiRequest(endpoint, options = {}) {
        const url = `${this.apiConfig.baseUrl}${endpoint}`;
        const requestOptions = {
            method: options.method || 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'MakesALot-Extension/2.0',
                ...options.headers
            },
            ...options
        };

        if (options.body && requestOptions.method !== 'GET') {
            requestOptions.body = JSON.stringify(options.body);
        }

        const response = await fetch(url, requestOptions);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    formatAdvancedAPIResponse(data) {
        return {
            symbol: data.symbol,
            current_price: data.current_price,
            change_percent: data.change_percent,
            trend: data.trend,
            recommendation: {
                action: data.recommendation.action,
                confidence: data.recommendation.confidence,
                reasoning: data.recommendation.reasoning,
                strategy_used: data.recommendation.strategy_used || 'ml_trading'
            },
            technical_indicators: data.technical_indicators,
            support_resistance: data.support_resistance,
            volume_analysis: data.volume_analysis,
            risk_assessment: data.risk_assessment,
            predictions: data.predictions,
            timestamp: data.timestamp,
            data_source: 'makesalot_api'
        };
    }

    formatSimpleAPIResponse(data) {
        return {
            symbol: data.symbol,
            current_price: data.current_price,
            change_percent: data.change_percent,
            trend: data.trend,
            recommendation: {
                action: data.recommendation?.toUpperCase() || 'HOLD',
                confidence: 0.7, // Default confidence for simple API
                reasoning: [`Technical analysis suggests ${data.recommendation || 'holding'} based on current indicators`],
                strategy_used: 'technical'
            },
            technical_indicators: {
                rsi: data.rsi || 50,
                macd: 0,
                macd_signal: 0,
                sma_20: data.sma_20 || data.current_price,
                sma_50: data.sma_50 || data.current_price
            },
            support_resistance: {
                support: [],
                resistance: []
            },
            volume_analysis: {
                current: data.volume || 1000000,
                average_20d: data.volume || 1000000,
                ratio: 1.0,
                trend: 'stable'
            },
            risk_assessment: 'medium',
            timestamp: new Date().toISOString(),
            data_source: 'makesalot_simple_api'
        };
    }

    async generateEnhancedMockAnalysis(symbol) {
        // Enhanced mock data generator with realistic patterns
        const baseData = this.generateMockMarketData(symbol);
        
        return {
            symbol: symbol,
            current_price: baseData.price,
            change_percent: baseData.change,
            trend: baseData.trend,
            recommendation: baseData.recommendation,
            technical_indicators: baseData.indicators,
            support_resistance: baseData.levels,
            volume_analysis: baseData.volume,
            risk_assessment: baseData.risk,
            timestamp: new Date().toISOString(),
            data_source: 'mock_data'
        };
    }

    generateMockMarketData(symbol) {
        // Base prices for known symbols
        const basePrices = {
            'MSFT': 350, 'AAPL': 180, 'GOOGL': 140, 'AMZN': 150,
            'TSLA': 200, 'NVDA': 800, 'META': 300, 'NFLX': 400,
            'SPY': 450, 'QQQ': 380, 'VOO': 400
        };

        const basePrice = basePrices[symbol] || (100 + Math.random() * 200);
        const change = (Math.random() - 0.5) * 6; // -3% to +3%
        const currentPrice = basePrice * (1 + change / 100);

        // Generate realistic RSI
        const rsi = 30 + Math.random() * 40; // 30-70 range
        
        // Determine trend and recommendation
        let trend, recommendation;
        if (change > 1 && rsi < 60) {
            trend = 'bullish';
            recommendation = { action: 'BUY', confidence: 0.7 + Math.random() * 0.2, reasoning: ['Strong upward momentum', 'RSI not overbought'] };
        } else if (change < -1 && rsi > 40) {
            trend = 'bearish';
            recommendation = { action: 'SELL', confidence: 0.6 + Math.random() * 0.2, reasoning: ['Downward pressure', 'Possible continuation'] };
        } else {
            trend = 'neutral';
            recommendation = { action: 'HOLD', confidence: 0.5 + Math.random() * 0.3, reasoning: ['Mixed signals', 'Wait for clearer direction'] };
        }

        return {
            price: currentPrice,
            change: change,
            trend: trend,
            recommendation: recommendation,
            indicators: {
                rsi: rsi,
                macd: (Math.random() - 0.5) * 2,
                macd_signal: (Math.random() - 0.5) * 1.5,
                sma_20: currentPrice * (0.98 + Math.random() * 0.04),
                sma_50: currentPrice * (0.96 + Math.random() * 0.08)
            },
            levels: {
                support: [currentPrice * 0.95, currentPrice * 0.92].sort((a, b) => b - a),
                resistance: [currentPrice * 1.05, currentPrice * 1.08].sort((a, b) => a - b)
            },
            volume: {
                current: Math.floor(500000 + Math.random() * 2000000),
                average_20d: Math.floor(800000 + Math.random() * 1500000),
                ratio: 0.8 + Math.random() * 0.4,
                trend: Math.random() > 0.5 ? 'increasing' : 'stable'
            },
            risk: Math.random() > 0.7 ? 'high' : Math.random() > 0.3 ? 'medium' : 'low'
        };
    }

    async displayResults(data) {
        console.log('📊 Displaying results for', data.symbol, data);
        
        try {
            // Display main recommendation
            this.displayRecommendation(data);
            
            // Display price information
            this.displayPriceInfo(data);
            
            // Display technical indicators
            this.displayTechnicalIndicators(data);
            
            // Update chart if data available
            if (data.chart_data || this.shouldGenerateChart()) {
                await this.updateChartWithData(data);
            }
            
            // Display additional info
            this.displayAdditionalInfo(data);
            
            // Show results container
            this.showResults();
            
        } catch (error) {
            console.error('❌ Error displaying results:', error);
            this.showError('Error displaying analysis results');
        }
    }

    displayRecommendation(data) {
        const recommendation = data.recommendation;
        const confidence = Math.round((recommendation.confidence || 0.5) * 100);
        
        document.getElementById('recommendationText').textContent = recommendation.action;
        document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
        
        // Create analysis summary
        const reasoning = Array.isArray(recommendation.reasoning) ? 
            recommendation.reasoning.join('. ') : 
            recommendation.reasoning || 'Analysis based on current market conditions';
            
        const strategy = recommendation.strategy_used ? 
            ` Strategy: ${recommendation.strategy_used}` : '';
            
        document.getElementById('analysisSummary').textContent = `${reasoning}.${strategy}`;
        
        // Set recommendation card style
        const card = document.getElementById('recommendationCard');
        card.className = `recommendation ${recommendation.action.toLowerCase()}`;
        
        // Add confidence-based styling
        if (confidence >= 80) {
            card.classList.add('high-confidence');
        } else if (confidence < 60) {
            card.classList.add('low-confidence');
        }
    }

    displayPriceInfo(data) {
        // Current price
        document.getElementById('currentPrice').textContent = `${data.current_price.toFixed(2)}`;
        
        // Price change
        const changeEl = document.getElementById('priceChange');
        const changeText = `${data.change_percent >= 0 ? '+' : ''}${data.change_percent.toFixed(2)}%`;
        changeEl.textContent = changeText;
        changeEl.className = `price-change ${data.change_percent >= 0 ? 'positive' : 'negative'}`;

        // Update price date context
        const periodText = this.getPeriodText(this.state.currentPeriod);
        const priceDate = document.getElementById('priceDate');
        if (priceDate) {
            priceDate.textContent = `vs ${periodText} ago`;
        }
    }

    displayTechnicalIndicators(data) {
        const indicators = data.technical_indicators;
        
        // RSI
        this.updateIndicatorDisplay('rsi', indicators.rsi, this.getRsiSignal(indicators.rsi));
        
        // MACD
        const macdValue = indicators.macd || 0;
        const macdSignal = indicators.macd_signal || 0;
        this.updateIndicatorDisplay('macd', macdValue, this.getMacdSignal(macdValue, macdSignal));
        
        // Bollinger Bands
        const bbPosition = this.calculateBbPosition(data.current_price, indicators);
        const bbSignal = this.getBbSignal(data.current_price, indicators);
        this.updateIndicatorDisplay('bb', bbPosition, bbSignal);
    }

    updateIndicatorDisplay(indicator, value, signal) {
        const valueEl = document.getElementById(`${indicator}Value`);
        const signalEl = document.getElementById(`${indicator}Signal`);
        
        if (valueEl) {
            let displayValue = typeof value === 'number' ? value.toFixed(2) : value;
            
            // Add context for specific indicators
            if (indicator === 'rsi') {
                if (value < 30) displayValue += ' (Oversold)';
                else if (value > 70) displayValue += ' (Overbought)';
                else if (value < 40) displayValue += ' (Weak)';
                else if (value > 60) displayValue += ' (Strong)';
            } else if (indicator === 'bb') {
                displayValue = (parseFloat(value) * 100).toFixed(0) + '%';
            }
            
            valueEl.textContent = displayValue;
        }
        
        if (signalEl) {
            signalEl.textContent = signal;
            signalEl.className = `indicator-signal signal-${signal.toLowerCase()}`;
        }
    }

    getRsiSignal(rsi) {
        if (rsi < 30) return 'BUY';
        if (rsi > 70) return 'SELL';
        if (rsi < 40) return 'WEAK BUY';
        if (rsi > 60) return 'WEAK SELL';
        return 'HOLD';
    }

    getMacdSignal(macd, macdSignal) {
        const diff = macd - macdSignal;
        if (diff > 0.5) return 'BUY';
        if (diff < -0.5) return 'SELL';
        if (diff > 0) return 'WEAK BUY';
        if (diff < 0) return 'WEAK SELL';
        return 'HOLD';
    }

    calculateBbPosition(price, indicators) {
        const upper = indicators.bb_upper;
        const lower = indicators.bb_lower;
        
        if (upper && lower && upper !== lower) {
            const position = (price - lower) / (upper - lower);
            return Math.max(0, Math.min(1, position));
        }
        return 0.5;
    }

    getBbSignal(price, indicators) {
        const position = this.calculateBbPosition(price, indicators);
        
        if (position <= 0.1) return 'BUY';
        if (position >= 0.9) return 'SELL';
        if (position <= 0.3) return 'WEAK BUY';
        if (position >= 0.7) return 'WEAK SELL';
        return 'HOLD';
    }

    displayAdditionalInfo(data) {
        // Support/Resistance levels
        if (data.support_resistance) {
            const support = data.support_resistance.support;
            const resistance = data.support_resistance.resistance;
            
            if (support && support.length > 0) {
                document.getElementById('lowPrice').textContent = `${support[0].toFixed(2)}`;
            }
            
            if (resistance && resistance.length > 0) {
                document.getElementById('highPrice').textContent = `${resistance[0].toFixed(2)}`;
            }
        }

        // Volume analysis
        if (data.volume_analysis) {
            const volume = data.volume_analysis;
            document.getElementById('avgVolume').textContent = this.formatVolume(volume.average_20d || volume.current);
        }

        // Risk assessment indicator
        this.updateRiskIndicator(data.risk_assessment);
    }

    updateRiskIndicator(riskLevel) {
        // Add risk indicator if element exists
        const riskEl = document.getElementById('riskIndicator');
        if (riskEl) {
            riskEl.textContent = riskLevel || 'medium';
            riskEl.className = `risk-indicator risk-${riskLevel || 'medium'}`;
        }
    }

    shouldGenerateChart() {
        return this.state.currentDataType === 'price'; // Generate chart for price view
    }

    async updateChartWithData(data) {
        try {
            // Generate or use existing chart data
            const chartData = data.chart_data || this.generateChartData(data);
            
            this.state.chartData = {
                symbol: data.symbol,
                current_price: data.current_price,
                price_change: data.change_percent,
                data: chartData
            };

            this.updateChartDisplay(this.state.chartData);
        } catch (error) {
            console.error('❌ Chart update error:', error);
        }
    }

    generateChartData(data) {
        // Generate mock chart data based on current analysis
        const days = this.getPeriodDays(this.state.currentPeriod);
        const currentPrice = data.current_price;
        const priceChange = data.change_percent / 100;
        
        const chartData = [];
        const basePrice = currentPrice / (1 + priceChange);
        
        for (let i = 0; i < days; i++) {
            const progress = i / (days - 1);
            const noise = (Math.random() - 0.5) * 0.02; // 2% daily noise
            const trend = priceChange * progress;
            const price = basePrice * (1 + trend + noise);
            
            const high = price * (1 + Math.random() * 0.02);
            const low = price * (1 - Math.random() * 0.02);
            const volume = Math.floor(500000 + Math.random() * 2000000);
            
            chartData.push({
                date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toISOString(),
                open: i === 0 ? basePrice : chartData[i-1].close,
                high: Math.max(high, price),
                low: Math.min(low, price),
                close: price,
                volume: volume
            });
        }
        
        return chartData;
    }

    getPeriodDays(period) {
        const periodMap = {
            '1w': 7, '2w': 14, '1m': 30, '3m': 90,
            '6m': 180, '1y': 365, '2y': 730
        };
        return periodMap[period] || 90;
    }

    getPeriodText(period) {
        const periodMap = {
            '1w': '1 week', '2w': '2 weeks', '1m': '1 month', '3m': '3 months',
            '6m': '6 months', '1y': '1 year', '2y': '2 years'
        };
        return periodMap[period] || period;
    }

    updateChartDisplay(chartData = null) {
        if (!chartData) chartData = this.state.chartData;
        if (!chartData) return;

        const dataType = this.state.currentDataType;
        
        // Update chart title and labels
        document.getElementById('chartTitle').textContent = 
            `${dataType === 'price' ? 'Price' : 'Volume'} Chart - ${this.getPeriodText(this.state.currentPeriod)}`;

        // Update statistics
        this.updateChartStatistics(chartData, dataType);
        
        // Create/update SVG chart
        this.createSVGChart(chartData, dataType);
    }

    updateChartStatistics(chartData, dataType) {
        if (!chartData.data || chartData.data.length === 0) return;

        if (dataType === 'price') {
            const prices = chartData.data.map(d => d.close);
            const highPrice = Math.max(...prices);
            const lowPrice = Math.min(...prices);
            
            document.getElementById('highPrice').textContent = `${highPrice.toFixed(2)}`;
            document.getElementById('lowPrice').textContent = `${lowPrice.toFixed(2)}`;
        } else {
            const volumes = chartData.data.map(d => d.volume);
            const maxVolume = Math.max(...volumes);
            const minVolume = Math.min(...volumes);
            
            document.getElementById('highPrice').textContent = this.formatVolume(maxVolume);
            document.getElementById('lowPrice').textContent = this.formatVolume(minVolume);
        }
        
        // Average volume
        const avgVol = chartData.data.reduce((sum, d) => sum + d.volume, 0) / chartData.data.length;
        document.getElementById('avgVolume').textContent = this.formatVolume(avgVol);
    }

    createSVGChart(chartData, dataType = 'price') {
        const svg = document.getElementById('chartSvg');
        const chartContent = document.getElementById('chartContent');
        const chartGrid = document.getElementById('chartGrid');
        
        if (!svg || !chartContent || !chartData.data) return;
        
        // Clear previous content
        chartContent.innerHTML = '';
        chartGrid.innerHTML = '';
        
        const data = chartData.data;
        const width = 600;
        const height = 120;
        const padding = { top: 10, right: 10, bottom: 10, left: 10 };
        
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        // Prepare data values
        const values = data.map(d => dataType === 'price' ? d.close : d.volume);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue || 1;

        // Create grid lines
        this.createChartGrid(chartGrid, width, height, padding);

        // Create chart path
        this.createChartPath(chartContent, data, values, minValue, valueRange, width, height, padding, dataType);
    }

    createChartGrid(chartGrid, width, height, padding) {
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i * (height - padding.top - padding.bottom) / 4);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', padding.left);
            line.setAttribute('y1', y);
            line.setAttribute('x2', width - padding.right);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
            line.setAttribute('stroke-width', '1');
            chartGrid.appendChild(line);
        }
    }

    createChartPath(chartContent, data, values, minValue, valueRange, width, height, padding, dataType) {
        const pathData = [];
        const areaData = [];
        
        data.forEach((point, index) => {
            const x = padding.left + (index / (data.length - 1)) * (width - padding.left - padding.right);
            const value = values[index];
            const y = height - padding.bottom - ((value - minValue) / valueRange) * (height - padding.top - padding.bottom);
            
            pathData.push(`${index === 0 ? 'M' : 'L'} ${x} ${y}`);
            
            if (index === 0) {
                areaData.push(`M ${x} ${height - padding.bottom}`);
                areaData.push(`L ${x} ${y}`);
            } else {
                areaData.push(`L ${x} ${y}`);
            }
        });
        
        // Close area path
        areaData.push(`L ${width - padding.right} ${height - padding.bottom}`);
        areaData.push('Z');

        // Create area gradient
        const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        area.setAttribute('d', areaData.join(' '));
        area.setAttribute('fill', dataType === 'price' ? 'url(#priceGradient)' : 'url(#volumeGradient)');
        area.setAttribute('opacity', '0.3');
        chartContent.appendChild(area);

        // Create line
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathData.join(' '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-width', '2');
        
        // Determine line color based on trend
        const firstValue = values[0];
        const lastValue = values[values.length - 1];
        const trend = (lastValue - firstValue) / firstValue;
        
        if (trend > 0.02) {
            path.setAttribute('stroke', '#22c55e');
        } else if (trend < -0.02) {
            path.setAttribute('stroke', '#ef4444');
        } else {
            path.setAttribute('stroke', '#f59e0b');
        }
        
        chartContent.appendChild(path);
    }

    formatVolume(volume) {
        if (!volume) return '0';
        
        if (volume >= 1000000000) {
            return `${(volume / 1000000000).toFixed(1)}B`;
        } else if (volume >= 1000000) {
            return `${(volume / 1000000).toFixed(1)}M`;
        } else if (volume >= 1000) {
            return `${Math.round(volume / 1000)}K`;
        } else {
            return Math.round(volume).toString();
        }
    }

    async updateChart() {
        if (!this.state.currentSymbol) return;
        
        this.showLoading(true, 'Updating chart...');
        
        try {
            // If we have current analysis data, just regenerate chart
            if (this.state.lastAnalysis) {
                await this.updateChartWithData(this.state.lastAnalysis);
            } else {
                // Re-analyze for new period
                await this.analyzeStock();
                return;
            }
        } catch (error) {
            console.error('❌ Chart update error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async refreshData() {
        if (!this.state.currentSymbol) return;
        
        // Clear cache for current symbol
        this.state.analysisCache.delete(this.state.currentSymbol);
        
        // Re-analyze
        await this.analyzeStock();
    }

    async sendMessageToBackground(message) {
        return new Promise((resolve, reject) => {
            try {
                chrome.runtime.sendMessage(message, (response) => {
                    if (chrome.runtime.lastError) {
                        reject(new Error(chrome.runtime.lastError.message));
                    } else {
                        resolve(response);
                    }
                });
            } catch (error) {
                reject(error);
            }
        });
    }

    showLoading(show, message = 'Analyzing...') {
        const loading = document.getElementById('loading');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (show) {
            loading.style.display = 'block';
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = message;
            
            // Add loading spinner text if exists
            const spinnerText = loading.querySelector('span');
            if (spinnerText) {
                spinnerText.textContent = message;
            }
        } else {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '📊 Analyze Stock';
        }
    }

    showError(message) {
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        const errorDiv = document.getElementById('error');
        errorDiv.style.display = 'none';
    }

    showResults() {
        const results = document.getElementById('results');
        results.style.display = 'block';
        
        // Smooth scroll to results if needed
        results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Initialize when popup loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 MakesALot Trading Assistant loading...');
    try {
        new TradingAssistant();
        console.log('✅ Trading Assistant initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize Trading Assistant:', error);
    }
});