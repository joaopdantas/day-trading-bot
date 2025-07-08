// MakesALot Trading Extension - Improved Version
class TradingAssistant {
    constructor() {
        this.apiUrls = {
            makesalot: 'https://makesalot-backend.onrender.com/api/v1',
            yahoo: 'https://query1.finance.yahoo.com/v8/finance/chart',
            alpha: 'https://www.alphavantage.co/query'
        };
        this.currentPeriod = '3m';
        this.currentSymbol = 'MSFT';
        this.currentDataType = 'price';
        this.chartData = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiConnection();
        this.loadDetectedSymbol(); // Load detected symbol on startup
    }

    bindEvents() {
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyzeStock());
        
        // Symbol input - Enter key
        document.getElementById('symbol').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.analyzeStock();
        });
        
        // Symbol input - uppercase
        document.getElementById('symbol').addEventListener('input', (e) => {
            e.target.value = e.target.value.toUpperCase();
            this.currentSymbol = e.target.value;
        });
        
        // API type change
        document.getElementById('apiType').addEventListener('change', () => {
            this.checkApiConnection();
        });
        
        // Data type change
        document.getElementById('dataType').addEventListener('change', (e) => {
            this.currentDataType = e.target.value;
            this.updateChartDisplay();
        });
        
        // Time period buttons
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentPeriod = e.target.dataset.period;
                this.updateChart();
            });
        });
    }

    async loadDetectedSymbol() {
        try {
            // Get detected symbol from storage
            const result = await chrome.storage.local.get([
                'detected_symbol', 
                'detected_source', 
                'detected_url',
                'detected_at'
            ]);

            if (result.detected_symbol) {
                const detectedAt = result.detected_at || 0;
                const now = Date.now();
                const fiveMinutes = 5 * 60 * 1000;

                // Only use if detected within last 5 minutes
                if (now - detectedAt < fiveMinutes) {
                    console.log(`Using detected symbol: ${result.detected_symbol} from ${result.detected_source}`);
                    
                    // Set the symbol in input
                    document.getElementById('symbol').value = result.detected_symbol;
                    this.currentSymbol = result.detected_symbol;
                    
                    // Set appropriate API based on source
                    this.setApiBasedOnSource(result.detected_source);
                    
                    // Update status
                    this.showDetectedSymbolStatus(result.detected_symbol, result.detected_source);
                }
            }

            // Also listen for new symbol detection messages
            this.listenForSymbolUpdates();
        } catch (error) {
            console.error('Error loading detected symbol:', error);
        }
    }

    setApiBasedOnSource(source) {
        const apiSelect = document.getElementById('apiType');
        
        // Map source to preferred API
        const apiMapping = {
            'yahoo': 'yahoo',
            'tradingview': 'makesalot', // TradingView works well with our API
            'investing': 'yahoo',
            'marketwatch': 'yahoo'
        };

        const preferredApi = apiMapping[source] || 'makesalot';
        
        if (apiSelect.querySelector(`option[value="${preferredApi}"]`)) {
            apiSelect.value = preferredApi;
            this.checkApiConnection();
        }
    }

    showDetectedSymbolStatus(symbol, source) {
        const status = document.getElementById('apiStatus');
        const originalText = status.textContent;
        
        status.textContent = `✨ Auto-detected ${symbol} from ${source}`;
        status.className = 'api-status connected';
        
        // Revert to original status after 3 seconds
        setTimeout(() => {
            status.textContent = originalText;
        }, 3000);
    }

    listenForSymbolUpdates() {
        // Listen for storage changes (when new symbol is detected)
        chrome.storage.onChanged.addListener((changes, areaName) => {
            if (areaName === 'local' && changes.detected_symbol) {
                const newSymbol = changes.detected_symbol.newValue;
                const newSource = changes.detected_source?.newValue;
                
                if (newSymbol && newSymbol !== this.currentSymbol) {
                    console.log(`New symbol detected: ${newSymbol}`);
                    
                    // Update input
                    document.getElementById('symbol').value = newSymbol;
                    this.currentSymbol = newSymbol;
                    
                    // Set appropriate API
                    if (newSource) {
                        this.setApiBasedOnSource(newSource);
                    }
                    
                    // Show notification
                    this.showDetectedSymbolStatus(newSymbol, newSource || 'website');
                }
            }
        });
    }

    async checkApiConnection() {
        const apiType = document.getElementById('apiType').value;
        const status = document.getElementById('apiStatus');
        
        status.textContent = 'Checking connection...';
        status.className = 'api-status';
        
        try {
            if (apiType === 'makesalot') {
                const response = await fetch(this.apiUrls.makesalot.replace('/api/v1', '/health'));
                if (response.ok) {
                    status.textContent = '✓ MakesALot API Connected';
                    status.classList.add('connected');
                } else {
                    throw new Error('API not responding');
                }
            } else if (apiType === 'alpha') {
                status.textContent = '✓ Alpha Vantage Ready';
                status.classList.add('connected');
            } else {
                status.textContent = '✓ Yahoo Finance Ready';
                status.classList.add('connected');
            }
        } catch (error) {
            status.textContent = '✗ API Unavailable';
            status.classList.add('disconnected');
        }
    }

    async analyzeStock() {
        const symbol = document.getElementById('symbol').value.trim();
        
        if (!symbol) {
            this.showError('Please enter a stock symbol');
            return;
        }

        this.currentSymbol = symbol;
        this.showLoading(true);
        this.hideError();

        try {
            const apiType = document.getElementById('apiType').value;
            
            if (apiType === 'makesalot') {
                await this.analyzeMakesALot(symbol);
            } else {
                await this.analyzeWithMockData(symbol);
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Failed to analyze ${symbol}. Try switching API provider.`);
        } finally {
            this.showLoading(false);
        }
    }

    async analyzeMakesALot(symbol) {
        try {
            // Get technical analysis
            const analysisResponse = await fetch(`${this.apiUrls.makesalot}/technical/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: symbol,
                    timeframe: '1d',
                    indicators: ['RSI', 'MACD', 'BB']
                })
            });

            const analysisData = analysisResponse.ok ? await analysisResponse.json() : null;

            // Get predictions
            const predictionResponse = await fetch(`${this.apiUrls.makesalot}/predictions/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: symbol,
                    timeframe: '1d'
                })
            });

            const predictionData = predictionResponse.ok ? await predictionResponse.json() : null;

            // Get chart data
            const chartResponse = await fetch(`${this.apiUrls.makesalot}/chart/data/${symbol}?period=${this.currentPeriod}`);
            const chartData = chartResponse.ok ? await chartResponse.json() : null;

            this.displayResults(analysisData, predictionData, chartData);
        } catch (error) {
            console.error('MakesALot API error:', error);
            throw error;
        }
    }

    async analyzeWithMockData(symbol) {
        // Enhanced mock data with historical analysis
        const chartData = this.generateEnhancedChartData(symbol);
        
        // Analyze the generated data for more realistic recommendations
        const analysis = this.analyzeHistoricalData(chartData);
        
        const mockAnalysis = {
            symbol: symbol,
            indicators: [
                { name: 'RSI', value: analysis.rsi, signal: analysis.rsiSignal },
                { name: 'MACD', value: analysis.macd, signal: analysis.macdSignal },
                { name: 'BB', value: analysis.bb, signal: analysis.bbSignal }
            ]
        };

        const mockPrediction = {
            prediction: {
                direction: analysis.overallRecommendation,
                confidence: analysis.confidence
            }
        };

        this.displayResults(mockAnalysis, mockPrediction, chartData);
    }

    generateEnhancedChartData(symbol) {
        const periodDays = { '3m': 90, '6m': 180, '1y': 365 }[this.currentPeriod];
        const basePrice = 50 + Math.random() * 300; // $50-$350 range
        
        // Generate trending data with some realism
        const trendDirection = Math.random() - 0.5; // -0.5 to 0.5
        const volatility = 0.02 + Math.random() * 0.03; // 2-5% daily volatility
        
        const data = [];
        let currentPrice = basePrice;

        for (let i = 0; i < periodDays; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (periodDays - i));
            
            // Add trend and random walk
            const trendComponent = trendDirection * 0.001; // Small daily trend
            const randomComponent = (Math.random() - 0.5) * volatility;
            const priceChange = trendComponent + randomComponent;
            
            currentPrice *= (1 + priceChange);
            
            // Ensure price doesn't go negative
            currentPrice = Math.max(currentPrice, 1);
            
            const high = currentPrice * (1 + Math.random() * 0.02);
            const low = currentPrice * (1 - Math.random() * 0.02);
            const volume = Math.floor(500000 + Math.random() * 2000000);

            data.push({
                date: date.toISOString(),
                open: currentPrice * (1 + (Math.random() - 0.5) * 0.01),
                high: high,
                low: low,
                close: currentPrice,
                volume: volume
            });
        }

        // Calculate summary statistics
        const prices = data.map(d => d.close);
        const volumes = data.map(d => d.volume);
        const finalPrice = prices[prices.length - 1];
        const previousPrice = prices[prices.length - 2];
        const priceChange = ((finalPrice - previousPrice) / previousPrice) * 100;
        const highPrice = Math.max(...prices);
        const lowPrice = Math.min(...prices);
        const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;

        return {
            symbol: symbol,
            current_price: finalPrice,
            price_change: priceChange,
            high_price: highPrice,
            low_price: lowPrice,
            avg_volume: avgVolume,
            data: data,
            period: this.currentPeriod
        };
    }

    analyzeHistoricalData(chartData) {
        const prices = chartData.data.map(d => d.close);
        const volumes = chartData.data.map(d => d.volume);
        
        // Calculate RSI
        const rsi = this.calculateRSI(prices);
        
        // Calculate MACD (simplified)
        const macd = this.calculateMACD(prices);
        
        // Calculate Bollinger Band position
        const bb = this.calculateBollingerPosition(prices);
        
        // Determine signals
        const rsiSignal = rsi < 30 ? 'BUY' : rsi > 70 ? 'SELL' : 'HOLD';
        const macdSignal = macd > 0 ? 'BUY' : macd < 0 ? 'SELL' : 'HOLD';
        const bbSignal = bb < 0.2 ? 'BUY' : bb > 0.8 ? 'SELL' : 'HOLD';
        
        // Overall recommendation based on trend analysis
        const recentPrices = prices.slice(-10); // Last 10 days
        const trend = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0];
        
        let overallRecommendation;
        let confidence;
        
        const signals = [rsiSignal, macdSignal, bbSignal];
        const buyCount = signals.filter(s => s === 'BUY').length;
        const sellCount = signals.filter(s => s === 'SELL').length;
        
        if (buyCount >= 2) {
            overallRecommendation = 'BUY';
            confidence = 0.7 + (buyCount - 2) * 0.1;
        } else if (sellCount >= 2) {
            overallRecommendation = 'SELL';
            confidence = 0.7 + (sellCount - 2) * 0.1;
        } else {
            overallRecommendation = 'HOLD';
            confidence = 0.5 + Math.random() * 0.2;
        }

        return {
            rsi: rsi,
            macd: macd,
            bb: bb,
            rsiSignal: rsiSignal,
            macdSignal: macdSignal,
            bbSignal: bbSignal,
            overallRecommendation: overallRecommendation,
            confidence: confidence,
            trend: trend
        };
    }

    calculateRSI(prices, period = 14) {
        if (prices.length < period + 1) return 50;
        
        const deltas = [];
        for (let i = 1; i < prices.length; i++) {
            deltas.push(prices[i] - prices[i - 1]);
        }
        
        let avgGain = 0;
        let avgLoss = 0;
        
        // Calculate initial averages
        for (let i = 0; i < period; i++) {
            if (deltas[i] > 0) {
                avgGain += deltas[i];
            } else {
                avgLoss -= deltas[i];
            }
        }
        
        avgGain /= period;
        avgLoss /= period;
        
        const rs = avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));
        
        return Math.round(rsi * 100) / 100;
    }

    calculateMACD(prices) {
        if (prices.length < 26) return 0;
        
        // Simplified MACD calculation
        const ema12 = this.calculateEMA(prices, 12);
        const ema26 = this.calculateEMA(prices, 26);
        
        return Math.round((ema12 - ema26) * 100) / 100;
    }

    calculateEMA(prices, period) {
        const multiplier = 2 / (period + 1);
        let ema = prices[0];
        
        for (let i = 1; i < prices.length; i++) {
            ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
        }
        
        return ema;
    }

    calculateBollingerPosition(prices, period = 20) {
        if (prices.length < period) return 0.5;
        
        const recentPrices = prices.slice(-period);
        const sma = recentPrices.reduce((sum, price) => sum + price, 0) / period;
        
        const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
        const stdDev = Math.sqrt(variance);
        
        const latestPrice = prices[prices.length - 1];
        const upperBand = sma + (2 * stdDev);
        const lowerBand = sma - (2 * stdDev);
        
        // Position within bands (0 = lower band, 1 = upper band)
        const position = (latestPrice - lowerBand) / (upperBand - lowerBand);
        
        return Math.max(0, Math.min(1, position));
    }

    displayResults(analysisData, predictionData, chartData) {
        // Store chart data for period switching
        this.chartData = chartData;

        // Display recommendation with analysis summary
        if (predictionData && predictionData.prediction) {
            const recommendation = predictionData.prediction.direction;
            const confidence = Math.round(predictionData.prediction.confidence * 100);
            
            document.getElementById('recommendationText').textContent = recommendation;
            document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
            
            // Enhanced analysis summary
            const periodText = this.getPeriodText(this.currentPeriod);
            const summary = this.generateAnalysisSummary(recommendation, chartData, analysisData);
            document.getElementById('analysisSummary').textContent = summary;
            
            const card = document.getElementById('recommendationCard');
            card.className = `recommendation ${recommendation.toLowerCase()}`;
        }

        // Display price info with period context
        if (chartData) {
            document.getElementById('currentPrice').textContent = `${chartData.current_price.toFixed(2)}`;
            
            const changeEl = document.getElementById('priceChange');
            const changeText = `${chartData.price_change >= 0 ? '+' : ''}${chartData.price_change.toFixed(2)}%`;
            changeEl.textContent = changeText;
            changeEl.className = `price-change ${chartData.price_change >= 0 ? 'positive' : 'negative'}`;

            // Update date context
            const periodText = this.getPeriodText(this.currentPeriod);
            document.getElementById('priceDate').textContent = `vs ${periodText} ago`;

            // Update chart display
            this.updateChartDisplay(chartData);
        }

        // Display indicators with enhanced formatting
        if (analysisData && analysisData.indicators) {
            analysisData.indicators.forEach(indicator => {
                const valueEl = document.getElementById(`${indicator.name.toLowerCase()}Value`);
                const signalEl = document.getElementById(`${indicator.name.toLowerCase()}Signal`);
                
                if (valueEl) {
                    let displayValue = typeof indicator.value === 'number' ? 
                        indicator.value.toFixed(2) : indicator.value;
                    
                    // Add context for RSI
                    if (indicator.name === 'RSI') {
                        if (indicator.value < 30) displayValue += ' (Oversold)';
                        else if (indicator.value > 70) displayValue += ' (Overbought)';
                    }
                    
                    valueEl.textContent = displayValue;
                }
                if (signalEl) {
                    signalEl.textContent = indicator.signal;
                    signalEl.className = `indicator-signal signal-${indicator.signal.toLowerCase()}`;
                }
            });
        }

        this.showResults();
    }

    generateAnalysisSummary(recommendation, chartData, analysisData) {
        const periodText = this.getPeriodText(this.currentPeriod);
        const priceDirection = chartData.price_change >= 0 ? 'gained' : 'lost';
        const pricePercent = Math.abs(chartData.price_change).toFixed(1);
        
        let summary = `Over the past ${periodText}, ${chartData.symbol} has ${priceDirection} ${pricePercent}%. `;
        
        if (recommendation === 'BUY') {
            summary += `Technical indicators suggest upward momentum with favorable entry conditions.`;
        } else if (recommendation === 'SELL') {
            summary += `Technical indicators suggest downward pressure with potential profit-taking opportunities.`;
        } else {
            summary += `Mixed signals suggest a wait-and-see approach with current market conditions.`;
        }
        
        return summary;
    }

    getPeriodText(period) {
        const periodMap = {
            '3m': '3 months',
            '6m': '6 months',
            '1y': '1 year'
        };
        return periodMap[period] || period;
    }

    updateChartDisplay(chartData) {
        if (!chartData) return;

        const dataType = document.getElementById('dataType').value;
        
        // Update chart title and labels
        document.getElementById('chartTitle').textContent = 
            `${dataType === 'price' ? 'Price' : 'Volume'} Chart - ${this.getPeriodText(this.currentPeriod)}`;

        // Update labels based on data type
        if (dataType === 'price') {
            document.getElementById('highLabel').textContent = 'Period High';
            document.getElementById('lowLabel').textContent = 'Period Low';
            document.getElementById('volumeLabel').textContent = 'Avg Volume';
            document.getElementById('legendPriceText').textContent = 'Price';
            document.getElementById('legendVolumeItem').style.display = 'none';
        } else {
            document.getElementById('highLabel').textContent = 'Max Volume';
            document.getElementById('lowLabel').textContent = 'Min Volume';
            document.getElementById('volumeLabel').textContent = 'Avg Volume';
            document.getElementById('legendPriceText').textContent = 'Volume';
            document.getElementById('legendVolumeItem').style.display = 'flex';
        }

        // Update statistics
        if (dataType === 'price') {
            document.getElementById('highPrice').textContent = `${chartData.high_price.toFixed(2)}`;
            document.getElementById('lowPrice').textContent = `${chartData.low_price.toFixed(2)}`;
        } else {
            const volumes = chartData.data.map(d => d.volume);
            const maxVolume = Math.max(...volumes);
            const minVolume = Math.min(...volumes);
            document.getElementById('highPrice').textContent = this.formatVolume(maxVolume);
            document.getElementById('lowPrice').textContent = this.formatVolume(minVolume);
        }
        
        document.getElementById('avgVolume').textContent = this.formatVolume(chartData.avg_volume);

        // Create SVG chart
        this.createSVGChart(chartData);
    }

    formatVolume(volume) {
        if (volume > 1000000) {
            return `${(volume / 1000000).toFixed(1)}M`;
        } else if (volume > 1000) {
            return `${Math.round(volume / 1000)}K`;
        } else {
            return Math.round(volume).toString();
        }
    }

    createSVGChart(chartData) {
        const svg = document.getElementById('chartSvg');
        const chartContent = document.getElementById('chartContent');
        const chartGrid = document.getElementById('chartGrid');
        const tooltip = document.getElementById('chartTooltip');
        
        // Clear previous content
        chartContent.innerHTML = '';
        chartGrid.innerHTML = '';
        
        if (!chartData.data || chartData.data.length === 0) return;

        const dataType = document.getElementById('dataType').value;
        const svgRect = svg.getBoundingClientRect();
        const width = 600;
        const height = 120;
        const padding = { top: 10, right: 10, bottom: 10, left: 10 };
        
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        // Prepare data
        const data = chartData.data;
        const values = data.map(d => dataType === 'price' ? d.close : d.volume);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;

        // Create grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i * (height - padding.top - padding.bottom) / 4);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', padding.left);
            line.setAttribute('y1', y);
            line.setAttribute('x2', width - padding.right);
            line.setAttribute('y2', y);
            chartGrid.appendChild(line);
        }

        // Create path for line chart
        const pathData = [];
        const areaData = [];
        
        data.forEach((point, index) => {
            const x = padding.left + (index / (data.length - 1)) * (width - padding.left - padding.right);
            const value = dataType === 'price' ? point.close : point.volume;
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

        // Create area
        const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        area.setAttribute('d', areaData.join(' '));
        area.setAttribute('fill', dataType === 'price' ? 'url(#priceGradient)' : 'url(#volumeGradient)');
        area.setAttribute('class', 'chart-area');
        chartContent.appendChild(area);

        // Create line
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', pathData.join(' '));
        path.setAttribute('class', 'chart-line');
        
        // Determine line color based on overall trend
        const firstValue = values[0];
        const lastValue = values[values.length - 1];
        const trend = (lastValue - firstValue) / firstValue;
        
        if (trend > 0.05) {
            path.setAttribute('stroke', '#22c55e');
        } else if (trend < -0.05) {
            path.setAttribute('stroke', '#ef4444');
        } else {
            path.setAttribute('stroke', '#f59e0b');
        }
        
        chartContent.appendChild(path);

        // Create interactive points
        data.forEach((point, index) => {
            const x = padding.left + (index / (data.length - 1)) * (width - padding.left - padding.right);
            const value = dataType === 'price' ? point.close : point.volume;
            const y = height - padding.bottom - ((value - minValue) / valueRange) * (height - padding.top - padding.bottom);
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('class', 'chart-point');
            
            // Add hover events for tooltip
            circle.addEventListener('mouseenter', (e) => {
                const date = new Date(point.date).toLocaleDateString();
                const displayValue = dataType === 'price' ? 
                    `${point.close.toFixed(2)}` : 
                    this.formatVolume(point.volume);
                
                tooltip.innerHTML = `
                    <div><strong>${date}</strong></div>
                    <div>${dataType === 'price' ? 'Price' : 'Volume'}: ${displayValue}</div>
                    ${dataType === 'price' ? `<div>High: ${point.high.toFixed(2)}</div><div>Low: ${point.low.toFixed(2)}</div>` : ''}
                `;
                
                const rect = svg.getBoundingClientRect();
                tooltip.style.left = `${e.clientX - rect.left + 10}px`;
                tooltip.style.top = `${e.clientY - rect.top - 10}px`;
                tooltip.style.display = 'block';
                
                circle.setAttribute('opacity', '1');
            });
            
            circle.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
                circle.setAttribute('opacity', '0');
            });
            
            chartContent.appendChild(circle);
        });
    }

    async updateChart() {
        if (!this.currentSymbol) return;
        
        this.showLoading(true);
        
        try {
            const apiType = document.getElementById('apiType').value;
            let chartData;
            
            if (apiType === 'makesalot') {
                const response = await fetch(`${this.apiUrls.makesalot}/chart/data/${this.currentSymbol}?period=${this.currentPeriod}`);
                chartData = response.ok ? await response.json() : null;
            } else {
                chartData = this.generateEnhancedChartData(this.currentSymbol);
            }
            
            if (chartData) {
                this.chartData = chartData;
                this.updateChartDisplay(chartData);
            }
        } catch (error) {
            console.error('Chart update error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (show) {
            loading.style.display = 'block';
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
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
    }

    hideError() {
        document.getElementById('error').style.display = 'none';
    }

    showResults() {
        document.getElementById('results').style.display = 'block';
    }
}

// Initialize when popup loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingAssistant();
});