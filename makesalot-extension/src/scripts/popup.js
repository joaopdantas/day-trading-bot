// MakesALot Trading Extension - Popup Logic with CSS Charts
class TradingAssistant {
    constructor() {
        this.apiUrls = {
            makesalot: 'https://makesalot-backend.onrender.com/api/v1',
            yahoo: 'https://query1.finance.yahoo.com/v8/finance/chart'
        };
        this.currentPeriod = '3m';
        this.currentSymbol = 'MSFT';
        this.chartData = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiConnection();
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
                await this.analyzeYahoo(symbol);
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

    async analyzeYahoo(symbol) {
        // Yahoo Finance fallback with mock data
        const mockAnalysis = {
            symbol: symbol,
            indicators: [
                { name: 'RSI', value: 45 + Math.random() * 20, signal: 'HOLD' },
                { name: 'MACD', value: (Math.random() - 0.5) * 2, signal: 'BUY' },
                { name: 'BB', value: Math.random(), signal: 'HOLD' }
            ]
        };

        const mockPrediction = {
            prediction: {
                direction: ['BUY', 'SELL', 'HOLD'][Math.floor(Math.random() * 3)],
                confidence: 0.6 + Math.random() * 0.3
            }
        };

        const mockChart = this.generateMockChartData(symbol);

        this.displayResults(mockAnalysis, mockPrediction, mockChart);
    }

    generateMockChartData(symbol) {
        const days = { '3m': 90, '6m': 180, '1y': 365 }[this.currentPeriod];
        const basePrice = 100 + Math.random() * 200;
        const data = [];

        for (let i = 0; i < days; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (days - i));
            
            const price = basePrice * (1 + (Math.random() - 0.5) * 0.1);
            data.push({
                date: date.toISOString(),
                close: price,
                high: price * (1 + Math.random() * 0.05),
                low: price * (1 - Math.random() * 0.05),
                volume: Math.floor(Math.random() * 1000000)
            });
        }

        const currentPrice = data[data.length - 1].close;
        const previousPrice = data[data.length - 2].close;
        const priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;

        // Calculate high/low for period
        const prices = data.map(d => d.close);
        const highPrice = Math.max(...prices);
        const lowPrice = Math.min(...prices);
        const avgVolume = data.reduce((sum, d) => sum + d.volume, 0) / data.length;

        return {
            symbol: symbol,
            current_price: currentPrice,
            price_change: priceChange,
            high_price: highPrice,
            low_price: lowPrice,
            avg_volume: avgVolume,
            data: data
        };
    }

    displayResults(analysisData, predictionData, chartData) {
        // Store chart data for period switching
        this.chartData = chartData;

        // Display recommendation
        if (predictionData && predictionData.prediction) {
            const recommendation = predictionData.prediction.direction;
            const confidence = Math.round(predictionData.prediction.confidence * 100);
            
            document.getElementById('recommendationText').textContent = recommendation;
            document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
            
            const card = document.getElementById('recommendationCard');
            card.className = `recommendation ${recommendation.toLowerCase()}`;
        }

        // Display price info
        if (chartData) {
            document.getElementById('currentPrice').textContent = `${chartData.current_price.toFixed(2)}`;
            
            const changeEl = document.getElementById('priceChange');
            const changeText = `${chartData.price_change >= 0 ? '+' : ''}${chartData.price_change.toFixed(2)}%`;
            changeEl.textContent = changeText;
            changeEl.className = `price-change ${chartData.price_change >= 0 ? 'positive' : 'negative'}`;

            // Update chart display
            this.updateChartDisplay(chartData);
        }

        // Display indicators
        if (analysisData && analysisData.indicators) {
            analysisData.indicators.forEach(indicator => {
                const valueEl = document.getElementById(`${indicator.name.toLowerCase()}Value`);
                const signalEl = document.getElementById(`${indicator.name.toLowerCase()}Signal`);
                
                if (valueEl) {
                    valueEl.textContent = typeof indicator.value === 'number' ? 
                        indicator.value.toFixed(2) : indicator.value;
                }
                if (signalEl) {
                    signalEl.textContent = indicator.signal;
                    signalEl.style.color = this.getSignalColor(indicator.signal);
                }
            });
        }

        this.showResults();
    }

    updateChartDisplay(chartData) {
        if (!chartData) return;

        // Update chart statistics
        document.getElementById('highPrice').textContent = `${chartData.high_price.toFixed(2)}`;
        document.getElementById('lowPrice').textContent = `${chartData.low_price.toFixed(2)}`;
        
        // Format volume
        const volume = chartData.avg_volume || 0;
        const volumeText = volume > 1000000 ? 
            `${(volume / 1000000).toFixed(1)}M` : 
            `${Math.round(volume / 1000)}K`;
        document.getElementById('avgVolume').textContent = volumeText;

        // Create simple chart visualization
        this.createSimpleChart(chartData);
    }

    createSimpleChart(chartData) {
        const chartLine = document.getElementById('chartLine');
        const chartPoints = document.getElementById('chartPoints');
        
        if (!chartData.data || chartData.data.length === 0) return;

        // Determine chart trend
        const firstPrice = chartData.data[0].close;
        const lastPrice = chartData.data[chartData.data.length - 1].close;
        const overallChange = ((lastPrice - firstPrice) / firstPrice) * 100;

        // Set chart line style based on trend
        chartLine.className = 'chart-line';
        if (overallChange > 5) {
            chartLine.classList.add('bullish');
        } else if (overallChange < -5) {
            chartLine.classList.add('bearish');
        } else {
            chartLine.classList.add('neutral');
        }

        // Clear previous points
        chartPoints.innerHTML = '';

        // Add price points (simplified - show every 10th point)
        const dataStep = Math.max(1, Math.floor(chartData.data.length / 8));
        const prices = chartData.data.map(d => d.close);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;

        for (let i = 0; i < chartData.data.length; i += dataStep) {
            const price = chartData.data[i].close;
            const normalizedPrice = (price - minPrice) / priceRange;
            const x = (i / chartData.data.length) * 100;
            const y = (1 - normalizedPrice) * 100;

            const point = document.createElement('div');
            point.className = 'chart-point';
            point.style.left = `${x}%`;
            point.style.top = `${y}%`;
            chartPoints.appendChild(point);
        }
    }

    async updateChart() {
        if (!this.currentSymbol) return;
        
        try {
            const apiType = document.getElementById('apiType').value;
            let chartData;
            
            if (apiType === 'makesalot') {
                const response = await fetch(`${this.apiUrls.makesalot}/chart/data/${this.currentSymbol}?period=${this.currentPeriod}`);
                chartData = response.ok ? await response.json() : null;
            } else {
                chartData = this.generateMockChartData(this.currentSymbol);
            }
            
            if (chartData) {
                this.chartData = chartData;
                this.updateChartDisplay(chartData);
            }
        } catch (error) {
            console.error('Chart update error:', error);
        }
    }

    getSignalColor(signal) {
        switch (signal.toLowerCase()) {
            case 'buy': return '#22c55e';
            case 'sell': return '#ef4444';
            default: return '#f59e0b';
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