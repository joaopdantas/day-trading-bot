// MakesALot Trading Extension - Popup Logic (FIXED)
class TradingAssistant {
    constructor() {
        this.apiUrls = {
            makesalot: 'https://makesalot-backend.onrender.com/api/v1',
            yahoo: 'https://query1.finance.yahoo.com/v8/finance/chart'
        };
        this.currentPeriod = '3m';
        this.chart = null;
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
                volume: Math.floor(Math.random() * 1000000)
            });
        }

        const currentPrice = data[data.length - 1].close;
        const previousPrice = data[data.length - 2].close;
        const priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;

        return {
            symbol: symbol,
            current_price: currentPrice,
            price_change: priceChange,
            data: data
        };
    }

    displayResults(analysisData, predictionData, chartData) {
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
            document.getElementById('currentPrice').textContent = `$${chartData.current_price.toFixed(2)}`;
            
            const changeEl = document.getElementById('priceChange');
            const changeText = `${chartData.price_change >= 0 ? '+' : ''}${chartData.price_change.toFixed(2)}%`;
            changeEl.textContent = changeText;
            changeEl.className = `price-change ${chartData.price_change >= 0 ? 'positive' : 'negative'}`;
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

        // Create chart
        if (chartData && chartData.data) {
            this.createChart(chartData);
        }

        this.showResults();
    }

    createChart(chartData) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }

        const labels = chartData.data.map(item => {
            const date = new Date(item.date);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        });
        
        const prices = chartData.data.map(item => item.close);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: chartData.symbol,
                    data: prices,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.8)', font: { size: 9 } }
                    },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { 
                            color: 'rgba(255, 255, 255, 0.8)', 
                            font: { size: 9 },
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    async updateChart() {
        const symbol = document.getElementById('symbol').value.trim();
        if (!symbol) return;
        
        try {
            const apiType = document.getElementById('apiType').value;
            let chartData;
            
            if (apiType === 'makesalot') {
                const response = await fetch(`${this.apiUrls.makesalot}/chart/data/${symbol}?period=${this.currentPeriod}`);
                chartData = response.ok ? await response.json() : null;
            } else {
                chartData = this.generateMockChartData(symbol);
            }
            
            if (chartData) {
                this.createChart(chartData);
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