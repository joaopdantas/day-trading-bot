// Chart optimization functions - MOVED TO TOP
function optimizeChartForTimeframe(chartData, symbol) {
    let timeframe = '1d'; // default
    
    // Method 1: Try to detect from URL (works in content script)
    try {
        const url = window.location.href;
        if (url.includes('range=6mo')) timeframe = '6m';
        else if (url.includes('range=1y')) timeframe = '1y';
        else if (url.includes('range=5d')) timeframe = '5d';
        else if (url.includes('range=1mo')) timeframe = '1m';
    } catch (e) {
        // URL detection failed (in popup), use data length to guess
    }
    
    // Method 2: Guess from data length (fallback for popup)
    if (timeframe === '1d' && chartData && chartData.length) {
        if (chartData.length > 200) timeframe = '1y';
        else if (chartData.length > 100) timeframe = '6m';
        else if (chartData.length > 50) timeframe = '1m';
        else if (chartData.length > 20) timeframe = '5d';
    }
    
    const chartConfig = {
        width: 280,
        height: getOptimalHeight(timeframe),
        dataPoints: getOptimalDataPoints(timeframe),
        strokeWidth: getOptimalStrokeWidth(timeframe),
        timeframe: timeframe
    };
    
    return chartConfig;
}

function getOptimalHeight(timeframe) {
    const heights = {
        '1d': 180,
        '5d': 160,
        '1m': 150,
        '6m': 130,  // Reduced height for 6-month view
        '1y': 120   // More compressed for 1-year view
    };
    return heights[timeframe] || 150;
}

function getOptimalDataPoints(timeframe) {
    const maxPoints = {
        '1d': 100,   // Show all detail for 1 day
        '5d': 70,    // Good detail for 5 days
        '1m': 50,    // Moderate detail for 1 month
        '6m': 35,    // Compressed for 6 months
        '1y': 25     // Most compressed for 1 year
    };
    return maxPoints[timeframe] || 50;
}

function getOptimalStrokeWidth(timeframe) {
    const strokeWidths = {
        '1d': 2,
        '5d': 1.8,
        '1m': 1.5,
        '6m': 1.2,   // Thinner lines for compressed view
        '1y': 1      // Thinnest for maximum compression
    };
    return strokeWidths[timeframe] || 1.5;
}

// Sample data points intelligently to reduce chart crowding
function sampleChartData(data, maxPoints) {
    if (!data || data.length <= maxPoints) return data;
    
    const step = Math.floor(data.length / maxPoints);
    const sampledData = [];
    
    // Always include first and last points
    sampledData.push(data[0]);
    
    // Sample intermediate points
    for (let i = step; i < data.length - step; i += step) {
        sampledData.push(data[i]);
    }
    
    // Always include the most recent point
    sampledData.push(data[data.length - 1]);
    
    return sampledData;
}

// Add grid lines for better readability on compressed charts
function addGridLines(content, width, height, padding) {
    // Add horizontal grid lines (3 lines)
    for (let i = 1; i <= 3; i++) {
        const y = padding + (i * (height - 2 * padding) / 4);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", padding);
        line.setAttribute("y1", y);
        line.setAttribute("x2", width - padding);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", "rgba(255, 255, 255, 0.1)");
        line.setAttribute("stroke-width", "0.5");
        content.appendChild(line);
    }
    
    // Add vertical grid lines (2 lines)
    for (let i = 1; i <= 2; i++) {
        const x = padding + (i * (width - 2 * padding) / 3);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x);
        line.setAttribute("y1", padding);
        line.setAttribute("x2", x);
        line.setAttribute("y2", height - padding);
        line.setAttribute("stroke", "rgba(255, 255, 255, 0.1)");
        line.setAttribute("stroke-width", "0.5");
        content.appendChild(line);
    }
}

document.addEventListener("DOMContentLoaded", function() {
  // REMOVED: CSS injection code (using styles.css instead)

  const baseURL = "http://localhost:8000";
  // Variables for card selection
  let selectedStrategy = null;
  let selectedMode = null;
  let selectedMonths = 3;

  // Check API connection on load
  checkAPIConnection();

  // Symbol detection logic (removed API detection since it's automatic now)
  chrome.runtime.sendMessage({ type: "GET_SYMBOL_FROM_TAB" }, (response) => {
    const symbolInput = document.getElementById("symbol");
    if (response?.symbol && symbolInput) {
      symbolInput.value = response.symbol;
    }
  });

  // API Connection Check with smart fallback info
  async function checkAPIConnection() {
    const statusElement = document.getElementById("apiStatus");
    const baseURL = "http://localhost:8000";  // Local testing
    
    statusElement.textContent = "🔄 Checking API connection (Polygon → Alpha Vantage fallback)...";
    statusElement.style.background = "rgba(255, 193, 7, 0.2)";
    statusElement.style.color = "#ffc107";

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds for local

      const response = await fetch(`${baseURL}/health`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (response.ok) {
        const healthData = await response.json();
        statusElement.textContent = `✅ API Connected (using ${healthData.active_api || 'auto-fallback'})`;
        statusElement.style.background = "rgba(76, 175, 80, 0.2)";
        statusElement.style.color = "#4CAF50";
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('API Connection Error:', error);
      
      if (error.name === 'AbortError') {
        statusElement.textContent = "⏰ API Timeout - Make sure your local server is running";
      } else {
        statusElement.textContent = `❌ API Error: ${error.message}`;
      }
      
      statusElement.style.background = "rgba(244, 67, 54, 0.2)";
      statusElement.style.color = "#f44336";
      
      // Add retry button
      const retryBtn = document.createElement("button");
      retryBtn.textContent = "🔄 Retry";
      retryBtn.style.marginLeft = "10px";
      retryBtn.style.padding = "4px 8px";
      retryBtn.style.fontSize = "10px";
      retryBtn.style.cursor = "pointer";
      retryBtn.onclick = checkAPIConnection;
      statusElement.appendChild(retryBtn);
    }
  }

  // Time button logic
  document.querySelectorAll(".time-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".time-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedMonths = parseInt(btn.dataset.period.replace("m", ""));
    });
  });
  document.querySelector(".time-btn[data-period='3m']").click();

  // Strategy card selection
  document.querySelectorAll('.strategy-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.strategy-card').forEach(c => 
        c.classList.remove('selected'));
      card.classList.add('selected');
      selectedStrategy = card.dataset.strategy;
      document.getElementById('modeSelection').style.display = 'block';
      updateAnalyzeButton();
    });
  });

  // Mode card selection  
  document.querySelectorAll('.mode-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.mode-card').forEach(c => 
        c.classList.remove('selected'));
      card.classList.add('selected');
      selectedMode = card.dataset.mode;
      updateAnalyzeButton();
    });
  });

  // Update analyze button based on selections
  function updateAnalyzeButton() {
    const analyzeBtn = document.getElementById("analyzeBtn");
    
    if (selectedStrategy && selectedMode) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = selectedMode === "advisory" ? 
        "Get Trading Recommendation" : "Execute Paper Trade";
    } else if (selectedStrategy && !selectedMode) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Select Trading Mode";
    } else {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Select Strategy & Mode";
    }
  }

  // Analyze button click - SIMPLIFIED (no API selection needed)
  const analyzeBtn = document.getElementById("analyzeBtn");
  analyzeBtn.addEventListener("click", async () => {
    const symbol = document.getElementById("symbol").value.toUpperCase();
    const strategy = selectedStrategy;
    const interval = document.getElementById("interval").value;
    const endDate = new Date();
    const startDate = new Date(new Date().setMonth(endDate.getMonth() - selectedMonths));
    const formatDate = (d) => d.toISOString().split("T")[0];
    const baseURL = "http://localhost:8000";  // Local testing

    // Show confirmation for automated mode
    if (selectedMode === "automated") {
      const confirmed = confirm(`Execute paper trade for ${symbol}?\nThis will use the ${strategy} strategy.\n\nThis is paper trading - no real money involved.`);
      if (!confirmed) return;
    }

    // Show loading state
    document.getElementById("loading").style.display = "flex";
    document.getElementById("results").style.display = "none";
    document.getElementById("error").textContent = "";

    try {
      // Fetch with timeout helper
      const fetchWithTimeout = async (url, options = {}) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds
        
        try {
          const response = await fetch(url, {
            ...options,
            signal: controller.signal
          });
          clearTimeout(timeoutId);
          return response;
        } catch (error) {
          clearTimeout(timeoutId);
          throw error;
        }
      };

      // Make API calls (no api parameter needed - automatic fallback)
      const [priceRes, histRes, signalRes] = await Promise.all([
        fetchWithTimeout(`${baseURL}/price/latest?symbol=${symbol}`),
        fetchWithTimeout(`${baseURL}/price/historical?symbol=${symbol}&interval=${interval}&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}`),
        fetchWithTimeout(`${baseURL}/strategy/signal`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol, 
            strategy,
            interval,
            start_date: formatDate(startDate),
            end_date: formatDate(endDate)
          })
        })
      ]);

      const priceData = await priceRes.json();
      const histData = await histRes.json();
      const signalData = await signalRes.json();

      // Hide loading
      document.getElementById("loading").style.display = "none";
      
      if (signalData.error) {
        document.getElementById("error").textContent = signalData.error;
        return;
      }

      // Display results with API source info
      displayResults(signalData, selectedMode, strategy, priceData, histData);

    } catch (e) {
      document.getElementById("loading").style.display = "none";
      
      if (e.name === 'AbortError') {
        document.getElementById("error").textContent = "Request timed out. Make sure your local API server is running (uvicorn main:app --reload)";
      } else {
        document.getElementById("error").textContent = "Error: " + e.message;
      }
    }
  });

  // Display results with API source information
  function displayResults(signalData, mode, strategy, priceData, histData) {
    document.getElementById("results").style.display = "block";
    
    const modeText = mode === "advisory" ? 
      "📊 Recommendation" : "🤖 Paper Trade Executed";
    
    const strategyNames = {
      'conservative': 'Conservative (Technical Analysis)',
      'aggressive': 'Aggressive (ML Trading)',
      'balanced': 'Balanced (Split-Capital)'
    };

    // Update result display
    document.getElementById("recommendationText").textContent = signalData.signal || signalData.action || "HOLD";
    document.getElementById("confidenceText").textContent = `Confidence: ${Math.round((signalData.confidence || 0.5) * 100)}%`;
    document.getElementById("analysisSummary").textContent = 
      `${modeText} using ${strategyNames[strategy] || strategy} strategy`;
    
    // Handle price data (could be from either API response format)
    const price = priceData.price || priceData.data?.[0]?.close || 0;
    document.getElementById("currentPrice").textContent = `$${price.toFixed(2)}`;
    document.getElementById("priceDate").textContent = priceData.timestamp || "Current";

    // Show which API was used
    const apiUsed = signalData.api_used || priceData.api_used || "auto-detected";
    document.getElementById("apiSource").textContent = `Data source: ${apiUsed} (smart fallback)`;

    // Chart display logic - IMPROVED
    const chartData = histData.data || histData;
    if (chartData && chartData.length > 0) {
      // 🚀 Get optimized chart configuration
      const config = optimizeChartForTimeframe(chartData, symbol);
      const sampledData = sampleChartData(chartData, config.dataPoints);
      
      const svg = document.getElementById("chartSvg");
      const content = document.getElementById("chartContent");
      content.innerHTML = "";
      
      // 🚀 Apply optimized height and add timeframe class
      svg.setAttribute("height", config.height);
      svg.style.height = config.height + "px";
      svg.className = `chart-svg chart-${config.timeframe}`;
      
      const prices = sampledData.map(d => d.close || d.Close);
      const min = Math.min(...prices);
      const max = Math.max(...prices);
      const w = svg.clientWidth;
      const h = config.height; // Use optimized height
      const pad = 20;
      
      const x = (i) => pad + i * (w - 2 * pad) / (prices.length - 1);
      const y = (p) => h - pad - ((p - min) * (h - 2 * pad)) / (max - min);
      
      let path = prices.map((p, i) => `${i ? "L" : "M"}${x(i)} ${y(p)}`).join(" ");
      let el = document.createElementNS("http://www.w3.org/2000/svg", "path");
      el.setAttribute("d", path); 
      el.setAttribute("stroke", "#22c55e");
      el.setAttribute("fill", "none"); 
      // 🚀 Use optimized stroke width
      el.setAttribute("stroke-width", config.strokeWidth);
      content.appendChild(el);
      
      // 🚀 Add grid lines for better readability on compressed charts
      if (config.height <= 130) { // For compressed charts (6m, 1y)
        addGridLines(content, w, h, pad);
      }
      
      document.getElementById("highPrice").textContent = `$${max.toFixed(2)}`;
      document.getElementById("lowPrice").textContent = `$${min.toFixed(2)}`;
    }

    // Add mode-specific result styling
    const recommendationElement = document.getElementById("recommendationText");
    if (mode === "automated") {
      recommendationElement.style.background = "rgba(33, 150, 243, 0.2)";
      recommendationElement.style.borderLeft = "4px solid #2196F3";
      recommendationElement.style.paddingLeft = "10px";
    } else {
      recommendationElement.style.background = "rgba(76, 175, 80, 0.2)";
      recommendationElement.style.borderLeft = "4px solid #4CAF50";
      recommendationElement.style.paddingLeft = "10px";
    }
  }
});