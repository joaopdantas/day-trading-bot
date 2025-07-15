document.addEventListener("DOMContentLoaded", () => {
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

    // Chart display logic
    const chartData = histData.data || histData;
    if (chartData && chartData.length > 0) {
      const svg = document.getElementById("chartSvg");
      const content = document.getElementById("chartContent");
      content.innerHTML = "";
      
      const prices = chartData.map(d => d.close || d.Close);
      const min = Math.min(...prices);
      const max = Math.max(...prices);
      const w = svg.clientWidth;
      const h = svg.clientHeight;
      const pad = 20;
      
      const x = (i) => pad + i * (w - 2 * pad) / (prices.length - 1);
      const y = (p) => h - pad - ((p - min) * (h - 2 * pad)) / (max - min);
      
      let path = prices.map((p, i) => `${i ? "L" : "M"}${x(i)} ${y(p)}`).join(" ");
      let el = document.createElementNS("http://www.w3.org/2000/svg", "path");
      el.setAttribute("d", path); 
      el.setAttribute("stroke", "#22c55e");
      el.setAttribute("fill", "none"); 
      el.setAttribute("stroke-width", 2);
      content.appendChild(el);
      
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