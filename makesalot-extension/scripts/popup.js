document.addEventListener("DOMContentLoaded", () => {
  const apiMapByHost = {
    "finance.yahoo.com": "yahoo_finance",
    "www.tradingview.com": "polygon",
    "www.investing.com": "polygon"
  };

  const extractSymbol = () => {
    const url = window.location.href;
    if (url.includes("finance.yahoo.com")) {
      const match = url.match(/quote\/([A-Z.]+)/);
      return match ? match[1] : "";
    }
    if (url.includes("tradingview.com/symbols/")) {
      const match = url.match(/symbols\/([^\/]+)/);
      return match ? match[1].toUpperCase() : "";
    }
    if (url.includes("investing.com")) {
      const match = url.match(/equities\/([^\/]+)/);
      return match ? match[1].replace("-", "_").toUpperCase() : "";
    }
    return "";
  };

  const getStartDateFromPeriod = (endDate, monthsBack) => {
    const end = new Date(endDate);
    const start = new Date(end.setMonth(end.getMonth() - monthsBack));
    return start.toISOString().split("T")[0];
  };

  const updateStatus = (text) => {
    const status = document.getElementById("apiStatus");
    if (status) status.textContent = text;
  };

  const drawChart = (data) => {
    const svg = document.getElementById("chartSvg");
    const content = document.getElementById("chartContent");
    if (!svg || !content || !data.length) return;

    content.innerHTML = "";

    const prices = data.map(d => d.close);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const width = svg.clientWidth || 300;
    const height = svg.clientHeight || 150;
    const padding = 20;

    const xScale = (i) => padding + i * (width - 2 * padding) / (prices.length - 1);
    const yScale = (p) => height - padding - ((p - minPrice) * (height - 2 * padding)) / (maxPrice - minPrice);

    let path = "";
    prices.forEach((price, i) => {
      const x = xScale(i);
      const y = yScale(price);
      path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });

    const pricePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pricePath.setAttribute("d", path);
    pricePath.setAttribute("stroke", "#22c55e");
    pricePath.setAttribute("fill", "none");
    pricePath.setAttribute("stroke-width", 2);
    content.appendChild(pricePath);

    document.getElementById("highPrice").textContent = `$${maxPrice.toFixed(2)}`;
    document.getElementById("lowPrice").textContent = `$${minPrice.toFixed(2)}`;
  };

  // Período (1M, 3M, ...)
  document.querySelectorAll(".time-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".time-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const months = parseInt(btn.dataset.period.replace("m", ""));
      const end = new Date().toISOString().split("T")[0];
      const start = getStartDateFromPeriod(end, months);
      const startDateInput = document.getElementById("startDate");
      const endDateInput = document.getElementById("endDate");
      if (startDateInput) startDateInput.value = start;
      if (endDateInput) endDateInput.value = end;
    });
  });
  document.querySelector(".time-btn[data-period='3m']").click();

  // Preencher símbolo automaticamente
  const detectedSymbol = extractSymbol();
  const symbolInput = document.getElementById("symbol");
  if (symbolInput && detectedSymbol) symbolInput.value = detectedSymbol;

  const currentHost = window.location.hostname;
  const defaultApi = apiMapByHost[currentHost] || "alpha_vantage";

  const analyzeBtn = document.getElementById("analyzeBtn");
  if (!analyzeBtn) return;

  analyzeBtn.addEventListener("click", async () => {
    const symbol = symbolInput ? symbolInput.value.toUpperCase() : "";
    let api = document.getElementById("apiType")?.value || "auto";
    const strategy = document.getElementById("strategy")?.value || "technical";
    const interval = document.getElementById("interval")?.value || "1d";
    const startDate = document.getElementById("startDate")?.value || "";
    const endDate = document.getElementById("endDate")?.value || new Date().toISOString().split("T")[0];

    if (api === "auto") api = defaultApi;

    document.getElementById("loading").style.display = "flex";
    document.getElementById("results").style.display = "none";
    document.getElementById("error").textContent = "";
    updateStatus("Analyzing...");

    try {
      const priceRes = await fetch(`https://makesalot-backend.onrender.com/price/latest?symbol=${symbol}&api=${api}`);
      const priceData = await priceRes.json();

      const histRes = await fetch(`https://makesalot-backend.onrender.com/price/historical?symbol=${symbol}&interval=${interval}&start_date=${startDate}&end_date=${endDate}&api=${api}`);
      const histData = await histRes.json();

      const signalRes = await fetch(`https://makesalot-backend.onrender.com/strategy/signal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, api, strategy, interval, start_date: startDate, end_date: endDate })
      });
      const signalData = await signalRes.json();

      document.getElementById("loading").style.display = "none";

      if (signalData.error) {
        document.getElementById("error").textContent = signalData.error;
        updateStatus("Error from API");
        return;
      }

      document.getElementById("results").style.display = "block";
      document.getElementById("recommendationText").textContent = signalData.signal || "HOLD";
      document.getElementById("confidenceText").textContent = `Confidence: ${signalData.confidence || 50}%`;
      document.getElementById("analysisSummary").textContent = `Analysis using ${strategy} strategy.`;
      document.getElementById("currentPrice").textContent = `$${(priceData.price || 0).toFixed(2)}`;
      document.getElementById("priceDate").textContent = priceData.date || "Current";
      drawChart(histData);
      updateStatus("Connected ✅");
    } catch (err) {
      document.getElementById("loading").style.display = "none";
      document.getElementById("error").textContent = "Erro: " + err.message;
      updateStatus("API failed ❌");
    }
  });
});
