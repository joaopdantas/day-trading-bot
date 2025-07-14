document.addEventListener("DOMContentLoaded", () => {
  let selectedMonths = 3;
  chrome.runtime.sendMessage({ type: "GET_SYMBOL_FROM_TAB" }, (response) => {
    const symbolInput = document.getElementById("symbol");
    const apiSelect = document.getElementById("apiType");
    if (response?.symbol && symbolInput) symbolInput.value = response.symbol;
    if (response?.api && apiSelect) apiSelect.value = response.api;
  });

  document.querySelectorAll(".time-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".time-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedMonths = parseInt(btn.dataset.period.replace("m", ""));
    });
  });
  document.querySelector(".time-btn[data-period='3m']").click();

  const analyzeBtn = document.getElementById("analyzeBtn");
  analyzeBtn.addEventListener("click", async () => {
    const symbol = document.getElementById("symbol").value.toUpperCase();
    const api = document.getElementById("apiType").value;
    const strategy = document.getElementById("strategy").value;
    const interval = document.getElementById("interval").value;
    const endDate = new Date();
    const startDate = new Date(new Date().setMonth(endDate.getMonth() - selectedMonths));
    const formatDate = (d) => d.toISOString().split("T")[0];
    const baseURL = "https://makesalot-backend.onrender.com";

    document.getElementById("loading").style.display = "flex";
    document.getElementById("results").style.display = "none";
    document.getElementById("error").textContent = "";

    try {
      const [priceRes, histRes, signalRes] = await Promise.all([
        fetch(`${baseURL}/price/latest?symbol=${symbol}&api=${api}`),
        fetch(`${baseURL}/price/historical?symbol=${symbol}&interval=${interval}&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}&api=${api}`),
        fetch(`${baseURL}/strategy/signal`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol, api, strategy, interval,
            start_date: formatDate(startDate),
            end_date: formatDate(endDate)
          })
        })
      ]);

      const priceData = await priceRes.json();
      const histData = await histRes.json();
      const signalData = await signalRes.json();

      document.getElementById("loading").style.display = "none";
      if (signalData.error) {
        document.getElementById("error").textContent = signalData.error;
        return;
      }

      document.getElementById("results").style.display = "block";
      document.getElementById("recommendationText").textContent = signalData.signal || "HOLD";
      document.getElementById("confidenceText").textContent = `Confidence: ${signalData.confidence || 50}%`;
      document.getElementById("analysisSummary").textContent = `Analysis using ${strategy} strategy.`;
      document.getElementById("currentPrice").textContent = `$${(priceData.price || 0).toFixed(2)}`;
      document.getElementById("priceDate").textContent = priceData.date || "Current";

      const svg = document.getElementById("chartSvg");
      const content = document.getElementById("chartContent");
      content.innerHTML = "";
      const prices = histData.map(d => d.close);
      const min = Math.min(...prices), max = Math.max(...prices);
      const w = svg.clientWidth, h = svg.clientHeight, pad = 20;
      const x = (i) => pad + i * (w - 2 * pad) / (prices.length - 1);
      const y = (p) => h - pad - ((p - min) * (h - 2 * pad)) / (max - min);
      let path = prices.map((p, i) => `${i ? "L" : "M"}${x(i)} ${y(p)}`).join(" ");
      let el = document.createElementNS("http://www.w3.org/2000/svg", "path");
      el.setAttribute("d", path); el.setAttribute("stroke", "#22c55e");
      el.setAttribute("fill", "none"); el.setAttribute("stroke-width", 2);
      content.appendChild(el);
      document.getElementById("highPrice").textContent = `$${max.toFixed(2)}`;
      document.getElementById("lowPrice").textContent = `$${min.toFixed(2)}`;
    } catch (e) {
      document.getElementById("loading").style.display = "none";
      document.getElementById("error").textContent = "Erro: " + e.message;
    }
  });
});