import React, { useState } from "react";
import {
  getMovingAverage,
  fetchStockData,
  fetchRecommendation,
} from "./services/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const apiOptions = [
  { label: "Yahoo Finance", value: "YAHOO_FINANCE_API_KEY" },
  { label: "Alpha Vantage", value: "ALPHA_VANTAGE_API_KEY" },
  // Add more APIs as needed
];

const dataTypeOptions = [
  { label: "Stock Price", value: "stock_price" },
  { label: "Volume", value: "volume" },
  // Add more data types as needed
];

const timeRanges = [
  { label: "3 Months", value: 3 },
  { label: "6 Months", value: 6 },
  { label: "12 Months", value: 12 },
];

const Popup = () => {
  const [api, setApi] = useState(apiOptions[0].value);
  const [symbol, setSymbol] = useState("");
  const [dataType, setDataType] = useState(dataTypeOptions[0].value);
  const [timeRange, setTimeRange] = useState(timeRanges[0].value);
  const [result, setResult] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalysis = async () => {
    setError(null);
    setRecommendation(null);
    if (!symbol.trim()) {
      setError("Por favor, insira um símbolo válido (ex: MSFT).");
      return;
    }
    setLoading(true);
    try {
      // Fetch stock data based on user selection
      const stockData = await fetchStockData({
        api,
        symbol,
        dataType,
        timeRange,
      });
      // Calculate moving average (example: period 3, type 'sma')
      const res = await getMovingAverage(stockData, symbol, 3, "sma");
      setResult(res.values); // assume backend returns { values: [...] }
      // Fetch recommendation
      const rec = await fetchRecommendation(symbol);
      setRecommendation(rec);
    } catch (error) {
      setError("Erro ao buscar dados. Tente novamente mais tarde.");
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Prepare data for chart
  const chartData = result
    ? result.map((value, index) => ({ name: index + 1, value }))
    : [];

  return (
    <div className="p-4 w-80">
      <h2 className="text-lg font-bold mb-3">Análise Técnica - SMA</h2>
      {error && (
        <div className="mb-2 p-2 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
      <div className="mb-2">
        <label className="block text-sm">Fonte da API</label>
        <select
          value={api}
          onChange={(e) => setApi(e.target.value)}
          className="w-full border rounded p-1"
        >
          {apiOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      <div className="mb-2">
        <label className="block text-sm">Símbolo</label>
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="ex: MSFT"
          className="w-full border rounded p-1"
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm">Tipo de Dado</label>
        <select
          value={dataType}
          onChange={(e) => setDataType(e.target.value)}
          className="w-full border rounded p-1"
        >
          {dataTypeOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      <div className="mb-2">
        <label className="block text-sm">Intervalo de Tempo</label>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(Number(e.target.value))}
          className="w-full border rounded p-1"
        >
          {timeRanges.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      <button
        onClick={handleAnalysis}
        className="mb-3 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded w-full disabled:opacity-50"
        disabled={loading}
      >
        {loading ? "A processar..." : "Analisar"}
      </button>
      {loading && (
        <div className="flex items-center justify-center mb-2">
          <svg
            className="animate-spin h-5 w-5 text-blue-500 mr-2"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8z"
            />
          </svg>
          <span>Buscando dados...</span>
        </div>
      )}
      {result && (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
            <XAxis dataKey="name" />
            <YAxis domain={["auto", "auto"]} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      )}
      {recommendation && (
        <div className="mt-3 p-2 border rounded bg-gray-100">
          <span className="font-semibold">Recomendação:</span> {recommendation}
        </div>
      )}
      {result && (
        <div className="mt-2 text-xs text-gray-500">
          Dica: Para testar a integração da API, altere o símbolo e clique em
          "Analisar". Veja o resultado no gráfico e na recomendação.
        </div>
      )}
    </div>
  );
};

export default Popup;
