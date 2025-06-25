import React, { useState } from "react";
import { getMovingAverage } from "./services/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

const Popup = () => {
  const [result, setResult] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysis = async () => {
    setLoading(true);
    try {
      const mockData = [100, 101, 102, 103, 104, 105, 106];
      const res = await getMovingAverage(mockData, "AAPL", 3, "sma");
      setResult(res.values); // assumir que backend devolve { values: [...] }
    } catch (error) {
      console.error("Erro ao obter média móvel:", error);
    } finally {
      setLoading(false);
    }
  };

  // Preparar dados para o gráfico
  const chartData = result
    ? result.map((value, index) => ({ name: index + 1, value }))
    : [];

  return (
    <div className="p-4 w-80">
      <h2 className="text-lg font-bold mb-3">Análise Técnica - SMA</h2>
      <button
        onClick={handleAnalysis}
        className="mb-3 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
      >
        Calcular SMA
      </button>

      {loading && <p>A processar...</p>}

      {result && (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
            <XAxis dataKey="name" />
            <YAxis domain={['auto', 'auto']} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default Popup;
