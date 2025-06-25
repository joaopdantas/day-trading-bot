export async function getMovingAverage(
  data: number[],
  symbol: string,
  period: number,
  type: "sma" | "ema"
) {
  const response = await fetch("https://makesalot-backend.onrender.com/api/analysis/moving-average", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      symbol,
      data,
      period,
      type,
    }),
  });

  if (!response.ok) throw new Error("Erro ao comunicar com o backend");
  return await response.json();
}

export async function getPrediction(data: number[], model: string) {
  const response = await fetch("https://makesalot-backend.onrender.com/api/prediction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data, model }),
  });

  if (!response.ok) throw new Error("Erro ao obter previsão");
  return response.json();
}

