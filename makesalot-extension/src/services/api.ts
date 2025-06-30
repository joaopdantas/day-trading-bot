export async function getMovingAverage(
  data: number[],
  symbol: string,
  period: number,
  type: "sma" | "ema"
) {
  const response = await fetch(
    "https://makesalot-backend.onrender.com/api/analysis/moving-average",
    {
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
    }
  );

  if (!response.ok) throw new Error("Erro ao comunicar com o backend");
  return await response.json();
}

export async function getPrediction(data: number[], model: string) {
  const response = await fetch(
    "https://makesalot-backend.onrender.com/api/prediction",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data, model }),
    }
  );

  if (!response.ok) throw new Error("Erro ao obter previsão");
  return response.json();
}

// Fetch stock data from selected API (mock implementation, extend as needed)
export async function fetchStockData({
  api,
  symbol,
  dataType,
  timeRange,
}: {
  api: string;
  symbol: string;
  dataType: string;
  timeRange: number;
}): Promise<number[]> {
  // TODO: Implement real API calls for each provider
  // For now, return mock data
  // You can add logic for Yahoo, Alpha Vantage, etc. here
  return [100, 102, 105, 103, 108, 110, 115, 120, 125, 130, 128, 135];
}

// Fetch buy/sell recommendation from a website (mock implementation)
export async function fetchRecommendation(symbol: string): Promise<string> {
  // TODO: Implement real fetch from a recommendation website or API
  // For now, return a random recommendation
  const options = ["Buy", "Sell", "Hold"];
  return options[Math.floor(Math.random() * options.length)];
}
