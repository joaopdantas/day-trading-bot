export type DataSource = "yahoo_finance" | "alpha_vantage";

export interface MarketData {
  symbol: string;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume: number;
  timestamp: number;
  source?: DataSource;
}

export interface DataSourceConfig {
  source: DataSource;
  symbol: string;
  apiKey?: string;
}

export interface SymbolSuggestion {
  symbol: string;
  name: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

export interface CachedData<T> {
  data: T;
  timestamp: number;
}
