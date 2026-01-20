export interface OptimizationRequest {
  symbol: string;
  initial_capital: number;
  min_trades: number;
  min_profit_factor: number;
  max_drawdown: number;
  timeframe: string;
  window_mode: "auto_shortest" | "fixed_recent" | "multi_recent";
  window_days_candidates?: number[];
  stability_windows?: number;
}

export interface ChartPoint {
  time: number;
  value: number;
}

export interface StrategyMetrics {
  total_pnl: number;
  pnl_percent: number;
  total_txns: number;
  profit_factor: number;
  max_dd: number;
  pine_code: string;
  tag: string;
  duration: string;
  note: string;
  window_days?: number; // New field from backend
}

export interface StrategyResponse {
  strategy_name: string;
  window_days: number;
  metrics: StrategyMetrics;
  chart_data: ChartPoint[];
}