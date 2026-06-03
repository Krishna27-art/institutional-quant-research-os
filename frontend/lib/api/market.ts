import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface MarketData {
  nifty: { value: number; change: number; change_pct: number };
  banknifty: { value: number; change: number; change_pct: number };
  vix: { value: number; change: number; change_pct: number };
  usdinr: { value: number; change: number; change_pct: number };
}

export async function getMarketData(): Promise<MarketData> {
  return apiClient.get<MarketData>(API_ENDPOINTS.MARKET);
}
