import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface AlphaData {
  live_alphas: number;
  avg_sharpe: number;
  best_sharpe: number;
  alpha_correlation: number;
}

export async function getAlphaData(): Promise<AlphaData> {
  return apiClient.get<AlphaData>(API_ENDPOINTS.ALPHA);
}
