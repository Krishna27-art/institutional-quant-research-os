import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface RegimeData {
  current_regime: { name: string; state: number; confidence: number };
  duration: number;
  transition_prob: number;
  regime_sharpe: number;
}

export async function getRegimeData(): Promise<RegimeData> {
  return apiClient.get<RegimeData>(API_ENDPOINTS.REGIME);
}
