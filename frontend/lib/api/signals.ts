import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface SignalsData {
  active_signals: number;
  hit_rate: number;
  avg_r: number;
  signal_strength: string;
}

export async function getSignalsData(): Promise<SignalsData> {
  return apiClient.get<SignalsData>(API_ENDPOINTS.SIGNALS);
}
