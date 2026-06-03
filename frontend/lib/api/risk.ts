import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface RiskData {
  var: number;
  cvar: number;
  gross_exposure: number;
  max_drawdown: number;
}

export async function getRiskData(): Promise<RiskData> {
  return apiClient.get<RiskData>(API_ENDPOINTS.RISK);
}
