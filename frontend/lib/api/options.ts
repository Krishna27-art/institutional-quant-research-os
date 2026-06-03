import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface OptionsData {
  atm_iv: number;
  iv_rank: number;
  pcr_oi: number;
  max_pain: number;
}

export async function getOptionsData(): Promise<OptionsData> {
  return apiClient.get<OptionsData>(API_ENDPOINTS.OPTIONS);
}
