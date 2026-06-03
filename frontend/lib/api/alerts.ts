import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface AlertsData {
  alerts_today: number;
  critical: number;
  warning: number;
  false_positive: number;
}

export async function getAlertsData(): Promise<AlertsData> {
  return apiClient.get<AlertsData>(API_ENDPOINTS.ALERTS);
}
