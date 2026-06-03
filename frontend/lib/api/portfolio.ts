import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface PortfolioData {
  aum: number;
  daily_pnl: number;
  mtd_pnl: number;
  net_exposure: number;
}

export async function getPortfolioData(): Promise<PortfolioData> {
  return apiClient.get<PortfolioData>(API_ENDPOINTS.PORTFOLIO);
}
