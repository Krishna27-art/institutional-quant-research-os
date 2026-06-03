import { apiClient } from './client';
import { API_ENDPOINTS } from '../config/constants';

export interface CopilotRequest {
  message: string;
}

export interface CopilotResponse {
  reply: string;
}

export async function sendCopilotMessage(request: CopilotRequest): Promise<CopilotResponse> {
  return apiClient.post<CopilotResponse>(API_ENDPOINTS.COPILOT_CHAT, request);
}
