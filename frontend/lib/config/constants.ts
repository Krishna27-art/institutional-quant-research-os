export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws';

export const API_ENDPOINTS = {
  MARKET: '/api/market',
  REGIME: '/api/regime',
  ALPHA: '/api/alpha',
  RISK: '/api/risk',
  PORTFOLIO: '/api/portfolio',
  OPTIONS: '/api/options',
  SIGNALS: '/api/signals',
  ALERTS: '/api/alerts',
  COPILOT_CHAT: '/api/copilot/chat',
} as const;

export const WS_RECONNECT_INTERVAL = 5000;
export const WS_MAX_RECONNECT_ATTEMPTS = 10;
