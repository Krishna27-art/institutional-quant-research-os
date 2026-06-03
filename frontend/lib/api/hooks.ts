'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getMarketData } from './market';
import { getRegimeData } from './regime';
import { getAlphaData } from './alpha';
import { getRiskData } from './risk';
import { getPortfolioData } from './portfolio';
import { getOptionsData } from './options';
import { getSignalsData } from './signals';
import { getAlertsData } from './alerts';
import { sendCopilotMessage } from './copilot';

// Market Data Hook
export function useMarketData() {
  return useQuery({
    queryKey: ['market'],
    queryFn: getMarketData,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// Regime Data Hook
export function useRegimeData() {
  return useQuery({
    queryKey: ['regime'],
    queryFn: getRegimeData,
    refetchInterval: 60000, // Refetch every minute
  });
}

// Alpha Data Hook
export function useAlphaData() {
  return useQuery({
    queryKey: ['alpha'],
    queryFn: getAlphaData,
    refetchInterval: 60000, // Refetch every minute
  });
}

// Risk Data Hook
export function useRiskData() {
  return useQuery({
    queryKey: ['risk'],
    queryFn: getRiskData,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// Portfolio Data Hook
export function usePortfolioData() {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolioData,
    refetchInterval: 15000, // Refetch every 15 seconds
  });
}

// Options Data Hook
export function useOptionsData() {
  return useQuery({
    queryKey: ['options'],
    queryFn: getOptionsData,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// Signals Data Hook
export function useSignalsData() {
  return useQuery({
    queryKey: ['signals'],
    queryFn: getSignalsData,
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}

// Alerts Data Hook
export function useAlertsData() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: getAlertsData,
    refetchInterval: 15000, // Refetch every 15 seconds
  });
}

// Copilot Mutation Hook
export function useCopilotChat() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: sendCopilotMessage,
    onSuccess: () => {
      // Invalidate any relevant queries if needed
      queryClient.invalidateQueries({ queryKey: ['copilot'] });
    },
  });
}
