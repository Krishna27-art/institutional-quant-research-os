'use client'

import { useEffect, useRef } from 'react'

interface TradingViewChartProps {
  symbol?: string
  theme?: 'light' | 'dark'
  autosize?: boolean
}

export default function TradingViewChart({
  symbol = 'NSE:NIFTY',
  theme = 'dark',
  autosize = true,
}: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return

    const script = document.createElement('script')
    script.src = 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js'
    script.async = true
    script.onload = () => {
      if (!containerRef.current) return

      const { createChart, ColorType } = (window as any).LightweightCharts

      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height: 400,
        layout: {
          background: { type: ColorType.Solid, color: '#111827' },
          textColor: '#f9fafb',
        },
        grid: {
          vertLines: { color: '#1f2937' },
          horzLines: { color: '#1f2937' },
        },
        crosshair: {
          mode: (window as any).LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
          borderColor: '#1f2937',
        },
        timeScale: {
          borderColor: '#1f2937',
          timeVisible: true,
          secondsVisible: false,
        },
      })

      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
      })

      // Generate sample data
      const now = new Date()
      const data = []
      let price = 22450

      for (let i = 100; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60000 * 60 * 24) // Daily data
        const volatility = 50
        const change = (Math.random() - 0.5) * volatility
        price += change

        const open = price
        const close = price + (Math.random() - 0.5) * volatility
        const high = Math.max(open, close) + Math.random() * 10
        const low = Math.min(open, close) - Math.random() * 10

        data.push({
          time: time.getTime() / 1000,
          open,
          high,
          low,
          close,
        })
      }

      candlestickSeries.setData(data)

      if (autosize) {
        const handleResize = () => {
          if (containerRef.current) {
            chart.applyOptions({
              width: containerRef.current.clientWidth,
            })
          }
        }

        window.addEventListener('resize', handleResize)
        return () => {
          window.removeEventListener('resize', handleResize)
          chart.remove()
        }
      }
    }

    document.head.appendChild(script)

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script)
      }
    }
  }, [symbol, theme, autosize])

  return <div ref={containerRef} className="w-full h-full" />
}
