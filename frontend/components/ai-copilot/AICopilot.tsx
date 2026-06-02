'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export default function AICopilot() {
  const [isOpen, setIsOpen] = useState(false)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your AI Copilot. I can help you understand signals, explain trades, analyze risk, and suggest research ideas. How can I assist you today?',
      timestamp: new Date(),
    },
  ])
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    // Simulate AI response (in production, this would call the backend API)
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: generateMockResponse(input),
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, aiResponse])
      setIsTyping(false)
    }, 1000)
  }

  const generateMockResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase()
    
    if (lowerQuery.includes('signal') || lowerQuery.includes('alpha')) {
      return 'Based on the current market data, the ORB signal is showing strong bullish momentum with a confidence score of 0.85. The signal is driven by positive institutional flow in the banking sector and favorable regime conditions (Bullish Trend regime with 92% probability).'
    } else if (lowerQuery.includes('risk') || lowerQuery.includes('drawdown')) {
      return 'Current portfolio risk metrics are within acceptable limits. VaR (95%) is at 1.2%, below the 2% threshold. The recent drawdown of 0.8% was primarily caused by sector rotation in IT stocks, which is expected given the current regime transition probability.'
    } else if (lowerQuery.includes('regime')) {
      return 'The current regime is Bullish Trend with 92% probability. The transition matrix shows a 5% probability of moving to Bearish Trend and 3% to High Volatility regime in the next 10 trading days. I recommend monitoring the regime change indicators closely.'
    } else if (lowerQuery.includes('research') || lowerQuery.includes('idea') || lowerQuery.includes('suggest')) {
      return 'Based on recent market patterns and your current portfolio, I suggest exploring the following research directions:\n\n1. Cross-sectional momentum with regime-aware weighting\n2. Options flow-based alpha for sector rotation\n3. Volatility surface arbitrage strategies\n\nWould you like me to elaborate on any of these?'
    } else {
      return 'I can help you with:\n- Explaining signals and trades\n- Analyzing risk and portfolio changes\n- Understanding regime transitions\n- Suggesting research ideas\n- Detecting anomalies\n\nWhat would you like to know more about?'
    }
  }

  return (
    <div className="fixed bottom-12 right-6 z-50">
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="w-14 h-14 bg-primary rounded-full flex items-center justify-center shadow-lg hover:bg-primary/90 transition-colors"
        >
          <span className="text-2xl">🤖</span>
        </button>
      )}

      {isOpen && (
        <div className="w-96 h-[600px] bg-surface border border-surface-light rounded-lg shadow-2xl flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-surface-light flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-xl">🤖</span>
              <h3 className="font-semibold text-text-primary">AI Copilot</h3>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-text-secondary hover:text-text-primary transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-white'
                      : 'bg-surface-light text-text-primary'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p className="text-xs mt-1 opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-surface-light p-3 rounded-lg">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-text-secondary rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-text-secondary rounded-full animate-bounce delay-100" />
                    <div className="w-2 h-2 bg-text-secondary rounded-full animate-bounce delay-200" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-surface-light">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about signals, risk, research..."
                className="flex-1 bg-surface-light border border-surface-light rounded-lg px-4 py-2 text-sm text-text-primary placeholder-text-secondary focus:outline-none focus:border-primary"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="bg-primary text-white px-4 py-2 rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
