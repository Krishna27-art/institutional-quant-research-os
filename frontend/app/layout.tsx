import type { Metadata } from 'next'
import './globals.css'
import { QueryClientProviderWrapper } from '../lib/api/query-client'

export const metadata: Metadata = {
  title: 'Institutional Quant Research OS',
  description: 'Next-generation quantitative research operating system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <QueryClientProviderWrapper>
          {children}
        </QueryClientProviderWrapper>
      </body>
    </html>
  )
}
