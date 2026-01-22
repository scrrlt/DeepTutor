// web/lib/token-utils.ts

/**
 * Basic pricing table (USD per 1k tokens)
 * Update these values based on your provider's current pricing.
 */
const PRICING = {
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'gpt-4o': { input: 0.005, output: 0.015 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  // Fallback for unknown models
  default: { input: 0.001, output: 0.002 },
}

/**
 * Estimates tokens using the standard "4 chars = 1 token" heuristic.
 * This is fast and runs client-side without heavy tokenizer libraries.
 */
export function estimateTokens(text: string | null | undefined): number {
  if (!text) return 0
  return Math.ceil(text.length / 4)
}

/**
 * Calculates the estimated cost of a transaction.
 */
export function calculateCost(model: string, inputTokens: number, outputTokens: number): number {
  const price = PRICING[model as keyof typeof PRICING] || PRICING['default']

  return (inputTokens / 1000) * price.input + (outputTokens / 1000) * price.output
}

/**
 * Formats a cost number to a readable currency string (e.g., "$0.0045").
 */
export function formatCost(cost: number): string {
  if (cost === 0) return '$0.00'
  return `$${cost.toFixed(6)}`
}

/**
 * Formats token counts with commas (e.g., "1,234").
 */
export function formatTokens(count: number): string {
  return new Intl.NumberFormat('en-US').format(count)
}
