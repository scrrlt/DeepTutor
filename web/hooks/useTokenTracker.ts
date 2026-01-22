// web/hooks/useTokenTracker.ts
import { useCallback, useState } from 'react'
import { calculateCost, formatCost, formatTokens } from '@/lib/token-utils'

interface TokenTrackerState {
  totalInputTokens: number
  totalOutputTokens: number
  callCount: number
  model: string
  estimatedCost: number
}

export function useTokenTracker(initialModel: string = 'gpt-3.5-turbo') {
  const [state, setState] = useState<TokenTrackerState>({
    totalInputTokens: 0,
    totalOutputTokens: 0,
    callCount: 0,
    model: initialModel,
    estimatedCost: 0,
  })

  const trackUsage = useCallback(
    (inputTokens: number, outputTokens: number, model?: string) => {
      // Use the model provided for THIS call, or fallback to state model
      const currentModel = model || state.model
      const txCost = calculateCost(currentModel, inputTokens, outputTokens)

      setState(prev => ({
        ...prev,
        totalInputTokens: prev.totalInputTokens + inputTokens,
        totalOutputTokens: prev.totalOutputTokens + outputTokens,
        callCount: prev.callCount + 1,
        estimatedCost: prev.estimatedCost + txCost,
        // Update model if provided
        model: model || prev.model,
      }))
    },
    [state.model]
  )

  const reset = useCallback(() => {
    setState({
      totalInputTokens: 0,
      totalOutputTokens: 0,
      callCount: 0,
      model: initialModel,
      estimatedCost: 0,
    })
  }, [initialModel])

  // Derived values
  const totalTokens = state.totalInputTokens + state.totalOutputTokens

  return {
    inputTokens: state.totalInputTokens,
    outputTokens: state.totalOutputTokens,
    totalTokens,
    calls: state.callCount,
    cost: state.estimatedCost,
    model: state.model,
    formattedCost: formatCost(state.estimatedCost),
    formattedTokens: formatTokens(totalTokens),
    trackUsage,
    reset,
  }
}
