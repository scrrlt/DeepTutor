'use client'

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'
import { wsUrl } from '@/lib/api'
import { calculateCost, estimateTokens } from '@/lib/token-utils'
import {
  SolverState,
  INITIAL_SOLVER_STATE,
  DEFAULT_SOLVER_AGENT_STATUS,
  DEFAULT_TOKEN_STATS,
} from '@/types/solver'
import { LogEntry, TokenStats } from '@/types/common'

// Context type
interface SolverContextType {
  solverState: SolverState
  setSolverState: React.Dispatch<React.SetStateAction<SolverState>>
  startSolver: (question: string, kb: string) => void
  stopSolver: () => void
  tokenStats: TokenStats
}

const SolverContext = createContext<SolverContextType | undefined>(undefined)

const DEFAULT_TOKEN_MODEL = 'gpt-3.5-turbo'
const LOG_FLUSH_INTERVAL_MS = 100 // Throttle UI updates to 10fps
const MAX_LOG_ENTRIES = 500 // Cap log history to avoid unbounded growth

export function SolverProvider({ children }: { children: React.ReactNode }) {
  const [solverState, setSolverState] = useState<SolverState>(INITIAL_SOLVER_STATE)
  const [tokenStats, setTokenStats] = useState<TokenStats>(DEFAULT_TOKEN_STATS)

  const solverWs = useRef<WebSocket | null>(null)
  const tokenStatsFromServer = useRef(false)
  const lastTokenModel = useRef(DEFAULT_TOKEN_MODEL)

  // Buffers for throttling high-frequency updates
  const logBuffer = useRef<LogEntry[]>([])
  const isFlushing = useRef(false)

  // Throttled Log Flusher
  useEffect(() => {
    const interval = setInterval(() => {
      if (logBuffer.current.length > 0) {
        const logsToFlush = [...logBuffer.current]
        logBuffer.current = [] // Clear buffer immediately

        setSolverState(prev => {
          const combined = [...prev.logs, ...logsToFlush]
          const trimmed =
            combined.length > MAX_LOG_ENTRIES
              ? combined.slice(combined.length - MAX_LOG_ENTRIES)
              : combined
          return {
            ...prev,
            logs: trimmed,
          }
        })
      }
    }, LOG_FLUSH_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [])

  const addSolverLog = useCallback((log: LogEntry) => {
    logBuffer.current.push(log)
  }, [])

  const startSolver = useCallback(
    (question: string, kb: string) => {
      if (solverWs.current) {
        solverWs.current.close()
      }

      tokenStatsFromServer.current = false
      lastTokenModel.current = DEFAULT_TOKEN_MODEL
      setTokenStats(DEFAULT_TOKEN_STATS)
      logBuffer.current = [] // Clear log buffer

      setSolverState(prev => ({
        ...prev,
        isSolving: true,
        logs: [],
        messages: [...prev.messages, { role: 'user' as const, content: question }],
        question,
        selectedKb: kb,
        agentStatus: { ...DEFAULT_SOLVER_AGENT_STATUS },
        progress: {
          stage: null,
          progress: {},
        },
      }))

      const ws = new WebSocket(wsUrl('/api/v1/solve'))
      solverWs.current = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({ question, kb_name: kb }))
        addSolverLog({ type: 'system', content: 'Initializing connection...' })
      }

      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'log') {
            addSolverLog(data)
          } else if (data.type === 'agent_status') {
            setSolverState(prev => ({
              ...prev,
              agentStatus: data.all_agents || {
                ...prev.agentStatus,
                [data.agent]: data.status,
              },
            }))
          } else if (data.type === 'token_stats') {
            tokenStatsFromServer.current = true
            lastTokenModel.current = data.stats?.model || DEFAULT_TOKEN_MODEL
            setTokenStats(prev => data.stats || prev)
          } else if (data.type === 'progress') {
            setSolverState(prev => ({
              ...prev,
              progress: {
                stage: data.stage,
                progress: data.progress || {},
              },
            }))
          } else if (data.type === 'result') {
            if (!tokenStatsFromServer.current) {
              const inputTokenEstimate = estimateTokens(question)
              const outputTokenEstimate = estimateTokens(data.final_answer || '')
              const model = data.model || lastTokenModel.current || DEFAULT_TOKEN_MODEL
              lastTokenModel.current = model

              setTokenStats(prev => {
                const calls = prev.calls + 1
                const input_tokens = prev.input_tokens + inputTokenEstimate
                const output_tokens = prev.output_tokens + outputTokenEstimate
                const tokens = input_tokens + output_tokens
                const costDelta = calculateCost(model, inputTokenEstimate, outputTokenEstimate)
                const cost = prev.cost + costDelta

                return {
                  model,
                  calls,
                  tokens,
                  input_tokens,
                  output_tokens,
                  cost,
                }
              })
            }

            // Extract relative path from output_dir
            let dirName = ''
            if (data.output_dir) {
              const parts = data.output_dir.split(/[/\\]/)
              dirName = parts[parts.length - 1]
            }

            setSolverState(prev => ({
              ...prev,
              messages: [
                ...prev.messages,
                {
                  role: 'assistant' as const,
                  content: data.final_answer,
                  outputDir: dirName,
                },
              ],
              isSolving: false,
            }))
            ws.close()
          } else if (data.type === 'error') {
            addSolverLog({
              type: 'error',
              content: `Error: ${data.content || data.message || 'Unknown error'}`,
            })
            setSolverState(prev => ({ ...prev, isSolving: false }))
          }
        } catch (e) {
          console.error('Failed to parse solver WebSocket message', e)
          addSolverLog({ type: 'error', content: 'Invalid server response format.' })
        }
      }

      ws.onerror = () => {
        addSolverLog({ type: 'error', content: 'Connection error' })
        setSolverState(prev => ({
          ...prev,
          isSolving: false,
          // Use constant keys to avoid maintenance drift
          agentStatus: Object.keys(DEFAULT_SOLVER_AGENT_STATUS).reduce(
            (acc, key) => ({ ...acc, [key]: 'error' }),
            {} as typeof DEFAULT_SOLVER_AGENT_STATUS
          ),
          progress: {
            stage: null,
            progress: {},
          },
        }))
      }

      ws.onclose = () => {
        if (solverWs.current === ws) {
          solverWs.current = null
        }
      }
    },
    [addSolverLog]
  )

  const stopSolver = useCallback(() => {
    if (solverWs.current) {
      solverWs.current.close()
      solverWs.current = null
    }
    setSolverState(prev => ({
      ...prev,
      isSolving: false,
    }))
    addSolverLog({ type: 'system', content: 'Solver stopped by user.' })
  }, [addSolverLog])

  return (
    <SolverContext.Provider
      value={{
        solverState,
        setSolverState,
        startSolver,
        stopSolver,
        tokenStats,
      }}
    >
      {children}
    </SolverContext.Provider>
  )
}

export const useSolver = () => {
  const context = useContext(SolverContext)
  if (!context) throw new Error('useSolver must be used within SolverProvider')
  return context
}
