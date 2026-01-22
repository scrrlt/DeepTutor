'use client'

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'
import { wsUrl } from '@/lib/api'
import { calculateCost, estimateTokens } from '@/lib/token-utils'
import { handleMimicMessage } from './mimicHandler' // Imported handler
import {
  QuestionContextState,
  QuestionTokenStats,
  INITIAL_QUESTION_CONTEXT_STATE,
  DEFAULT_QUESTION_AGENT_STATUS,
  DEFAULT_QUESTION_TOKEN_STATS,
} from '@/types/question'
import { LogEntry } from '@/types/common'

interface QuestionContextType {
  questionState: QuestionContextState
  setQuestionState: React.Dispatch<React.SetStateAction<QuestionContextState>>
  startQuestionGen: (topic: string, diff: string, type: string, count: number, kb: string) => void
  startMimicQuestionGen: (
    file: File | null,
    paperPath: string,
    kb: string,
    maxQuestions?: number
  ) => void
  resetQuestionGen: () => void
  tokenStats: QuestionTokenStats
}

const QuestionContext = createContext<QuestionContextType | undefined>(undefined)

const DEFAULT_TOKEN_MODEL = 'gpt-3.5-turbo'
const LOG_FLUSH_INTERVAL_MS = 100
const MAX_LOG_ENTRIES = 500

export function QuestionProvider({ children }: { children: React.ReactNode }) {
  const [questionState, setQuestionState] = useState<QuestionContextState>(
    INITIAL_QUESTION_CONTEXT_STATE
  )
  const [tokenStats, setTokenStats] = useState<QuestionTokenStats>(DEFAULT_QUESTION_TOKEN_STATS)

  const questionWs = useRef<WebSocket | null>(null)
  const lastTokenModel = useRef(DEFAULT_TOKEN_MODEL)
  const logBuffer = useRef<LogEntry[]>([])

  // Log Flush Loop
  useEffect(() => {
    const interval = setInterval(() => {
      if (logBuffer.current.length > 0) {
        const logsToFlush = [...logBuffer.current]
        logBuffer.current = []
        setQuestionState(prev => {
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

  const addQuestionLog = useCallback((log: LogEntry) => {
    logBuffer.current.push(log)
  }, [])

  // --- Start Normal Generation ---
  const startQuestionGen = useCallback(
    (topic: string, diff: string, type: string, count: number, kb: string) => {
      if (questionWs.current) questionWs.current.close()
      lastTokenModel.current = DEFAULT_TOKEN_MODEL
      setTokenStats(DEFAULT_QUESTION_TOKEN_STATS)
      logBuffer.current = []

      setQuestionState(prev => ({
        ...prev,
        step: 'generating',
        mode: 'knowledge',
        logs: [],
        results: [],
        topic,
        difficulty: diff,
        type,
        count,
        selectedKb: kb,
        progress: {
          stage: count > 1 ? 'planning' : 'generating',
          progress: { current: 0, total: count },
          subFocuses: [],
          activeQuestions: [],
          completedQuestions: 0,
          failedQuestions: 0,
        },
        agentStatus: { ...DEFAULT_QUESTION_AGENT_STATUS },
      }))

      const ws = new WebSocket(wsUrl('/api/v1/question/generate'))
      questionWs.current = ws

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            requirement: {
              knowledge_point: topic,
              difficulty: diff,
              question_type: type,
              additional_requirements: 'Ensure clarity and academic rigor.',
            },
            count: count,
            kb_name: kb,
          })
        )
        addQuestionLog({
          type: 'system',
          content: 'Initializing Generator...',
        })
      }

      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'log') {
            addQuestionLog(data)
            // Lightweight parsing for progress bars (Normal Mode)
            if (data.content && typeof data.content === 'string') {
              const match = data.content.match(/(\d+)\/(\d+)/)
              if (match) {
                setQuestionState(prev => ({
                  ...prev,
                  progress: {
                    ...prev.progress,
                    stage: 'generating',
                    progress: {
                      current: parseInt(match[1]),
                      total: parseInt(match[2]),
                    },
                  },
                }))
              }
              // ... other regex checks can remain or be moved to a handler if they grow
            }
          } else if (data.type === 'result') {
            const qPayload = data.question ? JSON.stringify(data.question) : ''
            const inputDelta = estimateTokens(topic)
            const outputDelta = estimateTokens(qPayload)
            const model = data.model || lastTokenModel.current || DEFAULT_TOKEN_MODEL
            lastTokenModel.current = model

            setTokenStats(prev => {
              const calls = prev.calls + 1
              const input_tokens = prev.input_tokens + inputDelta
              const output_tokens = prev.output_tokens + outputDelta
              const tokens = input_tokens + output_tokens
              const costDelta = calculateCost(model, inputDelta, outputDelta)
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
            const isExtended = data.extended || data.validation?.decision === 'extended'
            setQuestionState(prev => ({
              ...prev,
              results: [
                ...prev.results,
                {
                  success: true,
                  question_id: data.question_id || `q_${prev.results.length + 1}`,
                  question: data.question,
                  rounds: data.rounds || 1,
                  validation: data.validation,
                  extended: isExtended,
                },
              ],
              progress: {
                ...prev.progress,
                completedQuestions: prev.results.length + 1,
                progress: { ...prev.progress.progress, current: prev.results.length + 1 },
              },
            }))
            addQuestionLog({ type: 'success', content: `Question generated.` })
          } else if (data.type === 'complete') {
            setQuestionState(prev => ({
              ...prev,
              step: 'result',
              progress: { ...prev.progress, stage: 'complete' },
            }))
            ws.close()
          } else if (data.type === 'error') {
            addQuestionLog({ type: 'error', content: data.content || 'Error' })
            setQuestionState(prev => ({ ...prev, progress: { stage: null, progress: {} } }))
          }
          // Note: Add other normal-mode event types here (plan_ready, etc.) from original file
        } catch (e) {
          console.error('Failed to parse WebSocket message', e)
          addQuestionLog({ type: 'error', content: 'Invalid server response.' })
        }
      }

      ws.onerror = () => {
        addQuestionLog({ type: 'error', content: 'WebSocket connection error' })
        setQuestionState(prev => ({ ...prev, step: 'config' }))
      }
    },
    [addQuestionLog]
  )

  // --- Start Mimic Generation ---
  const startMimicQuestionGen = useCallback(
    async (file: File | null, paperPath: string, kb: string, maxQuestions?: number) => {
      if (questionWs.current) questionWs.current.close()

      const hasFile = file !== null
      const hasParsedPath = paperPath && paperPath.trim() !== ''

      if (!hasFile && !hasParsedPath) {
        addQuestionLog({
          type: 'error',
          content: 'Please upload a PDF file or provide a parsed exam directory',
        })
        return
      }

      setQuestionState(prev => ({
        ...prev,
        step: 'generating',
        mode: 'mimic',
        logs: [],
        results: [],
        selectedKb: kb,
        uploadedFile: file,
        paperPath: paperPath,
        progress: {
          stage: hasFile ? 'uploading' : 'parsing',
          progress: { current: 0, total: maxQuestions || 1 },
        },
        agentStatus: { ...DEFAULT_QUESTION_AGENT_STATUS },
      }))

      const ws = new WebSocket(wsUrl('/api/v1/question/mimic'))
      questionWs.current = ws

      ws.onopen = async () => {
        if (hasFile && file) {
          addQuestionLog({ type: 'system', content: 'Preparing to upload PDF file...' })
          const reader = new FileReader()

          reader.onload = () => {
            try {
              if (reader.result && typeof reader.result === 'string') {
                const base64Data = reader.result.split(',')[1]
                ws.send(
                  JSON.stringify({
                    mode: 'upload',
                    pdf_data: base64Data,
                    pdf_name: file.name,
                    kb_name: kb,
                    max_questions: maxQuestions,
                  })
                )
                addQuestionLog({ type: 'system', content: `Uploaded: ${file.name}, parsing...` })
              }
            } catch (e) {
              addQuestionLog({ type: 'error', content: 'File processing failed' })
              ws.close()
            }
          }
          reader.readAsDataURL(file)
        } else {
          ws.send(
            JSON.stringify({
              mode: 'parsed',
              paper_path: paperPath,
              kb_name: kb,
              max_questions: maxQuestions,
            })
          )
          addQuestionLog({ type: 'system', content: 'Initializing Mimic Generator...' })
        }
      }

      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data)
          // DELEGATE TO EXTRACTED HANDLER
          handleMimicMessage(data, setQuestionState, addQuestionLog, ws)
        } catch (e) {
          console.error('Invalid Mimic WS Message', e)
          addQuestionLog({ type: 'error', content: 'Server protocol error' })
        }
      }

      ws.onerror = () => {
        addQuestionLog({ type: 'error', content: 'WebSocket connection error' })
        setQuestionState(prev => ({ ...prev, step: 'config' }))
      }
    },
    [addQuestionLog]
  )

  const resetQuestionGen = useCallback(() => {
    setQuestionState(prev => ({
      ...INITIAL_QUESTION_CONTEXT_STATE,
      step: 'config',
    }))
  }, [])

  return (
    <QuestionContext.Provider
      value={{
        questionState,
        setQuestionState,
        startQuestionGen,
        startMimicQuestionGen,
        resetQuestionGen,
        tokenStats,
      }}
    >
      {children}
    </QuestionContext.Provider>
  )
}

export const useQuestion = () => {
  const context = useContext(QuestionContext)
  if (!context) throw new Error('useQuestion must be used within QuestionProvider')
  return context
}
