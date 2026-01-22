// web/context/question/mimicHandler.ts
import { LogEntry } from '@/types/common'
import { QuestionContextState } from '@/types/question'

type StateSetter = React.Dispatch<React.SetStateAction<QuestionContextState>>
type LogAdder = (log: LogEntry) => void

const STAGE_MAP: Record<string, string> = {
  init: 'uploading',
  upload: 'uploading',
  parsing: 'parsing',
  processing: 'extracting',
}

export const handleMimicMessage = (
  data: any,
  setQuestionState: StateSetter,
  addLog: LogAdder,
  ws: WebSocket
) => {
  switch (data.type) {
    case 'log':
      addLog(data)
      break

    case 'status': {
      const mappedStage = STAGE_MAP[data.stage] || data.stage
      addLog({
        type: 'system',
        content: data.content || data.message || `Stage: ${data.stage}`,
      })
      if (mappedStage) {
        setQuestionState(prev => ({
          ...prev,
          progress: { ...prev.progress, stage: mappedStage },
        }))
      }
      break
    }

    case 'progress': {
      const stage = data.stage || 'generating'
      if (data.message) {
        addLog({ type: 'system', content: data.message })
      }

      setQuestionState(prev => {
        const nextProgress = {
          ...prev.progress,
          stage: stage,
          progress: {
            ...prev.progress.progress,
            current: data.current ?? prev.progress.progress.current,
            total: data.total_questions ?? data.total ?? prev.progress.progress.total,
            status: data.status,
          },
        }

        // Specific logic for extracting completion
        if (stage === 'extracting' && data.status === 'complete' && data.reference_questions) {
          nextProgress.progress.total = data.total_questions || data.reference_questions.length
        }

        return { ...prev, progress: nextProgress }
      })
      break
    }

    case 'question_update': {
      const statusMessage =
        data.status === 'generating'
          ? `Generating mimic question ${data.index}...`
          : data.status === 'failed'
            ? `Question ${data.index} failed: ${data.error}`
            : `Question ${data.index}: ${data.status}`

      addLog({
        type: data.status === 'failed' ? 'warning' : 'system',
        content: statusMessage,
      })

      if (data.current !== undefined) {
        setQuestionState(prev => ({
          ...prev,
          progress: {
            ...prev.progress,
            progress: { ...prev.progress.progress, current: data.current },
          },
        }))
      }
      break
    }

    case 'result': {
      const isExtended = data.extended || data.validation?.decision === 'extended'
      addLog({
        type: 'success',
        content: `✅ Question ${data.index || (data.current ?? 0)} generated successfully`,
      })

      setQuestionState(prev => ({
        ...prev,
        results: [
          ...prev.results,
          {
            success: true,
            question_id: data.question_id || `q_${prev.results.length + 1}`,
            question: data.question,
            validation: data.validation,
            rounds: data.rounds || 1,
            reference_question: data.reference_question,
            extended: isExtended,
          },
        ],
        progress: {
          ...prev.progress,
          stage: 'generating',
          progress: {
            ...prev.progress.progress,
            current: data.current ?? prev.results.length + 1,
            total: data.total ?? prev.progress.progress.total ?? 1,
          },
          extendedQuestions: (prev.progress.extendedQuestions || 0) + (isExtended ? 1 : 0),
        },
      }))
      break
    }

    case 'summary':
      addLog({
        type: 'success',
        content: `Generation complete: ${data.successful}/${data.total_reference} succeeded`,
      })
      setQuestionState(prev => ({
        ...prev,
        progress: {
          ...prev.progress,
          stage: 'generating',
          progress: {
            current: data.successful,
            total: data.total_reference,
          },
          completedQuestions: data.successful,
          failedQuestions: data.failed,
        },
      }))
      break

    case 'complete':
      addLog({
        type: 'success',
        content: '✅ Mimic generation completed!',
      })
      setQuestionState(prev => ({
        ...prev,
        step: 'result',
        progress: {
          ...prev.progress,
          stage: 'complete',
          completedQuestions: prev.results.length,
        },
      }))
      if (ws.readyState === WebSocket.OPEN) ws.close()
      break

    case 'error':
      addLog({
        type: 'error',
        content: `Error: ${data.content || data.message || 'Unknown error'}`,
      })
      setQuestionState(prev => ({
        ...prev,
        step: 'config',
        progress: { stage: null, progress: {} },
      }))
      break
  }
}
