import type { ChatMessage, ChatDataPayload } from '../types'

export interface ChatStreamCallbacks {
  onText:  (text: string) => void
  onData:  (payload: ChatDataPayload) => void
  onError: (msg: string) => void
  onDone:  () => void
}

export function streamChat(
  message: string,
  history: ChatMessage[],
  callbacks: ChatStreamCallbacks,
): () => void {
  const ctrl = new AbortController()

  fetch('/api/chat/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ message, history }),
    signal:  ctrl.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        callbacks.onError(`HTTP ${res.status}: ${await res.text()}`)
        callbacks.onDone()
        return
      }
      if (!res.body) {
        callbacks.onError('No response body')
        callbacks.onDone()
        return
      }
      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() ?? ''

        for (const part of parts) {
          const eventLine = part.match(/^event:\s*(.+)$/m)?.[1]?.trim()
          const dataLine  = part.match(/^data:\s*(.*)$/m)?.[1]?.trim() ?? ''

          if (eventLine === 'text')  callbacks.onText(dataLine)
          if (eventLine === 'data') {
            try {
              callbacks.onData(JSON.parse(dataLine))
            } catch {
              callbacks.onError(`Invalid JSON in data event: ${dataLine.slice(0, 100)}`)
            }
          }
          if (eventLine === 'error') callbacks.onError(dataLine)
          if (eventLine === 'done')  callbacks.onDone()
        }
      }

      // Flush any remaining partial SSE event
      if (buf.trim()) {
        const eventLine = buf.match(/^event:\s*(.+)$/m)?.[1]?.trim()
        const dataLine  = buf.match(/^data:\s*(.*)$/m)?.[1]?.trim() ?? ''
        if (eventLine === 'text')  callbacks.onText(dataLine)
        if (eventLine === 'data') { try { callbacks.onData(JSON.parse(dataLine)) } catch {} }
        if (eventLine === 'error') callbacks.onError(dataLine)
        if (eventLine === 'done')  callbacks.onDone()
      }
    })
    .catch((e) => {
      if (!(e instanceof DOMException && e.name === 'AbortError') &&
          !(e instanceof Error && e.name === 'AbortError')) {
        callbacks.onError(String(e))
      }
    })

  return () => ctrl.abort()
}
