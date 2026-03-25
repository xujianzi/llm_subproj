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
      const reader  = res.body!.getReader()
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
          if (eventLine === 'data')  callbacks.onData(JSON.parse(dataLine) as ChatDataPayload)
          if (eventLine === 'error') callbacks.onError(dataLine)
          if (eventLine === 'done')  callbacks.onDone()
        }
      }
    })
    .catch((e) => {
      if ((e as Error).name !== 'AbortError') callbacks.onError(String(e))
    })

  return () => ctrl.abort()
}
