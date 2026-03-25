import { useRef, useState } from 'react'
import { useMapStore } from '../../store/useMapStore'
import { streamChat } from '../../api/chatApi'
import type { ChatDataPayload } from '../../types'

export function ChatPanel() {
  const { chatHistory, chatOpen, addChatMessage, setChatOpen, updateFromChatData } = useMapStore()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [aiDraft, setAiDraft] = useState('')
  // useRef to avoid stale closure in onDone callback — always holds latest draft value
  const aiDraftRef = useRef('')
  const stopRef = useRef<(() => void) | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  function send() {
    if (!input.trim() || loading) return
    const userMsg = { role: 'user' as const, content: input.trim() }
    addChatMessage(userMsg)
    setInput('')
    setLoading(true)
    aiDraftRef.current = ''
    setAiDraft('')

    stopRef.current = streamChat(
      userMsg.content,
      chatHistory,
      {
        onText(text) {
          aiDraftRef.current = text   // keep ref in sync
          setAiDraft(text)
        },
        onData(payload: ChatDataPayload) { updateFromChatData(payload) },
        onError(msg) {
          aiDraftRef.current = `Error: ${msg}`
          setAiDraft(`Error: ${msg}`)
        },
        onDone() {
          setLoading(false)
          // Read from ref — not from closed-over state — to get the final value
          if (aiDraftRef.current) {
            addChatMessage({ role: 'assistant', content: aiDraftRef.current })
          }
          aiDraftRef.current = ''
          setAiDraft('')
          setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
        },
      }
    )
  }

  // Minimized bubble
  if (!chatOpen) {
    return (
      <button
        onClick={() => setChatOpen(true)}
        className="absolute bottom-8 right-4 z-20 w-14 h-14 rounded-full
          bg-gradient-to-br from-primary to-accent shadow-xl flex items-center justify-center
          text-white text-2xl hover:scale-105 transition"
        title="Open Chat Agent"
      >
        💬
      </button>
    )
  }

  return (
    <div className="absolute bottom-8 right-4 z-20 w-80 md:w-96
      bg-panel/90 backdrop-blur border border-border rounded-2xl shadow-2xl
      flex flex-col overflow-hidden" style={{ height: '420px' }}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <span className="text-sm font-semibold text-text">Chat Agent</span>
        <div className="flex gap-2">
          <button onClick={() => setChatOpen(false)}
            className="text-muted hover:text-text text-lg leading-none">─</button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-sm">
        {chatHistory.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-xl px-3 py-2 ${
              m.role === 'user'
                ? 'bg-primary text-white'
                : 'bg-border text-text'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-xl px-3 py-2 bg-border text-text whitespace-pre-wrap">
              {aiDraft || <span className="animate-pulse text-muted">思考中...</span>}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-2 flex gap-2">
        <input
          className="flex-1 bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text
            placeholder:text-muted focus:outline-none focus:border-primary"
          placeholder="问我任何 ACS 数据问题..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          disabled={loading}
        />
        <button
          onClick={send}
          disabled={loading}
          className="px-3 py-1.5 rounded-lg text-sm font-medium text-white
            bg-gradient-to-r from-primary to-accent hover:opacity-90 disabled:opacity-50"
        >
          发送
        </button>
      </div>
    </div>
  )
}
