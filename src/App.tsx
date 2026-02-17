import { useState, useRef } from 'react'
import './App.css'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'

type Message = {
  role: 'user' | 'assistant'
  content: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streamingContent, setStreamingContent] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const streamedRef = useRef('')

  const MODEL_DIR = 'C:\\Users\\russ1\\Documents\\Concierge\\models\\tinyllama'
  const EVENTS_PATH = 'C:\\Users\\russ1\\Documents\\Concierge\\data\\events.json'

  async function handleSend() {
    const prompt = input.trim()
    if (!prompt || isStreaming) return

    setMessages((prev) => [...prev, { role: 'user', content: prompt }])
    setInput('')
    streamedRef.current = ''
    setStreamingContent('')
    setIsStreaming(true)

    const unlisten = await listen<string>('chat-token', (event: any) => {
      streamedRef.current += event.payload
      setStreamingContent((prev) => prev + event.payload)
    })

    invoke('generate_stream', {
      prompt,
      modelDir: MODEL_DIR,
      eventsPath: EVENTS_PATH ?? null,
    })
      .catch((err: any) => {
        console.error(err)
        setStreamingContent((prev) => prev + `\n[Error: ${err}]`)
      })
      .finally(() => {
        setMessages((prev) => [...prev, { role: 'assistant', content: streamedRef.current }])
        setStreamingContent('')
        streamedRef.current = ''
        setIsStreaming(false)
        unlisten()
      })
  }

  

  return (
    <div className="app">
      <h1>Concierge</h1>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            <span className="message-role">{msg.role}</span>
            <p className="message-content">{msg.content}</p>
          </div>
        ))}
        {streamingContent ? (
          <div className="message message-assistant">
            <span className="message-role">assistant</span>
            <p className="message-content">{streamingContent}</p>
          </div>
        ) : null}
      </div>
      <div className="input-row">
        <input
          type="text"
          className="input"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          disabled={isStreaming}
        />
        <button
          type="button"
          className="send-btn"
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default App
