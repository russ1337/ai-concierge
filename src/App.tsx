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
  const [maxTokens, setMaxTokens] = useState(128)
  const [temperature, setTemperature] = useState(0)
  const [useOllama, setUseOllama] = useState(false)
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434')
  const [ollamaModel, setOllamaModel] = useState('tinyllama')
  const streamedRef = useRef('')

  const MODEL_DIR = 'C:\\Users\\russ1\\Documents\\Concierge\\models\\tinyllama'
  const EVENTS_PATH = 'C:\\Users\\russ1\\Documents\\Concierge\\data\\events.json'

  /** Strip model-generated "User:" or "<|user|>" so we never show fake user prompts. */
  function stripFakeUserPrompts(text: string): string {
    const markers = ['\nUser:', '\n<|user|>', '\n\nUser:']
    const positions = markers.map((m) => text.indexOf(m)).filter((i) => i !== -1)
    const truncateAt = positions.length ? Math.min(...positions) : text.length
    return text.slice(0, truncateAt).trimEnd()
  }

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

    const currentDate = new Date().toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })

    invoke('generate_stream', {
      prompt,
      modelDir: MODEL_DIR,
      eventsPath: EVENTS_PATH ?? null,
      currentDate,
      maxTokens,
      temperature,
      ollamaUrl: useOllama ? ollamaUrl : null,
      ollamaModel: useOllama ? ollamaModel : null,
    })
      .catch((err: any) => {
        console.error(err)
        setStreamingContent((prev) => prev + `\n[Error: ${err}]`)
      })
      .finally(() => {
        const raw = streamedRef.current
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: stripFakeUserPrompts(raw) },
        ])
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
      <div className="settings-row">
        <label className="settings-label">
          <input
            type="checkbox"
            checked={useOllama}
            onChange={(e) => setUseOllama(e.target.checked)}
            disabled={isStreaming}
          />
          {' '}Ollama (GPU)
        </label>
        {useOllama ? (
          <>
            <label htmlFor="ollama-url" className="settings-label">URL</label>
            <input
              id="ollama-url"
              type="text"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              disabled={isStreaming}
              className="settings-input settings-input-wide"
              placeholder="http://localhost:11434"
            />
            <label htmlFor="ollama-model" className="settings-label">Model</label>
            <input
              id="ollama-model"
              type="text"
              value={ollamaModel}
              onChange={(e) => setOllamaModel(e.target.value)}
              disabled={isStreaming}
              className="settings-input"
              placeholder="tinyllama"
            />
          </>
        ) : null}
      </div>
      <div className="settings-row">
        <label htmlFor="max-tokens" className="settings-label">
          Max tokens
        </label>
        <input
          id="max-tokens"
          type="number"
          min={1}
          max={1024}
          value={maxTokens}
          onChange={(e) => setMaxTokens(Math.max(1, Math.min(1024, Number(e.target.value) || 128)))}
          disabled={isStreaming}
          className="settings-input"
        />
        <label htmlFor="temperature" className="settings-label">
          Temperature
        </label>
        <input
          id="temperature"
          type="number"
          min={0}
          max={2}
          step={0.1}
          value={temperature}
          onChange={(e) =>
            setTemperature(
              Math.max(0, Math.min(2, Number(e.target.value) ?? 0))
            )
          }
          disabled={isStreaming}
          className="settings-input"
        />
      </div>
    </div>
  )
}

export default App
