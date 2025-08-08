import React, { useEffect, useRef, useState } from 'react'

interface MonacoEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  theme?: string
  streaming?: boolean
}

const MonacoEditor: React.FC<MonacoEditorProps> = ({
  value,
  onChange,
  language = 'typescript',
  theme = 'vs-dark',
  streaming = false
}) => {
  const editorRef = useRef<HTMLDivElement>(null)
  const monacoRef = useRef<any>(null)
  const esRef = useRef<EventSource | null>(null)
  const bufferRef = useRef<string[]>([])
  const flushTimerRef = useRef<number | null>(null)
  const [status, setStatus] = useState<'connected' | 'paused' | 'disconnected'>('disconnected')
  const [ttftMs, setTtftMs] = useState<number | null>(null)
  const [tps, setTps] = useState<number | null>(null)
  const firstTokenAtRef = useRef<number | null>(null)
  const tokenCountRef = useRef<number>(0)

  useEffect(() => {
    if (!editorRef.current) return

    // Simple textarea fallback for now
    // In a real implementation, you'd load Monaco Editor here
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.style.width = '100%'
    textarea.style.height = '100%'
    textarea.style.border = 'none'
    textarea.style.outline = 'none'
    textarea.style.resize = 'none'
    textarea.style.fontFamily = 'monospace'
    textarea.style.fontSize = '14px'
    textarea.style.backgroundColor = theme === 'vs-dark' ? '#1e1e1e' : '#ffffff'
    textarea.style.color = theme === 'vs-dark' ? '#ffffff' : '#000000'
    
    textarea.addEventListener('input', (e) => {
      onChange((e.target as HTMLTextAreaElement).value)
    })

    editorRef.current.innerHTML = ''
    editorRef.current.appendChild(textarea)

    return () => {
      if (editorRef.current) {
        editorRef.current.innerHTML = ''
      }
    }
  }, [value, onChange, theme])

  // Streaming hook via SSE
  useEffect(() => {
    if (!streaming) {
      // cleanup any existing stream
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
      setStatus('disconnected')
      return
    }

    try {
      const url = `/stream?request_id=${encodeURIComponent(crypto.randomUUID())}`
      const es = new EventSource(url)
      esRef.current = es
      setStatus('connected')
      firstTokenAtRef.current = null
      tokenCountRef.current = 0
      bufferRef.current = []

      const flush = () => {
        if (!editorRef.current) return
        if (bufferRef.current.length === 0) return
        const node = editorRef.current.querySelector('textarea') as HTMLTextAreaElement | null
        if (!node) return
        const append = bufferRef.current.join('')
        bufferRef.current = []
        const atEnd = node.selectionStart === node.value.length && node.selectionEnd === node.value.length
        node.value += append
        onChange(node.value)
        if (atEnd) {
          node.selectionStart = node.value.length
          node.selectionEnd = node.value.length
          node.scrollTop = node.scrollHeight
        }
      }

      const scheduleFlush = () => {
        if (flushTimerRef.current) return
        flushTimerRef.current = window.setTimeout(() => {
          flushTimerRef.current = null
          flush()
        }, 16) // ~60fps micro-batch
      }

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data)
          if (data.ttft_ms != null && firstTokenAtRef.current == null) {
            setTtftMs(data.ttft_ms)
            firstTokenAtRef.current = performance.now()
          }
          if (data.done) {
            flush()
            setStatus('paused')
            es.close()
            esRef.current = null
            // compute tokens/s
            if (firstTokenAtRef.current != null) {
              const dt = (performance.now() - firstTokenAtRef.current) / 1000
              if (dt > 0) setTps(tokenCountRef.current / dt)
            }
            return
          }
          if (typeof data.token === 'string') {
            bufferRef.current.push(data.token)
            if (bufferRef.current.length > 100) {
              // backpressure: flush and trim
              flush()
              bufferRef.current = bufferRef.current.slice(-10)
            }
            tokenCountRef.current += 1
            scheduleFlush()
          }
        } catch (_e) {
          // ignore parse errors
        }
      }
      es.onerror = () => {
        setStatus('disconnected')
        es.close()
        esRef.current = null
        // simple auto-reconnect
        setTimeout(() => {
          if (streaming) {
            // trigger re-effect by toggling state
            setStatus('connected')
          }
        }, 500)
      }
    } catch (_e) {
      setStatus('disconnected')
    }

    return () => {
      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current)
        flushTimerRef.current = null
      }
      setStatus('disconnected')
    }
  }, [streaming, onChange])

  return (
    <div className="w-full h-full min-h-[300px] flex flex-col">
      <div className="flex-1" 
        ref={editorRef}
        style={{ 
          backgroundColor: theme === 'vs-dark' ? '#1e1e1e' : '#ffffff',
          color: theme === 'vs-dark' ? '#ffffff' : '#000000'
        }}
      />
      <div className="h-7 text-xs px-2 py-1 flex items-center gap-3 border-t border-gray-700">
        <span>Status: {status}</span>
        <span>TTFT: {ttftMs != null ? `${ttftMs} ms` : '—'}</span>
        <span>Tok/s: {tps != null ? tps.toFixed(1) : '—'}</span>
      </div>
    </div>
  )
}

export default MonacoEditor 