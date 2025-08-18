import React, { useState } from 'react'

// TypeScript declaration for Tauri global
declare global {
  interface Window {
    __TAURI__: {
      tauri: {
        invoke: (command: string, args?: any) => Promise<any>;
      };
    };
  }
}

function App() {
  const [prompt, setPrompt] = useState('')
  const [generatedCode, setGeneratedCode] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [decision, setDecision] = useState<any | null>(null)
  const [consensus, setConsensus] = useState<any | null>(null)
  const [warmup, setWarmup] = useState<{ median?: number, history: number[] }>({ history: [] })

  // Test Tauri IPC
  const testTauri = async () => {
    try {
      console.log('🧪 Testing Tauri IPC...')
      const result = await window.__TAURI__.tauri.invoke('test_tauri')
      console.log('✅ Tauri test result:', result)
      return true
    } catch (error) {
      console.error('❌ Tauri test failed:', error)
      return false
    }
  }

  const fetchArtifacts = async () => {
    try {
      const d = await window.__TAURI__.tauri.invoke('get_latest_selector_decision')
      setDecision(d)
    } catch {}
    try {
      const c = await window.__TAURI__.tauri.invoke('get_latest_consensus')
      setConsensus(c)
    } catch {}
    try {
      const hist = await window.__TAURI__.tauri.invoke('get_warmup_history', { limit: 20 })
      if (Array.isArray(hist)) {
        const medians = hist
          .map((x: any) => (typeof x?.median_ttft_ms === 'number' ? x.median_ttft_ms : undefined))
          .filter((x: any) => typeof x === 'number')
        setWarmup({ median: medians[0], history: medians.slice(0, 20).reverse() })
      }
    } catch {}
  }

  React.useEffect(() => {
    fetchArtifacts()
    const id = setInterval(fetchArtifacts, 4000)
    return () => clearInterval(id)
  }, [])

  const runWarmup = async () => {
    try {
      await window.__TAURI__.tauri.invoke('run_warmup_smoke')
      await fetchArtifacts()
    } catch (e) {
      console.error('Warm-up failed', e)
    }
  }

  const handleGenerate = async () => {
    console.log('🎯 Button clicked!') // DEBUG

    if (!prompt.trim()) {
      console.log('❌ Empty prompt!') // DEBUG
      return
    }

    console.log('📝 Prompt:', prompt) // DEBUG

    // Test Tauri first
    const tauriWorks = await testTauri()
    console.log('🔍 Tauri available:', tauriWorks) // DEBUG

    setIsGenerating(true)
    setGeneratedCode('// Generating code... Please wait...')

    try {
      console.log('⏳ Setting loading state...') // DEBUG

      if (!tauriWorks) {
        console.log('⚠️ Tauri not available, using fallback') // DEBUG
        throw new Error('Tauri not available')
      }

      console.log('🚀 Calling Rust backend...') // DEBUG

      // Call Rust backend via Tauri
      const result = await window.__TAURI__.tauri.invoke('generate_code', {
        prompt: prompt.trim()
      })

      console.log('✅ Got result from Rust:', result) // DEBUG

      setGeneratedCode(result as string)
      console.log('✅ Code generated successfully from Rust backend!')

    } catch (error) {
      console.error('❌ Rust backend failed:', error)

      // Enhanced fallback based on prompt content
      let fallbackCode = ''
      const promptLower = prompt.toLowerCase()

      console.log('🔄 Using fallback code for:', promptLower) // DEBUG

      if (promptLower.includes('python') && promptLower.includes('sort')) {
        fallbackCode = `def sort_list(numbers):
    """Sort a list of numbers in ascending order."""
    return sorted(numbers)

# Example usage:
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = sort_list(my_list)
print(f"Original: {my_list}")
print(f"Sorted: {sorted_list}")`
      } else if (promptLower.includes('python') && promptLower.includes('function')) {
        fallbackCode = `def example_function():
    """A sample Python function."""
    result = "Hello from CodeConductor!"
    print(result)
    return result

# Call the function
output = example_function()
print(f"Function returned: {output}")`
      } else if (promptLower.includes('javascript') || promptLower.includes('js')) {
        fallbackCode = `function exampleFunction() {
    const message = "Hello from CodeConductor!";
    console.log(message);
    return message;
}

// Call the function
const result = exampleFunction();
console.log(\`Function returned: \${result}\`);`
      } else if (promptLower.includes('rust')) {
        fallbackCode = `fn example_function() -> String {
    let message = "Hello from CodeConductor!";
    println!("{}", message);
    message.to_string()
}

fn main() {
    let result = example_function();
    println!("Function returned: {}", result);
}`
      } else {
        fallbackCode = `// Generated code for: "${prompt}"
// Note: Using fallback generation (Rust backend not connected)

function processData(input) {
  // TODO: Implement your logic here
  console.log('Processing:', input);
  return input;
}

// Example usage:
const result = processData('Hello CodeConductor!');
console.log('Result:', result);`
      }

      setGeneratedCode(fallbackCode)
      console.log('⚠️ Using enhanced fallback code')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#f5f5f5',
      minHeight: '100vh',
      fontFamily: 'system-ui, sans-serif'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{
          fontSize: '2rem',
          color: '#333',
          marginBottom: '10px'
        }}>
          CodeConductor
        </h1>
        <p style={{ color: '#666', marginBottom: '30px' }}>
          Personal AI Development Platform
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'start' }}>
          {/* Prompt Input */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ marginBottom: '15px' }}>AI Prompt</h3>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                Describe what you want to generate:
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g., Create a Python function that sorts a list of numbers..."
                style={{
                  width: '100%',
                  height: '120px',
                  padding: '12px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  resize: 'none',
                  fontFamily: 'monospace'
                }}
              />
            </div>
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              style={{
                width: '100%',
                marginTop: '15px',
                padding: '12px',
                backgroundColor: isGenerating || !prompt.trim() ? '#ccc' : '#007acc',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isGenerating || !prompt.trim() ? 'not-allowed' : 'pointer',
                fontSize: '16px'
              }}
            >
              {isGenerating ? 'Generating...' : 'Generate Code'}
            </button>
          </div>

          {/* Generated Code */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ marginBottom: '15px' }}>Generated Code</h3>
            <textarea
              value={generatedCode}
              onChange={(e) => setGeneratedCode(e.target.value)}
              placeholder="Generated code will appear here..."
              style={{
                width: '100%',
                height: '300px',
                padding: '12px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '14px',
                backgroundColor: '#1e1e1e',
                color: '#ffffff',
                resize: 'none'
              }}
            />
          </div>
        </div>

        {/* Why chosen + Consensus + Warm-up */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginTop: '20px' }}>
          {/* Why chosen card */}
          <div style={{ background: 'white', padding: 16, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3 style={{ marginTop: 0 }}>Why chosen</h3>
            {decision ? (
              <div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ background: '#eef', color: '#335', padding: '2px 8px', borderRadius: 12, fontSize: 12 }}>
                    Policy: {decision.policy || 'latency'}
                  </span>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <strong>Chosen</strong>: {decision.chosen || decision.selected_model || '—'}
                </div>
                {decision.scores ? (
                  <div>
                    <div style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>Top-3 candidates</div>
                    <ol style={{ margin: 0, paddingLeft: 16 }}>
                      {Object.entries(decision.scores)
                        .sort((a: any, b: any) => (b[1] as number) - (a[1] as number))
                        .slice(0, 3)
                        .map(([mid, sc]: any) => (
                          <li key={mid} style={{ fontFamily: 'monospace' }}>{mid}: {(sc as number).toFixed(3)}</li>
                        ))}
                    </ol>
                  </div>
                ) : null}
                <div style={{ marginTop: 10 }}>
                  <button onClick={() => {
                    try {
                      navigator.clipboard.writeText(JSON.stringify(decision, null, 2))
                    } catch {}
                  }}>Copy decision JSON</button>
                </div>
              </div>
            ) : (
              <div>No recent decision. Run a task to populate artifacts.</div>
            )}
          </div>

          {/* Consensus + Warm-up side panel */}
          <div style={{ display: 'grid', gridTemplateRows: 'auto auto', gap: 16 }}>
            <div style={{ background: 'white', padding: 16, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong>Consensus</strong>{' '}
                  {typeof consensus?.winner?.score === 'number' ? (
                    <span title="fast CodeBLEU + heuristik" style={{ color: '#333' }}>
                      {(consensus.winner.score as number).toFixed(2)}
                    </span>
                  ) : (
                    <span>—</span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#555' }}>{consensus?.winner?.model || '—'}</div>
              </div>
            </div>

            <div style={{ background: 'white', padding: 16, borderRadius: 8, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong>Warm-up</strong>{' '}
                  {typeof warmup.median === 'number' ? (
                    <span>median {warmup.median} ms</span>
                  ) : (
                    <span>no data</span>
                  )}
                </div>
                <button onClick={runWarmup}>Run warm-up</button>
              </div>
              {/* simple sparkline */}
              <div style={{ marginTop: 8, height: 40, display: 'flex', gap: 2, alignItems: 'end' }}>
                {warmup.history.map((v, i) => (
                  <div key={i} title={`${v} ms`} style={{ width: 6, height: Math.max(4, Math.min(36, 36 - (v - (warmup.median || 0)) / 10)), background: '#4ecdc4' }} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
