import React, { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api";
import { Play, Save, Settings, Zap, Code, FileText } from "lucide-react";
import { cn } from "./lib/utils";
import MonacoEditor from "./components/MonacoEditor";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";

interface CodeGenerationRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  model?: string;
}

interface CodeGenerationResponse {
  id: string;
  code: string;
  model: string;
  timestamp: string;
  metadata: {
    tokens_generated: number;
    generation_time_ms: number;
    model_version: string;
    consensus_score?: number;
  };
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [generatedCode, setGeneratedCode] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGeneration, setLastGeneration] = useState<CodeGenerationResponse | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"editor" | "settings" | "stats">("editor");

  const editorRef = useRef<any>(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const stats = await invoke("get_generation_stats");
      setStats(stats);
    } catch (error) {
      console.error("Failed to load stats:", error);
    }
  };

  const generateCode = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    try {
      const request: CodeGenerationRequest = {
        prompt: prompt.trim(),
        max_tokens: 2048,
        temperature: 0.1,
        model: "vllm-awq",
      };

      const response: CodeGenerationResponse = await invoke("generate_code", { request });
      
      setGeneratedCode(response.code);
      setLastGeneration(response);
      await loadStats();
    } catch (error) {
      console.error("Failed to generate code:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const saveCode = async () => {
    if (!generatedCode) return;
    
    try {
      await invoke("save_code_to_file", {
        code: generatedCode,
        filename: "generated_code.py",
      });
    } catch (error) {
      console.error("Failed to save code:", error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      generateCode();
    }
  };

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center space-x-2">
            <Code className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-bold">CodeConductor</h1>
            <Badge variant="secondary" className="text-xs">
              v0.1.0
            </Badge>
          </div>
          
          <nav className="flex items-center space-x-1">
            <Button
              variant={activeTab === "editor" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("editor")}
            >
              <FileText className="h-4 w-4 mr-2" />
              Editor
            </Button>
            <Button
              variant={activeTab === "stats" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("stats")}
            >
              <Zap className="h-4 w-4 mr-2" />
              Stats
            </Button>
            <Button
              variant={activeTab === "settings" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("settings")}
            >
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex">
        {activeTab === "editor" && (
          <div className="flex-1 flex flex-col">
            {/* Prompt Input */}
            <div className="p-4 border-b">
              <div className="flex space-x-2">
                <div className="flex-1">
                  <Textarea
                    placeholder="Describe the code you want to generate... (Ctrl+Enter to generate)"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="min-h-[80px] resize-none"
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <Button
                    onClick={generateCode}
                    disabled={isGenerating || !prompt.trim()}
                    className="px-6"
                  >
                    {isGenerating ? (
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                        Generating...
                      </div>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Generate
                      </>
                    )}
                  </Button>
                  {generatedCode && (
                    <Button variant="outline" onClick={saveCode}>
                      <Save className="h-4 w-4 mr-2" />
                      Save
                    </Button>
                  )}
                </div>
              </div>
            </div>

            {/* Code Editor */}
            <div className="flex-1 p-4">
              <Card className="h-full">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">
                    Generated Code
                    {lastGeneration && (
                      <Badge variant="outline" className="ml-2">
                        {lastGeneration.metadata.generation_time_ms}ms
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0 h-[calc(100%-60px)]">
                  <MonacoEditor
                    ref={editorRef}
                    value={generatedCode}
                    language="python"
                    theme="vs-dark"
                    options={{
                      readOnly: false,
                      minimap: { enabled: true },
                      fontSize: 14,
                      lineNumbers: "on",
                      roundedSelection: false,
                      scrollBeyondLastLine: false,
                      automaticLayout: true,
                    }}
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {activeTab === "stats" && (
          <div className="flex-1 p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Total Generations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {stats?.total_generations || 0}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">App Version</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-sm">{stats?.app_version || "0.1.0"}</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Backend Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <Badge variant={stats?.backend_status === "mock" ? "secondary" : "default"}>
                    {stats?.backend_status || "Unknown"}
                  </Badge>
                </CardContent>
              </Card>

              {lastGeneration && (
                <Card className="md:col-span-2 lg:col-span-3">
                  <CardHeader>
                    <CardTitle className="text-sm">Last Generation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div><strong>ID:</strong> {lastGeneration.id}</div>
                      <div><strong>Model:</strong> {lastGeneration.model}</div>
                      <div><strong>Tokens:</strong> {lastGeneration.metadata.tokens_generated}</div>
                      <div><strong>Time:</strong> {lastGeneration.metadata.generation_time_ms}ms</div>
                      {lastGeneration.metadata.consensus_score && (
                        <div><strong>Consensus Score:</strong> {lastGeneration.metadata.consensus_score}</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}

        {activeTab === "settings" && (
          <div className="flex-1 p-4">
            <Card>
              <CardHeader>
                <CardTitle>Settings</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">vLLM Backend</label>
                    <p className="text-sm text-muted-foreground">
                      Configure connection to vLLM backend in WSL2
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Model Selection</label>
                    <p className="text-sm text-muted-foreground">
                      Choose between Mixtral, DeepSeek, or Ensemble
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium">CodeBLEU Consensus</label>
                    <p className="text-sm text-muted-foreground">
                      Enable advanced code quality validation
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 