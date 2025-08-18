import React, { forwardRef, useEffect, useRef } from "react";
import * as monaco from "monaco-editor";

interface MonacoEditorProps {
  value: string;
  language?: string;
  theme?: string;
  options?: monaco.editor.IStandaloneEditorConstructionOptions;
  onChange?: (value: string) => void;
}

const MonacoEditor = forwardRef<monaco.editor.IStandaloneCodeEditor, MonacoEditorProps>(
  ({ value, language = "python", theme = "vs-dark", options = {}, onChange }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);

    useEffect(() => {
      if (containerRef.current && !editorRef.current) {
        // Create editor
        editorRef.current = monaco.editor.create(containerRef.current, {
          value,
          language,
          theme,
          automaticLayout: true,
          minimap: { enabled: true },
          fontSize: 14,
          lineNumbers: "on",
          roundedSelection: false,
          scrollBeyondLastLine: false,
          ...options,
        });

        // Set up change listener
        editorRef.current.onDidChangeModelContent(() => {
          const currentValue = editorRef.current?.getValue() || "";
          onChange?.(currentValue);
        });

        // Expose editor through ref
        if (ref) {
          if (typeof ref === "function") {
            ref(editorRef.current);
          } else {
            ref.current = editorRef.current;
          }
        }
      }

      return () => {
        if (editorRef.current) {
          editorRef.current.dispose();
          editorRef.current = null;
        }
      };
    }, []);

    // Update value when prop changes
    useEffect(() => {
      if (editorRef.current && value !== editorRef.current.getValue()) {
        editorRef.current.setValue(value);
      }
    }, [value]);

    // Update language when prop changes
    useEffect(() => {
      if (editorRef.current) {
        monaco.editor.setModelLanguage(editorRef.current.getModel()!, language);
      }
    }, [language]);

    // Update theme when prop changes
    useEffect(() => {
      if (editorRef.current) {
        monaco.editor.setTheme(theme);
      }
    }, [theme]);

    return <div ref={containerRef} className="w-full h-full" />;
  }
);

MonacoEditor.displayName = "MonacoEditor";

export default MonacoEditor;
