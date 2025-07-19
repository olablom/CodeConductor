"""
Cursor API Integration - Core to Gabriel's Vision

Connects multi-agent consensus to Cursor IDE for actual code generation.
This is the bridge between our local reasoning and cloud implementation.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time


class CursorAPI:
    """Integration med Cursor IDE för kodgenerering"""

    def __init__(self, cursor_path: str = "cursor"):
        self.cursor_path = cursor_path
        self.temp_dir = Path("temp_cursor")
        self.temp_dir.mkdir(exist_ok=True)

    def is_available(self) -> bool:
        """Check if Cursor CLI is installed and accessible"""
        try:
            result = subprocess.run(
                [self.cursor_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def generate_code(
        self, prompt: str, output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Send optimized prompt to Cursor and get code back

        Args:
            prompt: Optimized prompt from multi-agent consensus
            output_path: Where to save generated code (optional)

        Returns:
            Dict with success status, code, and metadata
        """

        if not self.is_available():
            return {
                "success": False,
                "code": None,
                "message": "Cursor CLI not available",
                "fallback": True,
            }

        # Create unique prompt file
        timestamp = int(time.time())
        prompt_file = self.temp_dir / f"prompt_{timestamp}.md"

        try:
            # Write optimized prompt to file
            prompt_file.write_text(prompt)

            # Determine output path
            if output_path is None:
                output_path = self.temp_dir / f"generated_{timestamp}.py"

            # Call Cursor CLI with optimized prompt
            result = subprocess.run(
                [
                    self.cursor_path,
                    "generate",
                    "--prompt",
                    str(prompt_file),
                    "--output",
                    str(output_path),
                    "--model",
                    "gpt-4",
                    "--temperature",
                    "0.1",  # Low temperature for consistent code
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes timeout
            )

            if result.returncode == 0:
                # Read generated code
                if output_path.exists():
                    code = output_path.read_text()

                    return {
                        "success": True,
                        "code": code,
                        "prompt_file": str(prompt_file),
                        "output_file": str(output_path),
                        "message": "Code generated successfully by Cursor",
                        "metadata": {
                            "lines_of_code": len(code.splitlines()),
                            "cursor_version": self._get_cursor_version(),
                            "generation_time": time.time() - timestamp,
                        },
                    }
                else:
                    return {
                        "success": False,
                        "code": None,
                        "message": "Cursor succeeded but no output file found",
                        "stderr": result.stderr,
                    }
            else:
                return {
                    "success": False,
                    "code": None,
                    "message": f"Cursor generation failed: {result.stderr}",
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "code": None,
                "message": "Cursor generation timed out (120s)",
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "code": None,
                "message": f"Unexpected error: {str(e)}",
                "error": str(e),
            }
        finally:
            # Cleanup prompt file
            if prompt_file.exists():
                prompt_file.unlink()

    def _get_cursor_version(self) -> str:
        """Get Cursor CLI version"""
        try:
            result = subprocess.run(
                [self.cursor_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about code generation"""
        if not self.temp_dir.exists():
            return {"total_generations": 0, "success_rate": 0.0}

        files = list(self.temp_dir.glob("generated_*.py"))
        return {
            "total_generations": len(files),
            "temp_dir": str(self.temp_dir),
            "cursor_available": self.is_available(),
        }


class MockCursorAPI:
    """Mock Cursor API for testing and demo purposes"""

    def __init__(self):
        self.generation_count = 0

    def is_available(self) -> bool:
        return True

    def generate_code(
        self, prompt: str, output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate mock code based on prompt"""
        self.generation_count += 1

        # Extract project type from prompt
        project_type = "API"
        if "web app" in prompt.lower() or "frontend" in prompt.lower():
            project_type = "WebApp"
        elif "cli" in prompt.lower() or "command line" in prompt.lower():
            project_type = "CLI"
        elif "ml" in prompt.lower() or "machine learning" in prompt.lower():
            project_type = "ML"

        # Generate simple mock code
        mock_code = self._generate_simple_code(project_type)

        # Save to file if output_path provided
        if output_path:
            output_path.write_text(mock_code)

        return {
            "success": True,
            "code": mock_code,
            "message": f"Mock {project_type} code generated successfully",
            "metadata": {
                "lines_of_code": len(mock_code.splitlines()),
                "cursor_version": "mock-1.0.0",
                "generation_time": 2.5,
                "project_type": project_type,
            },
        }

    def _generate_simple_code(self, project_type: str) -> str:
        """Generate simple mock code"""
        if project_type == "API":
            return """# Generated by CodeConductor v2.0 (Mock Cursor)
# Project: REST API

import fastapi
from fastapi import FastAPI

app = FastAPI(title="Generated API")

@app.get("/")
def hello():
    return {"message": "Hello from CodeConductor!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""

        elif project_type == "WebApp":
            return """# Generated by CodeConductor v2.0 (Mock Cursor)
# Project: Web Application

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from CodeConductor WebApp!"

if __name__ == '__main__':
    app.run(debug=True)"""

        elif project_type == "CLI":
            return '''# Generated by CodeConductor v2.0 (Mock Cursor)
# Project: CLI Tool

import click

@click.command()
def cli():
    """Generated CLI Tool"""
    click.echo("Hello from CodeConductor CLI!")

if __name__ == '__main__':
    cli()'''

        elif project_type == "ML":
            return """# Generated by CodeConductor v2.0 (Mock Cursor)
# Project: Machine Learning Model

import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        
    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained successfully!")
        
    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    print("ML Model ready!")"""

        else:
            return """# Generated by CodeConductor v2.0 (Mock Cursor)
# Project: Generic Python Application

def main():
    print("Hello from CodeConductor!")
    print("Your application is ready!")

if __name__ == '__main__':
    main()"""

    def cleanup_temp_files(self):
        """Mock cleanup"""
        pass

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get mock generation stats"""
        return {
            "total_generations": self.generation_count,
            "success_rate": 1.0,
            "mock_mode": True,
        }
