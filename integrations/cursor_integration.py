#!/usr/bin/env python3
"""
Cursor Integration for CodeConductor MVP.

Provides clipboard management and code extraction for manual Cursor integration.
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    pyperclip = None
    CLIPBOARD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedFile:
    """Represents an extracted file from Cursor output."""

    path: Path
    content: str
    language: str = "python"


class ClipboardManager:
    """Manages clipboard operations for Cursor integration."""

    def __init__(self):
        self.available = CLIPBOARD_AVAILABLE
        if not self.available:
            logger.warning("pyperclip not available. Clipboard operations will fail.")

    def copy_to_clipboard(self, text: str) -> bool:
        """
        Copy text to clipboard.

        Args:
            text: Text to copy to clipboard

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            logger.error("Clipboard not available - pyperclip not installed")
            return False

        try:
            pyperclip.copy(text)
            logger.info(f"Copied {len(text)} characters to clipboard")
            return True
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False

    def read_from_clipboard(self) -> str:
        """
        Read text from clipboard.

        Returns:
            Clipboard content as string, empty string if failed
        """
        if not self.available:
            logger.error("Clipboard not available - pyperclip not installed")
            return ""

        try:
            content = pyperclip.paste()
            logger.info(f"Read {len(content)} characters from clipboard")
            return content
        except Exception as e:
            logger.error(f"Failed to read from clipboard: {e}")
            return ""


class CodeExtractor:
    """Extracts code files from Cursor output."""

    def __init__(self):
        # Regex to match code blocks with optional filename
        self.code_block_pattern = re.compile(
            r"```(\w+)?\s*\n(?:#\s*([^\n]+)\s*\n)?(.*?)```", re.DOTALL
        )
        self.file_counter = 0

    def extract_cursor_code(self, cursor_output: str) -> List[Tuple[Path, str]]:
        """
        Extract code files from Cursor output.

        Args:
            cursor_output: Raw output from Cursor

        Returns:
            List of (file_path, code_content) tuples
        """
        if not cursor_output:
            logger.warning("Empty cursor output provided")
            return []

        extracted_files = []

        # Find all code blocks
        code_blocks = re.findall(r"```(\w+)?\s*\n(.*?)```", cursor_output, re.DOTALL)

        for language, content in code_blocks:
            # Determine language (default to python)
            language = language.lower() if language else "python"

            # Split content into lines to look for filename comment
            lines = content.strip().split("\n")
            filename_comment = ""
            code_lines = []

            # Look for filename in first line if it starts with #
            if lines and lines[0].strip().startswith("#"):
                filename_comment = lines[0].strip()
                code_lines = lines[1:]
            else:
                code_lines = lines

            # Extract filename from comment or generate default
            filename = self._extract_filename(filename_comment, language)

            # Clean up code content
            code_content = self._clean_code_content("\n".join(code_lines))

            if code_content.strip():
                file_path = Path(filename)
                extracted_files.append((file_path, code_content))
                logger.info(f"Extracted {filename} ({len(code_content)} chars)")

        logger.info(f"Extracted {len(extracted_files)} files from Cursor output")
        return extracted_files

    def _extract_filename(self, filename_comment: str, language: str) -> str:
        """
        Extract filename from comment or generate default.

        Args:
            filename_comment: Comment line that might contain filename
            language: Programming language for default extension

        Returns:
            Filename string
        """
        if filename_comment:
            # Remove common comment prefixes and clean up
            filename = filename_comment.strip()
            filename = re.sub(r"^#\s*", "", filename)  # Remove # prefix
            filename = re.sub(r"^\s*//\s*", "", filename)  # Remove // prefix
            filename = filename.strip()

            # If it looks like a filename, use it
            if "." in filename or "/" in filename or "\\" in filename:
                return filename

        # Generate default filename
        self.file_counter += 1
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
            "go": ".go",
            "rust": ".rs",
            "php": ".php",
            "ruby": ".rb",
        }
        ext = extensions.get(language, ".txt")
        return f"generated_{self.file_counter}{ext}"

    def _clean_code_content(self, code_content: str) -> str:
        """
        Clean up code content by removing extra whitespace.

        Args:
            code_content: Raw code content

        Returns:
            Cleaned code content
        """
        # Remove leading/trailing whitespace
        code_content = code_content.strip()

        # Normalize line endings
        code_content = code_content.replace("\r\n", "\n")

        return code_content


class CursorIntegration:
    """Main integration class for Cursor workflow."""

    def __init__(self):
        self.clipboard_manager = ClipboardManager()
        self.code_extractor = CodeExtractor()

    def copy_prompt_to_clipboard(self, prompt: str) -> bool:
        """
        Copy prompt to clipboard for Cursor.

        Args:
            prompt: Prompt to copy

        Returns:
            True if successful
        """
        logger.info("Copying prompt to clipboard for Cursor...")
        return self.clipboard_manager.copy_to_clipboard(prompt)

    def wait_for_cursor_output(self) -> str:
        """
        Wait for user to paste Cursor output.

        Returns:
            Cursor output as string
        """
        print("\n" + "=" * 60)
        print("📋 PROMPT COPIED TO CLIPBOARD!")
        print("=" * 60)
        print("1. Paste the prompt into Cursor")
        print("2. Generate the code")
        print("3. Copy the generated code output")
        print("4. Press Enter when ready to continue...")
        print("=" * 60)

        input("Press Enter when you have Cursor output ready...")

        logger.info("Reading Cursor output from clipboard...")
        return self.clipboard_manager.read_from_clipboard()

    def extract_and_save_files(
        self, cursor_output: str, output_dir: Path = None
    ) -> List[Path]:
        """
        Extract code files from Cursor output and save them.

        Args:
            cursor_output: Raw output from Cursor
            output_dir: Directory to save files (default: current directory)

        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = Path.cwd()

        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_files = self.code_extractor.extract_cursor_code(cursor_output)
        saved_files = []

        for file_path, code_content in extracted_files:
            # Create full path
            full_path = output_dir / file_path

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(code_content)

                saved_files.append(full_path)
                logger.info(f"Saved {full_path}")

            except Exception as e:
                logger.error(f"Failed to save {full_path}: {e}")

        logger.info(f"Saved {len(saved_files)} files to {output_dir}")
        return saved_files

    def run_cursor_workflow(self, prompt: str, output_dir: Path = None) -> List[Path]:
        """
        Run complete Cursor workflow.

        Args:
            prompt: Prompt to send to Cursor
            output_dir: Directory to save generated files

        Returns:
            List of generated file paths
        """
        logger.info("Starting Cursor workflow...")

        # Step 1: Copy prompt to clipboard
        if not self.copy_prompt_to_clipboard(prompt):
            logger.error("Failed to copy prompt to clipboard")
            return []

        # Step 2: Wait for user to generate code in Cursor
        cursor_output = self.wait_for_cursor_output()

        if not cursor_output.strip():
            logger.error("No Cursor output received")
            return []

        # Step 3: Extract and save files
        saved_files = self.extract_and_save_files(cursor_output, output_dir)

        logger.info(f"Cursor workflow completed. Generated {len(saved_files)} files.")
        return saved_files


def main():
    """Demo function for Cursor integration."""
    integration = CursorIntegration()

    # Example prompt
    prompt = """
    Create a simple Python calculator class with the following requirements:
    
    1. Support basic arithmetic operations (add, subtract, multiply, divide)
    2. Handle division by zero errors
    3. Include type hints
    4. Add comprehensive unit tests using pytest
    
    Please provide the implementation and test files.
    """

    print("🚀 CodeConductor Cursor Integration Demo")
    print("=" * 50)

    # Run workflow
    generated_files = integration.run_cursor_workflow(prompt)

    if generated_files:
        print(f"\n✅ Successfully generated {len(generated_files)} files:")
        for file_path in generated_files:
            print(f"   - {file_path}")
    else:
        print("\n❌ No files were generated")


if __name__ == "__main__":
    main()
