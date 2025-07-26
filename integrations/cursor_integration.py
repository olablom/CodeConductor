import pyperclip
import re
import sys


class CursorIntegration:
    """
    MVP for Cursor Integration Layer:
    - Copies prompt to clipboard
    - Notifies user
    - Accepts pasted Cursor response
    - Extracts code blocks from response
    """

    def __init__(self):
        pass

    def copy_prompt_to_clipboard(self, prompt: str):
        pyperclip.copy(prompt)
        print(
            "\n[INFO] Prompt copied to clipboard! Paste it into Cursor and run the generation.\n"
        )

    def get_cursor_response(self) -> str:
        print(
            "\n[INPUT] Paste Cursor's response below. End with a line containing only 'END'.\n"
        )
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)

    def extract_code_blocks(self, cursor_response: str):
        """
        Extract code blocks of the form:
        ```filename.py
        ...code...
        ```
        Returns: List of (filename, code) tuples
        """
        pattern = r"```([\w\-/\\.]+)\n(.*?)```"
        matches = re.findall(pattern, cursor_response, re.DOTALL)
        code_blocks = []
        for filename, code in matches:
            code_blocks.append((filename.strip(), code.strip()))
        return code_blocks

    def show_code_blocks(self, code_blocks):
        if not code_blocks:
            print("[WARN] No code blocks found in Cursor response.")
            return
        print("\n[RESULT] Extracted code blocks:")
        for filename, code in code_blocks:
            print(f"\n--- {filename} ---\n{code}\n")
