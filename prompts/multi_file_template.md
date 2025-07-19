# Multi-File Project Generation Template

## Project Structure

Generate a complete project with the following structure:

```
project_name/
├── main.py              # Main application entry point
├── utils.py             # Utility functions and helpers
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── tests/
│   ├── __init__.py
│   ├── test_main.py    # Main application tests
│   └── test_utils.py   # Utility function tests
└── .gitignore          # Git ignore file
```

## Generation Rules

1. **File Dependencies**: Ensure proper imports between files
2. **Testing**: Include comprehensive tests for each module
3. **Documentation**: Add docstrings and README
4. **Best Practices**: Follow Python conventions and security practices
5. **Error Handling**: Include proper exception handling
6. **Configuration**: Use environment variables where appropriate

## Output Format

For each file, provide:

- **Filename**: The complete file path
- **Content**: The full file content with proper syntax
- **Purpose**: Brief description of the file's role

## Example Structure

````
# File: main.py
```python
# Main application logic here
````

# File: utils.py

```python
# Utility functions here
```

# File: tests/test_main.py

```python
# Test cases here
```

```

## Quality Requirements

- All files must be syntactically correct Python
- Tests must be runnable with pytest
- Code must pass basic linting (pylint score > 7.0)
- Security best practices must be followed
- Documentation must be clear and complete
```
