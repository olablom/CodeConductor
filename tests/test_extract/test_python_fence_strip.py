from codeconductor.utils.extract import extract_code, normalize_python


def test_python_fence_strip_and_compile():
    text = """
Here is the implementation:

```python
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```
"""

    raw = extract_code(text, lang_hint="python")
    code = normalize_python(raw)
    # Should compile cleanly
    compile(code, "<test>", "exec")
