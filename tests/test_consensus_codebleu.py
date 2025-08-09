import pytest

from codeconductor.ensemble.consensus_calculator import ConsensusCalculator


def make_resp(model: str, content: str, confidence: float = 0.5):
    return {"model": model, "content": content, "confidence": confidence}


def test_similarity_identical_snippets_high():
    cc = ConsensusCalculator()
    a = """
```python
def add(a: int, b: int) -> int:
    return a + b
```
"""
    b = """
```python
def add(a: int, b: int) -> int:
    return a + b
```
"""
    # private call via the public API path
    resps = [
        {"response": make_resp("m1", a)},
        {"response": make_resp("m2", b)},
    ]
    score = cc._calculate_consistency(resps)
    assert score > 0.95


def test_similarity_different_snippets_low():
    cc = ConsensusCalculator()
    a = """
```python
def add(a: int, b: int) -> int:
    return a + b
```
"""
    b = """
```python
def multiply(a: int, b: int) -> int:
    return a * b
```
"""
    resps = [
        {"response": make_resp("m1", a)},
        {"response": make_resp("m2", b)},
    ]
    score = cc._calculate_consistency(resps)
    assert score < 0.7


def test_consensus_returns_reasonable_fields():
    cc = ConsensusCalculator()
    content = """
```python
def add(a: int, b: int) -> int:
    # Add two numbers
    return a + b
```
"""
    result = cc.calculate_consensus(
        [
            make_resp("m1", content, 0.7),
            make_resp("m2", content, 0.8),
        ]
    )
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert result.syntax_valid is True
    assert result.code_quality_score >= 0.5
