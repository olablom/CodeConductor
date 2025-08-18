#!/usr/bin/env python3
"""
Improved Prompt Generator with Self-Reflection and Test-Driven Development
"""


class ImprovedPromptGenerator:
    """Enhanced prompt generator with self-reflection and test validation"""

    def __init__(self):
        self.test_cases = {
            "fibonacci": [
                "assert fibonacci(1) == 1",
                "assert fibonacci(2) == 1",
                "assert fibonacci(5) == 5",
                "assert fibonacci(10) == 55",
                "assert fibonacci(0) == 0 or raises ValueError",
            ],
            "binary_search": [
                "arr = [1, 3, 5, 7, 9, 11, 13, 15]",
                "assert binary_search(arr, 7) == 3",
                "assert binary_search(arr, 1) == 0",
                "assert binary_search(arr, 15) == 7",
                "assert binary_search(arr, 10) == -1",
            ],
            "rest_api": [
                "app = create_app()",
                "client = app.test_client()",
                "response = client.post('/login', json={'username': 'user1', 'password': 'password1'})",
                "assert response.status_code == 200",
                "assert 'token' in response.json",
            ],
        }

    def generate_improved_prompt(self, task_type: str, description: str) -> str:
        """Generate improved prompt with self-reflection and test cases"""

        test_cases = self.test_cases.get(task_type, [])

        prompt = f"""### System
You are Coder – an AI development expert who writes production-grade code that MUST pass all tests.

CRITICAL REQUIREMENTS:
1. Write code that passes ALL test cases below
2. After writing, mentally RUN each test and FIX any failures
3. Use iterative approach if recursion risks stack-overflow
4. Handle edge cases (empty arrays, invalid inputs, etc.)
5. Return proper values (not None, not -1 unless specified)

### Test Cases
{chr(10).join(f"# {i+1}. {test}" for i, test in enumerate(test_cases))}

### Self-Reflection Steps:
1. Write the function
2. Mentally run each test case
3. If any test fails, FIX the code
4. Ensure all edge cases are handled
5. Verify return values are correct

### User
{description}

### Coder:
Let me write production-grade code that passes all tests:

```python
"""

        return prompt

    def generate_fix_prompt(
        self, original_code: str, error_message: str, task_type: str
    ) -> str:
        """Generate prompt to fix failed code"""

        test_cases = self.test_cases.get(task_type, [])

        prompt = f"""### System
You are Coder – an AI development expert who FIXES code that failed tests.

### Previous Attempt Failed
Error: {error_message}

### Original Code
```python
{original_code}
```

### Test Cases (Must Pass)
{chr(10).join(f"# {i+1}. {test}" for i, test in enumerate(test_cases))}

### Fix Instructions:
1. Analyze the error message
2. Identify the root cause
3. Fix the code to pass ALL tests
4. Use iterative approach if recursion fails
5. Handle edge cases properly

### User
Please fix the code to pass all tests.

### Coder:
Let me fix the code to pass all tests:

```python
"""

        return prompt


# Global instance
improved_prompt_generator = ImprovedPromptGenerator()
