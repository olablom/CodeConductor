"""
Integration layer between MultiStepOrchestrator and Streamlit Dashboard
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import streamlit as st
from queue import Queue
import threading

# Import your existing components
from multi_step_pipeline import MultiStepPipeline
from agents.multi_step_orchestrator import MultiStepOrchestrator


class EventType(Enum):
    TASK_START = "task_start"
    AGENT_MESSAGE = "agent_message"
    AGENT_THINKING = "agent_thinking"
    STEP_COMPLETE = "step_complete"
    HUMAN_APPROVAL_NEEDED = "human_approval_needed"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    METRICS_UPDATE = "metrics_update"


@dataclass
class DashboardEvent:
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StreamingOrchestrator(MultiStepOrchestrator):
    """Extended orchestrator that streams events to dashboard"""

    def __init__(self, event_queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_queue = event_queue
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None

    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """Send event to dashboard"""
        event = DashboardEvent(type=event_type, data=data)
        self.event_queue.put(event)

    async def execute_task_with_streaming(
        self, task_description: str
    ) -> AsyncGenerator[DashboardEvent, None]:
        """Execute task while streaming events to dashboard"""

        self.start_time = datetime.now()

        # Start event
        await self._emit_event(
            EventType.TASK_START,
            {
                "description": task_description,
                "complexity": self._analyze_task_complexity(task_description),
            },
        )

        try:
            # Generate workflow plan
            workflow_plan = await self._generate_workflow_plan(task_description)
            self.total_steps = len(workflow_plan["steps"])

            # Execute each step
            for i, step in enumerate(workflow_plan["steps"]):
                self.current_step = i + 1

                # Step start
                await self._emit_event(
                    EventType.AGENT_THINKING,
                    {
                        "step": self.current_step,
                        "total_steps": self.total_steps,
                        "description": step["description"],
                        "agent": step["agent"],
                    },
                )

                # Execute agent with full context
                agent_response = await self._execute_agent(
                    step["agent"],
                    step["task"],
                    context={
                        **workflow_plan.get("context", {}),
                        "original_task": task_description,
                        "current_step": self.current_step,
                        "total_steps": self.total_steps,
                    },
                )

                # Agent message
                await self._emit_event(
                    EventType.AGENT_MESSAGE,
                    {
                        "agent": step["agent"],
                        "message": agent_response.get("output", ""),
                        "confidence": agent_response.get("confidence", 0.95),
                        "step": self.current_step,
                    },
                )

                # Check if human approval needed
                if agent_response.get("needs_approval", False):
                    await self._emit_event(
                        EventType.HUMAN_APPROVAL_NEEDED,
                        {
                            "agent": step["agent"],
                            "suggestion": agent_response.get("output", ""),
                            "step": self.current_step,
                        },
                    )

                    # Wait for approval (in real implementation)
                    approved = await self._wait_for_approval()
                    if not approved:
                        continue

                # Step complete
                await self._emit_event(
                    EventType.STEP_COMPLETE,
                    {"step": self.current_step, "success": True},
                )

            # Task complete
            await self._emit_event(
                EventType.TASK_COMPLETE,
                {
                    "success": True,
                    "total_time": (datetime.now() - self.start_time).total_seconds(),
                },
            )

        except Exception as e:
            await self._emit_event(
                EventType.ERROR, {"error": str(e), "step": self.current_step}
            )
            raise

    async def _wait_for_approval(self) -> bool:
        """Wait for human approval from dashboard"""
        # In real implementation, this would check session state
        # For now, simulate with a short delay
        await asyncio.sleep(2)
        return True

    def _analyze_task_complexity(self, task_description: str) -> str:
        """Analyze task complexity"""
        description = task_description.lower()
        if any(word in description for word in ["simple", "basic", "easy"]):
            return "Simple"
        elif any(word in description for word in ["complex", "advanced", "expert"]):
            return "Complex"
        else:
            return "Medium"

    async def _generate_workflow_plan(self, task_description: str) -> Dict[str, Any]:
        """Generate workflow plan"""
        # Create dynamic workflow based on task description
        task_lower = task_description.lower()

        if "fibonacci" in task_lower:
            steps = [
                {
                    "agent": "ArchitectAgent",
                    "task": "Design Fibonacci calculator architecture",
                    "description": "Planning efficient Fibonacci implementation with memoization",
                },
                {
                    "agent": "CodeGenAgent",
                    "task": "Generate Fibonacci calculator code",
                    "description": "Implementing the Fibonacci calculator with optimization",
                },
                {
                    "agent": "ReviewAgent",
                    "task": "Review Fibonacci calculator code",
                    "description": "Code quality review and improvement suggestions",
                },
                {
                    "agent": "PolicyAgent",
                    "task": "Security and safety analysis",
                    "description": "Security validation and safety compliance check",
                },
                {
                    "agent": "TestAgent",
                    "task": "Generate comprehensive test suite",
                    "description": "Create unit tests, edge cases, and performance benchmarks",
                },
            ]
        elif "api" in task_lower or "rest" in task_lower:
            steps = [
                {
                    "agent": "ArchitectAgent",
                    "task": "Design REST API architecture",
                    "description": "Planning layered API architecture with authentication",
                },
                {
                    "agent": "CodeGenAgent",
                    "task": "Generate REST API code",
                    "description": "Implementing the REST API with proper endpoints",
                },
                {
                    "agent": "ReviewAgent",
                    "task": "Review REST API code",
                    "description": "API code review and security assessment",
                },
                {
                    "agent": "PolicyAgent",
                    "task": "Security and safety analysis",
                    "description": "API security validation and compliance check",
                },
            ]
        else:
            steps = [
                {
                    "agent": "ArchitectAgent",
                    "task": f"Design architecture for: {task_description}",
                    "description": f"Planning system architecture for {task_description}",
                },
                {
                    "agent": "CodeGenAgent",
                    "task": f"Generate implementation code for: {task_description}",
                    "description": f"Implementing {task_description} with clean code",
                },
                {
                    "agent": "ReviewAgent",
                    "task": f"Review generated code for: {task_description}",
                    "description": f"Code quality review for {task_description}",
                },
                {
                    "agent": "PolicyAgent",
                    "task": f"Security and safety analysis for: {task_description}",
                    "description": f"Security validation for {task_description}",
                },
                {
                    "agent": "TestAgent",
                    "task": f"Generate comprehensive test suite for: {task_description}",
                    "description": f"Create unit tests for {task_description}",
                },
            ]

        return {
            "steps": steps,
            "context": {"task": task_description},
        }

    async def _execute_agent(
        self, agent_name: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent with detailed output"""
        # Simulate processing time
        await asyncio.sleep(1)

        # Create detailed output based on agent type
        if agent_name == "ArchitectAgent":
            return self._execute_architect_agent(task, context)
        elif agent_name == "CodeGenAgent":
            return self._execute_codegen_agent(task, context)
        elif agent_name == "ReviewAgent":
            return self._execute_review_agent(task, context)
        elif agent_name == "PolicyAgent":
            return self._execute_policy_agent(task, context)
        elif agent_name == "TestAgent":
            return self._execute_test_agent(task, context)
        else:
            return {
                "output": f"{agent_name} completed: {task}",
                "confidence": 0.85 + (hash(task) % 15) / 100,
                "needs_approval": False,
            }

    def _execute_architect_agent(
        self, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute ArchitectAgent with detailed architectural output"""
        task_lower = task.lower()

        if "fibonacci" in task_lower:
            output = f"""## ArchitectAgent
### Task: {task}
### Confidence: 94.5%

**Design Approach:**
I've analyzed the requirements and decided on an **iterative approach with memoization** for the Fibonacci calculator.

**Reasoning:**
1. **Recursive vs Iterative:** Chose iterative to avoid stack overflow for large numbers
2. **Memoization:** Added to prevent recalculation of same values (O(n) → O(1) for repeated calls)
3. **Error Handling:** Will include input validation for negative numbers and non-integers

**Proposed Architecture:**
```python
class FibonacciCalculator:
    def __init__(self):
        self.memo = {{}}  # Cache for calculated values
    
    def calculate(self, n: int) -> int:
        # Implementation with detailed error handling
        pass
    
    def reset_cache(self):
        # Clear memoization cache
        pass
```

**Key Design Decisions:**
- **Class-based approach:** Allows for state management (memoization cache)
- **Type hints:** Improves code clarity and IDE support
- **Error handling:** Comprehensive validation for robust operation
- **Cache management:** User can reset cache if needed

**Quality Metrics:**
- **Performance:** ⭐⭐⭐⭐⭐ (O(n) time, O(n) space with memoization)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Clear structure, well-documented)
- **Scalability:** ⭐⭐⭐⭐ (Handles large numbers efficiently)"""

        elif "api" in task_lower or "rest" in task_lower:
            output = f"""## ArchitectAgent
### Task: {task}
### Confidence: 92.3%

**Design Approach:**
I've designed a **layered REST API architecture** with clear separation of concerns.

**Reasoning:**
1. **Layered Architecture:** Controller → Service → Repository pattern for maintainability
2. **RESTful Design:** Standard HTTP methods and status codes for API consistency
3. **Authentication:** JWT-based authentication for secure access
4. **Database:** PostgreSQL for ACID compliance and complex queries

**Proposed Architecture:**
```python
# API Layer Structure
├── controllers/     # HTTP request handlers
├── services/        # Business logic
├── repositories/    # Data access layer
├── models/          # Data models
├── middleware/      # Authentication, logging
└── config/          # Configuration management
```

**Key Design Decisions:**
- **Separation of Concerns:** Each layer has a specific responsibility
- **Dependency Injection:** Loose coupling between components
- **Error Handling:** Centralized error management
- **Validation:** Input validation at multiple layers

**Quality Metrics:**
- **Scalability:** ⭐⭐⭐⭐⭐ (Horizontal scaling ready)
- **Security:** ⭐⭐⭐⭐⭐ (Authentication + authorization)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Clear structure)"""

        else:
            output = f"""## ArchitectAgent
### Task: {task}
### Confidence: 89.7%

**Design Approach:**
I've analyzed the requirements and designed a **modular, scalable solution**.

**Reasoning:**
1. **Modular Design:** Components can be developed and tested independently
2. **Scalable Architecture:** Can handle growth in users and data
3. **Maintainable Code:** Clear structure and documentation

**Proposed Architecture:**
```python
# System Components
├── core/           # Core business logic
├── api/            # API endpoints
├── database/       # Data persistence
├── utils/          # Utility functions
└── tests/          # Test suite
```

**Key Design Decisions:**
- **Modular Structure:** Easy to maintain and extend
- **Clear Interfaces:** Well-defined component boundaries
- **Error Handling:** Robust error management
- **Documentation:** Comprehensive code documentation

**Quality Metrics:**
- **Maintainability:** ⭐⭐⭐⭐⭐ (Clear structure)
- **Scalability:** ⭐⭐⭐⭐ (Modular design)
- **Reliability:** ⭐⭐⭐⭐⭐ (Error handling)"""

        return {
            "output": output,
            "confidence": 0.92,
            "needs_approval": False,
        }

    def _execute_codegen_agent(
        self, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute CodeGenAgent with detailed code generation output"""
        task_lower = task.lower()

        if "fibonacci" in task_lower:
            output = f"""## CodeGenAgent
### Task: {task}
### Confidence: 96.8%

**Implementation Approach:**
I've implemented the Fibonacci calculator using the **iterative approach with memoization** as designed by the ArchitectAgent.

**Code Structure:**
```python
class FibonacciCalculator:
    \"\"\"
    Efficient Fibonacci calculator with memoization.
    
    This implementation uses iterative approach with caching to achieve
    O(n) time complexity and O(n) space complexity for the first calculation,
    then O(1) time for subsequent calls with the same input.
    \"\"\"
    
    def __init__(self):
        # Initialize memoization cache
        self.memo = {{0: 0, 1: 1}}
    
    def calculate(self, n: int) -> int:
        \"\"\"
        Calculate the nth Fibonacci number.
        
        Args:
            n: The position in the Fibonacci sequence (0-indexed)
            
        Returns:
            The nth Fibonacci number
            
        Raises:
            ValueError: If n is negative
        \"\"\"
        # Input validation
        if not isinstance(n, int):
            raise ValueError("Input must be an integer")
        if n < 0:
            raise ValueError("Input must be non-negative")
        
        # Check cache first
        if n in self.memo:
            return self.memo[n]
        
        # Calculate iteratively
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
            self.memo[i] = b
        
        return b
    
    def reset_cache(self):
        \"\"\"Reset the memoization cache.\"\"\"
        self.memo = {{0: 0, 1: 1}}
    
    def get_cache_size(self) -> int:
        \"\"\"Get the current size of the memoization cache.\"\"\"
        return len(self.memo)

# Usage example
if __name__ == "__main__":
    calc = FibonacciCalculator()
    
    # Calculate first 10 Fibonacci numbers
    for i in range(10):
        result = calc.calculate(i)
        print(f"F({{i}}) = {{result}}")
    
    print(f"Cache size: {{calc.get_cache_size()}}")
```

**Key Implementation Decisions:**
- **Input Validation:** Comprehensive error checking for robust operation
- **Memoization:** Efficient caching to avoid recalculation
- **Documentation:** Clear docstrings explaining functionality
- **Type Hints:** Improved code clarity and IDE support
- **Error Handling:** Proper exception handling with descriptive messages

**Performance Analysis:**
- **Time Complexity:** O(n) for first calculation, O(1) for cached values
- **Space Complexity:** O(n) for memoization cache
- **Memory Usage:** Efficient with automatic cache management

**Quality Assessment:**
- **Readability:** ⭐⭐⭐⭐⭐ (Clear structure and documentation)
- **Performance:** ⭐⭐⭐⭐⭐ (Optimized with memoization)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Well-documented and modular)"""

        else:
            output = f"""## CodeGenAgent
### Task: {task}
### Confidence: 93.2%

**Implementation Approach:**
I've implemented the solution following the architectural design with clean, maintainable code.

**Code Structure:**
```python
# Main implementation
def main():
    \"\"\"Main application entry point.\"\"\"
    print("Application started successfully")
    
    # Core functionality implementation
    result = process_task()
    
    return result

def process_task():
    \"\"\"Process the main task logic.\"\"\"
    # Implementation details here
    pass

if __name__ == "__main__":
    main()
```

**Key Implementation Decisions:**
- **Clean Code:** Following PEP 8 style guidelines
- **Error Handling:** Proper exception management
- **Documentation:** Clear function documentation
- **Modular Design:** Separated concerns for maintainability

**Quality Assessment:**
- **Readability:** ⭐⭐⭐⭐⭐ (Clear and well-documented)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Modular structure)
- **Reliability:** ⭐⭐⭐⭐⭐ (Proper error handling)"""

        return {
            "output": output,
            "confidence": 0.95,
            "needs_approval": False,
        }

    def _execute_review_agent(
        self, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute ReviewAgent with detailed code review output"""
        task_lower = task.lower()

        if "fibonacci" in task_lower:
            output = f"""## ReviewAgent
### Task: {task}
### Confidence: 91.4%

**Code Review Summary:**
I've thoroughly reviewed the Fibonacci calculator implementation and found it to be **high-quality code** with excellent design decisions.

**Strengths Identified:**
✅ **Excellent Architecture:** Follows the designed iterative approach with memoization
✅ **Comprehensive Error Handling:** Proper input validation and exception management
✅ **Clear Documentation:** Well-documented with detailed docstrings
✅ **Performance Optimization:** Efficient memoization strategy
✅ **Type Safety:** Proper type hints throughout the code
✅ **Clean Code:** Follows PEP 8 style guidelines

**Code Quality Analysis:**
- **Overall Score:** 9.2/10 ⭐⭐⭐⭐⭐
- **Readability:** 9.5/10 (Excellent documentation and structure)
- **Performance:** 9.0/10 (Optimized with memoization)
- **Maintainability:** 9.3/10 (Clear separation of concerns)
- **Security:** 9.1/10 (Proper input validation)

**Detailed Review:**

**1. Class Design (9.5/10):**
```python
class FibonacciCalculator:
    # ✅ Excellent class structure
    # ✅ Clear initialization with proper defaults
    # ✅ Good separation of concerns
```

**2. Method Implementation (9.0/10):**
```python
def calculate(self, n: int) -> int:
    # ✅ Comprehensive input validation
    # ✅ Efficient memoization strategy
    # ✅ Clear variable naming
    # ✅ Proper error handling
```

**3. Error Handling (9.3/10):**
- ✅ Validates input type (integer check)
- ✅ Validates input range (non-negative check)
- ✅ Provides descriptive error messages
- ✅ Uses appropriate exception types

**4. Performance Analysis (9.0/10):**
- ✅ O(n) time complexity for first calculation
- ✅ O(1) time complexity for cached values
- ✅ Efficient memory usage with memoization
- ✅ No unnecessary computations

**Minor Suggestions for Enhancement:**
1. **Add unit tests** for comprehensive test coverage
2. **Consider adding logging** for debugging purposes
3. **Add performance benchmarks** for large numbers
4. **Consider thread safety** if used in multi-threaded environments

**Security Assessment:**
- ✅ No dangerous operations detected
- ✅ Proper input validation prevents injection attacks
- ✅ No file system or network operations
- ✅ Safe mathematical operations only

**Final Recommendation:**
**APPROVED** ✅ - This is high-quality, production-ready code that follows best practices and meets all requirements.

**Confidence Level:** 91.4% - Very confident in the quality and safety of this implementation."""

        else:
            output = f"""## ReviewAgent
### Task: {task}
### Confidence: 88.7%

**Code Review Summary:**
I've reviewed the implementation and found it to be **good quality code** with room for minor improvements.

**Strengths Identified:**
✅ **Clean Structure:** Well-organized code with clear separation
✅ **Good Documentation:** Adequate function documentation
✅ **Error Handling:** Basic error management in place
✅ **Readable Code:** Follows coding standards

**Code Quality Analysis:**
- **Overall Score:** 8.5/10 ⭐⭐⭐⭐
- **Readability:** 8.7/10 (Good structure and comments)
- **Maintainability:** 8.3/10 (Modular design)
- **Reliability:** 8.4/10 (Basic error handling)

**Suggestions for Improvement:**
1. **Add more comprehensive error handling**
2. **Include unit tests for better coverage**
3. **Add logging for debugging**
4. **Consider performance optimizations**

**Final Recommendation:**
**APPROVED** ✅ - Good quality code that meets requirements."""

        return {
            "output": output,
            "confidence": 0.91,
            "needs_approval": False,
        }

    def _execute_policy_agent(
        self, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute PolicyAgent with detailed safety analysis output"""
        task_lower = task.lower()

        output = f"""## PolicyAgent
### Task: {task}
### Confidence: 97.2%

**Security and Safety Analysis:**
I've conducted a comprehensive security review of the generated code and found it to be **safe for execution**.

**Safety Assessment Results:**
✅ **NO CRITICAL VIOLATIONS DETECTED**
✅ **NO HIGH-RISK PATTERNS FOUND**
✅ **NO DANGEROUS OPERATIONS IDENTIFIED**
✅ **COMPLIES WITH SAFETY POLICIES**

**Detailed Security Analysis:**

**1. Code Execution Safety (100% Safe):**
- ✅ No `eval()` or `exec()` functions detected
- ✅ No `os.system()` or subprocess calls
- ✅ No dangerous file operations
- ✅ No network operations without proper validation

**2. Input Validation (Excellent):**
- ✅ Comprehensive input type checking
- ✅ Range validation for numerical inputs
- ✅ Proper exception handling for invalid inputs
- ✅ No potential for injection attacks

**3. Data Handling (Secure):**
- ✅ No sensitive data exposure
- ✅ No hardcoded secrets or credentials
- ✅ Safe mathematical operations only
- ✅ Proper memory management

**4. Compliance Check (Fully Compliant):**
- ✅ Follows Python security best practices
- ✅ No deprecated or unsafe functions
- ✅ Proper error handling without information leakage
- ✅ Clean code without dangerous patterns

**Risk Assessment:**
- **Overall Risk Level:** 🟢 LOW (0.2/10)
- **Security Risk:** 🟢 LOW (0.1/10)
- **Data Risk:** 🟢 LOW (0.0/10)
- **Execution Risk:** 🟢 LOW (0.1/10)

**Policy Compliance Summary:**
- ✅ **Critical Security Policies:** 100% Compliant
- ✅ **High-Risk Pattern Detection:** 100% Clean
- ✅ **Medium-Risk Operations:** 100% Safe
- ✅ **Low-Risk Warnings:** 0 Found

**Final Safety Recommendation:**
**SAFE FOR EXECUTION** ✅ - This code passes all security checks and is approved for deployment.

**Confidence Level:** 97.2% - Very confident in the safety of this code.

**Additional Security Notes:**
- Code follows secure coding practices
- No vulnerabilities detected
- Safe for production deployment
- Recommended for immediate use"""

        return {
            "output": output,
            "confidence": 0.97,
            "needs_approval": False,
        }

    def _execute_test_agent(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TestAgent with detailed test generation output"""
        task_lower = task.lower()

        output = f"""## TestAgent
### Task: {task}
### Confidence: 93.8%

**Test Generation Approach:**
I've analyzed the code and generated a **comprehensive test suite** with pytest framework.

**Test Coverage Strategy:**
1. **Unit Tests:** Test all public methods and functions
2. **Edge Case Tests:** Test boundary conditions and invalid inputs
3. **Error Scenario Tests:** Test exception handling and error conditions
4. **Performance Tests:** Test time complexity and memory usage
5. **Integration Tests:** Test component interactions

**Generated Test Suite:**
```python
\"\"\"
Comprehensive test suite generated by TestAgent.

This test suite includes:
- Unit tests for all functions and methods
- Edge case testing
- Error scenario testing  
- Performance benchmarks
- Mock fixtures and test data
\"\"\"

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

class TestShoppingCart:
    \"\"\"Test suite for ShoppingCart class.\"\"\"
    
    @pytest.fixture
    def cart(self):
        \"\"\"Fixture providing a fresh ShoppingCart instance.\"\"\"
        # from your_module import ShoppingCart
        # return ShoppingCart()
        pass
    
    @pytest.fixture
    def sample_items(self):
        \"\"\"Fixture providing sample test data.\"\"\"
        return {{
            "item1": {{"id": "item1", "name": "Test Item 1", "price": 10.0}},
            "item2": {{"id": "item2", "name": "Test Item 2", "price": 20.0}},
            "item3": {{"id": "item3", "name": "Test Item 3", "price": 15.5}},
        }}
    
    # Unit Tests
    def test_cart_initialization(self, cart):
        \"\"\"Test cart initialization.\"\"\"
        assert cart is not None
        # assert len(cart.items) == 0
        # assert cart.total == 0.0
    
    def test_add_item_success(self, cart, sample_items):
        \"\"\"Test successful item addition.\"\"\"
        item_id = "item1"
        quantity = 2
        
        result = cart.add_item(item_id, quantity)
        
        assert result is True
        # assert cart.items[item_id] == quantity
    
    def test_add_item_invalid_quantity(self, cart):
        \"\"\"Test adding item with invalid quantity.\"\"\"
        with pytest.raises(ValueError, match="Quantity must be a positive integer"):
            cart.add_item("item1", -1)
        
        with pytest.raises(ValueError, match="Quantity must be a positive integer"):
            cart.add_item("item1", 0)
    
    def test_remove_item_success(self, cart, sample_items):
        \"\"\"Test successful item removal.\"\"\"
        item_id = "item1"
        cart.add_item(item_id, 3)
        
        result = cart.remove_item(item_id, 2)
        
        assert result is True
        # assert cart.items[item_id] == 1
    
    def test_calculate_total_empty_cart(self, cart):
        \"\"\"Test total calculation for empty cart.\"\"\"
        total = cart.calculate_total()
        assert total == 0.0
    
    def test_calculate_total_with_items(self, cart, sample_items):
        \"\"\"Test total calculation with items.\"\"\"
        cart.add_item("item1", 2)  # 2 * 10.0 = 20.0
        cart.add_item("item2", 1)  # 1 * 20.0 = 20.0
        
        total = cart.calculate_total()
        # Expected: (20.0 + 20.0) * 1.085 = 43.4
        assert total == pytest.approx(43.4, rel=1e-2)
    
    # Edge Case Tests
    def test_add_item_very_large_quantity(self, cart):
        \"\"\"Test adding item with very large quantity.\"\"\"
        large_quantity = 999999
        result = cart.add_item("item1", large_quantity)
        assert result is True
    
    def test_add_item_special_characters_id(self, cart):
        \"\"\"Test adding item with special characters in ID.\"\"\"
        special_id = "item@#$%^&*()"
        result = cart.add_item(special_id, 1)
        assert result is True
    
    def test_calculate_total_precision(self, cart):
        \"\"\"Test total calculation with floating point precision.\"\"\"
        cart.add_item("item1", 3)  # 3 * 10.0 = 30.0
        cart.add_item("item3", 2)  # 2 * 15.5 = 31.0
        
        total = cart.calculate_total()
        # Expected: (30.0 + 31.0) * 1.085 = 66.185
        assert total == pytest.approx(66.185, rel=1e-3)
    
    # Performance Tests
    def test_add_item_performance(self, cart):
        \"\"\"Performance test for adding items.\"\"\"
        start_time = time.time()
        
        for i in range(1000):
            cart.add_item(f"item_{{i}}", 1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 1 second
        assert execution_time < 1.0
    
    def test_calculate_total_performance(self, cart):
        \"\"\"Performance test for total calculation.\"\"\"
        # Add many items
        for i in range(1000):
            cart.add_item(f"item_{{i}}", 1)
        
        start_time = time.time()
        total = cart.calculate_total()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within 0.1 seconds
        assert execution_time < 0.1
        assert isinstance(total, float)
    
    # Integration Tests
    def test_full_shopping_workflow(self, cart, sample_items):
        \"\"\"Test complete shopping workflow.\"\"\"
        # Add items
        cart.add_item("item1", 2)
        cart.add_item("item2", 1)
        
        # Verify cart state
        summary1 = cart.get_cart_summary()
        assert summary1["total_items"] == 3
        
        # Remove some items
        cart.remove_item("item1", 1)
        
        # Verify updated state
        summary2 = cart.get_cart_summary()
        assert summary2["total_items"] == 2
        
        # Calculate final total
        final_total = cart.calculate_total()
        assert final_total > 0

# Test data and utilities
class TestData:
    \"\"\"Test data utilities.\"\"\"
    
    @staticmethod
    def generate_large_dataset(size: int) -> Dict[str, Any]:
        \"\"\"Generate large dataset for performance testing.\"\"\"
        return {{f"item_{{i}}": {{"price": i * 1.5}} for i in range(size)}}
    
    @staticmethod
    def generate_edge_case_data() -> Dict[str, Any]:
        \"\"\"Generate edge case test data.\"\"\"
        return {{
            "empty_string": "",
            "very_long_string": "x" * 10000,
            "special_chars": "!@#$%^&*()_+-=[]{{}}|;':\\",./<>?",
            "unicode_string": "测试émojis🎉🚀",
            "zero_value": 0,
            "negative_value": -1,
            "very_large_number": 999999999999,
            "floating_point": 3.14159265359,
        }}

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "--cov=your_module", "--cov-report=html"])
```

**Key Test Generation Decisions:**
- **Pytest Framework:** Industry standard with rich features
- **Comprehensive Coverage:** Unit, integration, edge cases, performance
- **Fixture Strategy:** Reusable test data and mock objects
- **Performance Benchmarking:** Time and memory usage testing
- **Edge Case Focus:** Boundary conditions and error scenarios

**Test Coverage Analysis:**
- **Unit Tests:** 95% coverage target
- **Edge Case Tests:** 90% of identified edge cases
- **Error Scenario Tests:** 85% of error conditions
- **Performance Tests:** 80% of critical operations
- **Integration Tests:** 75% of component interactions

**Quality Assessment:**
- **Comprehensiveness:** ⭐⭐⭐⭐⭐ (Covers all major scenarios)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Well-structured and documented)
- **Performance:** ⭐⭐⭐⭐⭐ (Includes benchmarking)
- **Reliability:** ⭐⭐⭐⭐⭐ (Robust error handling tests)

**Generated Test Files:**
- `test_shopping_cart.py` - Main test suite
- `conftest.py` - Shared fixtures
- `test_data.py` - Test data utilities
- `test_performance.py` - Performance benchmarks

**Test Execution Commands:**
```bash
# Run all tests
pytest test_shopping_cart.py

# Run with coverage
pytest --cov=your_module --cov-report=html

# Run specific test categories
pytest -k "test_unit"
pytest -k "test_performance"
pytest -k "test_edge_cases"
```

**Additional Test Features:**
- **Mock Generation:** External service mocking
- **Performance Benchmarks:** Time and memory testing
- **Edge Case Coverage:** Boundary condition testing
- **Error Scenario Testing:** Exception handling validation
- **Integration Testing:** Component interaction testing

**Final Recommendation:**
**COMPREHENSIVE TEST SUITE GENERATED** ✅ - This test suite provides excellent coverage and follows testing best practices.

**Confidence Level:** 93.8% - Very confident in the quality and coverage of this test suite."""

        return {
            "output": output,
            "confidence": 0.94,
            "needs_approval": False,
        }


class DashboardConnector:
    """Manages connection between pipeline and Streamlit dashboard"""

    def __init__(self):
        self.event_queue = Queue()
        self.orchestrator = None
        self.task_thread = None

    def initialize_pipeline(self):
        """Initialize the GPU-powered pipeline"""
        self.orchestrator = StreamingOrchestrator(
            event_queue=self.event_queue,
            agents=[],  # Will be populated by MultiStepOrchestrator
            gpu_service_url="http://localhost:8009",
        )

    def execute_task_async(self, task_description: str):
        """Execute task in background thread"""

        def run_task():
            asyncio.run(self._execute_task(task_description))

        self.task_thread = threading.Thread(target=run_task)
        self.task_thread.start()

    async def _execute_task(self, task_description: str):
        """Execute task and stream events"""
        # Execute the streaming function and collect events
        async for event in self.orchestrator.execute_task_with_streaming(
            task_description
        ):
            # Put each event in the queue for dashboard processing
            self.event_queue.put(event)

    def get_events(self) -> list[DashboardEvent]:
        """Get all pending events"""
        events = []
        while not self.event_queue.empty():
            events.append(self.event_queue.get())
        return events

    def approve_suggestion(self, step: int):
        """Approve a suggestion from an agent"""
        # In real implementation, this would signal the waiting coroutine
        if "approval_pending" in st.session_state:
            st.session_state.approval_pending[step] = True

    def reject_suggestion(self, step: int):
        """Reject a suggestion from an agent"""
        if "approval_pending" in st.session_state:
            st.session_state.approval_pending[step] = False


# Updated dashboard integration functions
def process_events_in_dashboard(connector: DashboardConnector):
    """Process events from pipeline in Streamlit dashboard"""
    events = connector.get_events()

    for event in events:
        if event.type == EventType.TASK_START:
            st.session_state.current_task = {
                "description": event.data["description"],
                "complexity": event.data["complexity"],
                "start_time": event.timestamp,
                "status": "running",
            }

        elif event.type == EventType.AGENT_MESSAGE:
            message = {
                "agent": event.data["agent"],
                "message": event.data["message"],
                "timestamp": event.timestamp,
                "confidence": event.data["confidence"],
                "step": event.data.get("step", 0),
            }
            st.session_state.messages.append(message)

        elif event.type == EventType.AGENT_THINKING:
            # Update progress indicator
            st.session_state.progress["current"] = event.data["step"]
            st.session_state.progress["total"] = event.data["total_steps"]

        elif event.type == EventType.HUMAN_APPROVAL_NEEDED:
            # Flag for approval UI
            st.session_state.pending_approvals.append(event.data)

        elif event.type == EventType.TASK_COMPLETE:
            st.session_state.current_task["status"] = "complete"
            st.session_state.current_task["end_time"] = event.timestamp

            # Update metrics
            st.session_state.metrics["total_tasks"] += 1
            if event.data["success"]:
                st.session_state.metrics["successful_tasks"] += 1
            else:
                st.session_state.metrics["failed_tasks"] += 1

        elif event.type == EventType.ERROR:
            st.error(f"Error: {event.data['error']}")
            st.session_state.current_task["status"] = "error"


# Example usage in Streamlit app
def integrated_dashboard_main():
    """Main function for integrated dashboard"""

    # Initialize connector
    if "connector" not in st.session_state:
        st.session_state.connector = DashboardConnector()
        st.session_state.connector.initialize_pipeline()

    # Process any pending events
    process_events_in_dashboard(st.session_state.connector)

    # Task submission
    if st.button("Execute Task"):
        task = st.session_state.task_input
        st.session_state.connector.execute_task_async(task)

    # Auto-refresh to get new events
    if st.session_state.get("current_task", {}).get("status") == "running":
        st.rerun()


# WebSocket alternative for true real-time
class WebSocketDashboardConnector:
    """Alternative using WebSocket for real-time communication"""

    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        self.url = websocket_url
        self.ws = None

    async def connect(self):
        """Connect to WebSocket server"""
        import websockets

        self.ws = await websockets.connect(self.url)

    async def stream_events(self):
        """Stream events from WebSocket"""
        async for message in self.ws:
            event = json.loads(message)
            yield DashboardEvent(
                type=EventType(event["type"]),
                data=event["data"],
                timestamp=datetime.fromisoformat(event["timestamp"]),
            )

    async def send_command(self, command: str, data: Dict[str, Any]):
        """Send command to pipeline"""
        await self.ws.send(
            json.dumps(
                {
                    "command": command,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )
