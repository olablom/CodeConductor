# 🚀 Cursor Prompt: Improve Agent Output for CodeConductor

## 🎯 **Objective:**

Update all agent prompts in the CodeConductor system to output much more detailed, verbose, and educational responses instead of just "task completed" messages.

## 📋 **Specific Requirements:**

### **1. ArchitectAgent Improvements:**

- **Show detailed design reasoning** with step-by-step explanations
- **Explain architectural patterns** and why they were chosen
- **Include trade-off analysis** for different approaches
- **Provide system diagrams** or structure outlines
- **Show confidence scores** for each design decision

### **2. CodeGenAgent Improvements:**

- **Display actual generated code** with inline comments
- **Explain implementation choices** and algorithms used
- **Show error handling** and edge case considerations
- **Include usage examples** and documentation
- **Break down complex logic** into understandable parts

### **3. ReviewAgent Improvements:**

- **Provide line-by-line feedback** with specific suggestions
- **Identify potential issues** and security concerns
- **Suggest optimizations** with explanations
- **Rate code quality** on multiple dimensions
- **Show before/after improvements**

### **4. PolicyAgent Improvements:**

- **List all security checks** performed
- **Explain policy violations** in detail
- **Provide remediation suggestions** for issues found
- **Show compliance status** for different standards
- **Include risk assessments** for flagged code

## 🎨 **Output Format Standards:**

### **Use Markdown Formatting:**

````markdown
## Agent: [AgentName]

### Task: [Task Description]

### Confidence: [X.XX%]

**Reasoning Process:**

1. [Step 1 with explanation]
2. [Step 2 with explanation]
3. [Step 3 with explanation]

**Design/Implementation:**

```python
# Actual code with comments
def example_function():
    # Explain what this does
    pass
```
````

**Key Decisions:**

- **Decision 1:** [Why this was chosen]
- **Decision 2:** [Why this was chosen]

**Quality Assessment:**

- **Performance:** [Rating with explanation]
- **Security:** [Rating with explanation]
- **Maintainability:** [Rating with explanation]

```

## 🔍 **Files to Update:**

### **Primary Agent Files:**
- `agents/architect_agent.py` - Update propose() method
- `agents/codegen_agent.py` - Update propose() method
- `agents/review_agent.py` - Update propose() method
- `agents/policy_agent.py` - Update propose() method

### **Prompt Templates:**
- `prompts/architect_prompts.md` - If exists
- `prompts/codegen_prompts.md` - If exists
- `prompts/review_prompts.md` - If exists
- `prompts/policy_prompts.md` - If exists

### **Configuration Files:**
- Any config files that define agent prompts
- Template files for agent responses

## 🎯 **Example Transformations:**

### **Before (Current):**
```

ArchitectAgent completed: Design architecture

````

### **After (Improved):**
```markdown
## ArchitectAgent
### Task: Design Fibonacci Calculator Architecture
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
        self.memo = {}  # Cache for calculated values

    def calculate(self, n: int) -> int:
        # Implementation with detailed error handling
        pass

    def reset_cache(self):
        # Clear memoization cache
        pass
````

**Key Design Decisions:**

- **Class-based approach:** Allows for state management (memoization cache)
- **Type hints:** Improves code clarity and IDE support
- **Error handling:** Comprehensive validation for robust operation
- **Cache management:** User can reset cache if needed

**Quality Metrics:**

- **Performance:** ⭐⭐⭐⭐⭐ (O(n) time, O(n) space with memoization)
- **Maintainability:** ⭐⭐⭐⭐⭐ (Clear structure, well-documented)
- **Scalability:** ⭐⭐⭐⭐ (Handles large numbers efficiently)

```

## 🚀 **Implementation Strategy:**

### **Step 1: Find All Agent Files**
Use `@workspace` to locate all agent-related files and prompt definitions.

### **Step 2: Update Agent Methods**
Modify the `propose()` method in each agent to return detailed, formatted responses.

### **Step 3: Update Prompt Templates**
If prompt templates exist, update them to encourage verbose, educational output.

### **Step 4: Test Integration**
Ensure the dashboard can properly display the new detailed format.

## 💡 **Additional Enhancements:**

### **Add Educational Elements:**
- **Learning points:** Explain concepts for educational value
- **Best practices:** Highlight industry standards used
- **Alternative approaches:** Mention other valid solutions
- **Performance analysis:** Explain time/space complexity

### **Improve Readability:**
- **Use emojis** for visual separation (🎯, 💡, ⚠️, ✅)
- **Color coding** through markdown formatting
- **Structured sections** with clear headers
- **Code blocks** with syntax highlighting

## 🎯 **Success Criteria:**

After these updates, when a user submits a task, they should see:
1. **Detailed reasoning** from each agent
2. **Actual code/designs** produced
3. **Educational explanations** of choices made
4. **Quality assessments** with specific metrics
5. **Professional formatting** that's easy to read

## 🚀 **Execute This Prompt:**

Use this prompt in Cursor with `@workspace` to find and update all relevant files. The goal is to transform the agent output from simple "task completed" messages to rich, educational, and detailed responses that show exactly what each agent is thinking and producing.

**Make the agents verbose, educational, and professional!** 🎯
```
