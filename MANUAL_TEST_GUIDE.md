# 🖐️ **CODECONDUCTOR - MANUAL TEST GUIDE**

**Fokus: Endast UI/UX och subjektiva kvalitetsbedömningar (20% av testerna)**

---

## 📋 **TEST SUMMARY CHECKLIST**

| **Test Area**          | **Priority**    | **Status**             | **Score**  | **Notes**            |
| ---------------------- | --------------- | ---------------------- | ---------- | -------------------- |
| **Automated Tests**    | 🔴 Critical     | ⬜ Pass/❌ Fail        | \_\_\_/15  | Must be 80%+         |
| **UI/UX Design**       | 🟡 Important    | ⬜ Pass/❌ Fail        | \_\_\_/10  | Visual assessment    |
| **Code Quality**       | 🔴 Critical     | ⬜ Pass/❌ Fail        | \_\_\_/30  | 3 tasks (10 each)    |
| **RAG System**         | 🟢 Nice-to-have | ⬜ Pass/❌ Fail        | \_\_\_/20  | Context quality      |
| **Ensemble Consensus** | 🔴 Critical     | ⬜ Pass/❌ Fail        | \_\_\_/20  | Model selection      |
| **Performance**        | 🟡 Important    | ⬜ Pass/❌ Fail        | \_\_\_/10  | Response times       |
| **Overall**            | -               | ⬜ Ready/❌ Needs Work | \_\_\_/105 | Production readiness |

**Legend:** 🔴 Critical (must pass) | 🟡 Important | 🟢 Nice-to-have

---

## 🛠️ **ENVIRONMENT SPECIFICATIONS**

### **System Requirements**

- **Python:** 3.10+
- **Browser:** Chrome 90+ / Firefox 88+ / Edge 90+
- **Screen Resolution:** 1920x1080 minimum, 2560x1440 recommended
- **RAM:** 16GB+ (for 6 local LLMs)
- **Network:** Stable internet for external context

### **Required Services**

- **LM Studio:** Running on port 1234
- **Ollama:** Running on port 11434
- **Health API:** Running on port 5000
- **Streamlit App:** Running on port 8501

### **Environment Variables**

```bash
# Optional but recommended
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # For GPU acceleration
```

---

## 🚀 **STEG 1: KÖR AUTOMATISERADE TESTER FÖRST** ⏱️ 2-3 min

```bash
# Starta health API
python health_api.py

# Kör automatiserade tester
python test_automated_suite.py
```

**Förväntat resultat:** 80%+ success rate innan manuell testning

**✅ Success Criteria:** 13/15+ tests pass

---

## 🎨 **STEG 2: UI/UX MANUELLA TESTER** ⏱️ 5-7 min

### **2.1 Visual Design Assessment** 🟡 Important

- **Öppna:** `http://localhost:8501`
- **Bedöm:**
  - [ ] Moderna gradients och design (se bild 1)
  - [ ] Responsiv layout (testa 1920x1080, 1366x768, mobile)
  - [ ] Professionell färgschema (blå/grön gradient)
  - [ ] Tydlig typografi och spacing
  - [ ] Loading spinners fungerar (se bild 2)
  - [ ] Smooth transitions mellan tabs

**Visual Examples:**

- **Bild 1 - Correct Design:** Modern gradient header med "CodeConductor" title
- **Bild 2 - Loading State:** Animated progress bar med status text
- **Bild 3 - Error State:** Red error box med "Generation failed" message

### **2.2 Navigation och UX** 🔴 Critical

- **Testa:**
  - [ ] Tab navigation fungerar (Task Input → Results → Patterns → Validation)
  - [ ] Sidebar expanderar/kollapsar (hamburger menu)
  - [ ] Buttons är klickbara och responsiva (hover effects)
  - [ ] Form inputs fungerar korrekt (text areas, dropdowns)
  - [ ] Error messages visas tydligt (red boxes)
  - [ ] Success feedback fungerar (green checkmarks)

### **2.3 Real-time Feedback** 🟡 Important

- **Bedöm:**
  - [ ] Progress bars uppdateras (0% → 100%)
  - [ ] Status indicators fungerar (🟢 Online / 🔴 Offline)
  - [ ] Live metrics uppdateras (response times, model usage)
  - [ ] Model health visas korrekt (6/6 models online)
  - [ ] RAG context panel fungerar (external results)

---

## 📊 **STEG 3: KODKVALITET BEDÖMNING** ⏱️ 8-10 min 🔴 Critical

### **3.1 Enkel Task - Hello World** (3 min)

- **Input:** `Create a Python function that prints "Hello, World!"`
- **Bedöm kodkvalitet (1-10):**
  - [ ] Funktionen är korrekt (def print_hello(): print("Hello, World!"))
  - [ ] Kod är läsbar (proper indentation)
  - [ ] Inga onödiga imports
  - [ ] Proper indentation (4 spaces)
  - **Total score: \_\_\_/10**

**Expected Output:**

```python
def print_hello():
    print("Hello, World!")
```

### **3.2 Medium Task - Todo List** (3 min)

- **Input:** `Create a Python class for managing a todo list with add, remove, list, and mark_complete methods`
- **Bedöm kodkvalitet (1-10):**
  - [ ] Alla 4 metoder implementerade
  - [ ] Type hints finns (List[str], Optional[int])
  - [ ] Docstrings finns ("""Add a new todo item""")
  - [ ] Error handling (try/except)
  - [ ] Clean code principles (single responsibility)
  - **Total score: \_\_\_/10**

**Expected Output:**

```python
class TodoList:
    def __init__(self):
        self.todos: List[str] = []

    def add(self, item: str) -> None:
        """Add a new todo item"""
        self.todos.append(item)
```

### **3.3 Complex Task - Flask API** (4 min)

- **Input:** `Create a Flask REST API with JWT authentication and user registration`
- **Bedöm kodkvalitet (1-10):**
  - [ ] Flask app struktur (app = Flask(**name**))
  - [ ] JWT implementation (jwt.encode/decode)
  - [ ] User registration (@app.route('/register', methods=['POST']))
  - [ ] Error handling (try/except, proper HTTP codes)
  - [ ] Security best practices (password hashing)
  - [ ] Code organization (separate functions)
  - **Total score: \_\_\_/10**

---

## 🧠 **STEG 4: RAG SYSTEM KVALITET** ⏱️ 3-4 min 🟢 Nice-to-have

### **4.1 Context Relevans**

- **Input:** `Create a FastAPI application with SQLAlchemy models`
- **Bedöm RAG context (1-10):**
  - [ ] Relevanta dokument hittas (FastAPI docs, SQLAlchemy examples)
  - [ ] Stack Overflow results är användbara (recent, relevant)
  - [ ] Context förbättrar prompten (adds specific details)
  - [ ] Information är aktuell (2023+ content)
  - **Total score: \_\_\_/10**

### **4.2 Prompt Enhancement**

- **Bedöm prompt kvalitet (1-10):**
  - [ ] Prompt innehåller relevant context (FastAPI patterns)
  - [ ] Tydlig struktur (## Task, ## Context, ## Requirements)
  - [ ] Specific instructions (use FastAPI, include models)
  - [ ] Proper formatting (markdown, code blocks)
  - **Total score: \_\_\_/10**

---

## 🎯 **STEG 5: ENSEMBLE KONSENSUS KVALITET** ⏱️ 3-4 min 🔴 Critical

### **5.1 Consensus Quality**

- **Bedöm consensus (1-10):**
  - [ ] Modeller är överens (similar approaches)
  - [ ] Confidence score stämmer (0.7+ for good consensus)
  - [ ] Bästa modell vald (fastest + most accurate)
  - [ ] Fallback fungerar (timeout → alternative model)
  - **Total score: \_\_\_/10**

### **5.2 Model Selection**

- **Bedöm model selection (1-10):**
  - [ ] Rätt modell för rätt task (phi3 for simple, codellama for complex)
  - [ ] Performance vs quality balans (speed vs accuracy)
  - [ ] Timeout handling (30s timeout, graceful fallback)
  - [ ] Error recovery (failed model → alternative)
  - **Total score: \_\_\_/10**

---

## 📈 **STEG 6: PERFORMANCE BEDÖMNING** ⏱️ 2-3 min 🟡 Important

### **6.1 Response Times**

- **Mät och bedöm:**
  - [ ] Enkel task: < 10 sekunder
  - [ ] Medium task: < 20 sekunder
  - [ ] Complex task: < 30 sekunder
  - [ ] Acceptabel för production
  - **Bedömning: Acceptabel/Inte acceptabel**

### **6.2 User Experience**

- **Bedöm UX (1-10):**
  - [ ] Intuitiv workflow (clear steps)
  - [ ] Tydlig feedback (progress bars, status)
  - [ ] Minimal frustration (no hanging, clear errors)
  - [ ] Professional feel (smooth, responsive)
  - **Total score: \_\_\_/10**

---

## 📝 **STEG 7: SAMMANFATTNING** ⏱️ 2-3 min

### **7.1 Overall Assessment**

- **UI/UX Score:** \_\_\_/10
- **Code Quality Score:** \_\_\_/30
- **RAG Quality Score:** \_\_\_/20
- **Ensemble Quality Score:** \_\_\_/20
- **Performance Score:** \_\_\_/10
- **Overall Score:** \_\_\_/90

### **7.2 Production Readiness**

- [ ] ✅ Ready for production (80+ points)
- [ ] ⚠️ Needs minor improvements (60-79 points)
- [ ] ❌ Needs major improvements (40-59 points)
- [ ] ❌ Not ready (< 40 points)

### **7.3 Key Findings**

**Vad fungerar bäst:**

- Ensemble engine consensus
- Code generation quality
- UI responsiveness

**Vad behöver förbättras:**

- RAG context relevance
- Response times for complex tasks
- Error message clarity

**Rekommendationer:**

- Install sentence-transformers for full RAG
- Optimize model selection for speed
- Add more visual feedback

---

## 🎯 **SLUTANALYS**

**Automatiserade tester:** \_\_\_/15 passerade
**Manuella tester:** \_\_\_/90 poäng
**Total bedömning:** \_\_\_/105

**Status:** ✅ Production Ready / ⚠️ Needs Work / ❌ Not Ready

**Time Estimate:** 25-35 minutes total

---

**Klistra in dina resultat här så analyserar jag dem!** 🚀
