# CodeConductor Test Strategy

## 🎯 **Testnivåer**

### **1. Unit/CI Tester (Mockade, Snabba)**
- **Syfte:** Validera kodlogik, API-format, async-flöden
- **Körs:** I CI, på alla maskiner, utan GPU
- **GPU:** Mock-data, `CC_GPU_DISABLED=1`
- **Exempel:** `pytest -k "not gpu and not vllm"`

### **2. GPU Integrationstester (Riktiga, Lokala)**
- **Syfte:** Validera riktig hårdvara på RTX 5090
- **Körs:** Lokalt på din 5090, inte i CI
- **GPU:** Riktiga anrop, `nvidia-smi`, PyTorch CUDA
- **Exempel:** `pytest -m gpu`

### **3. vLLM Tester (Riktiga, Lokala)**
- **Syfte:** Validera vLLM-integration
- **Körs:** Lokalt med vLLM igång
- **GPU:** Riktiga modeller, inference
- **Exempel:** `pytest -m vllm`

## 🚀 **Kör tester**

### **Snabba tester (CI-kompatibla)**
```bash
# Alla tester utom GPU/vLLM
pytest -k "not gpu and not vllm"

# Endast unit-tester
pytest -m unit

# Med coverage
pytest --cov=codeconductor -k "not gpu and not vllm"
```

### **GPU-tester (lokalt på 5090)**
```bash
# Aktivera riktig GPU
set CC_GPU_DISABLED=0  # Windows
export CC_GPU_DISABLED=0  # Linux/Mac

# Kör GPU-tester
pytest -m gpu -v

# Endast GPU memory tester
pytest tests/test_gpu_integration.py::test_real_gpu_memory_info -v
```

### **vLLM-tester (lokalt med vLLM)**
```bash
# Starta vLLM först
# Kör vLLM-tester
pytest -m vllm -v
```

## 🔒 **Säkerhetsfunktioner**

### **Automatisk GPU-sanitizer**
- **Default:** `CC_GPU_DISABLED=1` (mock-mode)
- **Override:** Sätt miljövariabel för riktiga tester
- **Cleanup:** VRAM-städning efter varje test

### **Miljövariabler**
```bash
# Säkert läge (default)
CC_TESTING_MODE=1
CC_GPU_DISABLED=1

# Riktigt GPU-läge
CC_TESTING_MODE=0
CC_GPU_DISABLED=0
```

## 📊 **Testresultat**

### **CI (GitHub Actions)**
- ✅ **Unit-tester:** Alla passerar
- ⏭️ **GPU-tester:** Skippas (ingen GPU)
- ⏭️ **vLLM-tester:** Skippas (ingen vLLM)

### **Lokalt (RTX 5090)**
- ✅ **Unit-tester:** Alla passerar
- ✅ **GPU-tester:** Validerar riktig hårdvara
- ✅ **vLLM-tester:** Validerar modeller

## 🧪 **Testa lokalt**

### **1. Snabba tester (säkert)**
```bash
python scripts/test_simple.py
```

### **2. GPU-tester (riktig hårdvara)**
```bash
# Aktivera GPU
set CC_GPU_DISABLED=0

# Kör GPU-tester
pytest tests/test_gpu_integration.py -v

# Återställ säkert läge
set CC_GPU_DISABLED=1
```

### **3. Fullständig testsvit**
```bash
# Säkert läge
pytest -k "not gpu and not vllm" --tb=short

# GPU-läge (lokalt)
pytest -m gpu --tb=short

# Alla tester (varning: kan ta lång tid)
pytest --tb=short
```

## 🚨 **Felsökning**

### **GPU-tester hänger**
- Kontrollera att `CC_GPU_DISABLED=0`
- Verifiera att `nvidia-smi` fungerar
- Starta om om nödvändigt

### **vLLM-tester misslyckas**
- Kontrollera att vLLM är igång
- Verifiera att modeller är tillgängliga
- Kontrollera port 8000

### **Async-fel**
- Kör `pytest -k "not gpu"` först
- Kontrollera att debattpipelinen är korrekt
- Verifiera `ensure_async` wrapper

## 📝 **Lägg till nya tester**

### **Unit-test (mockad)**
```python
def test_model_manager_creation():
    """Test that ModelManager can be created"""
    from codeconductor.ensemble.model_manager import ModelManager
    mm = ModelManager()
    assert mm is not None
```

### **GPU-test (riktig hårdvara)**
```python
@pytest.mark.gpu
def test_real_gpu_functionality():
    """Test real GPU functionality"""
    # Din GPU-test här
    pass
```

### **vLLM-test (riktig modell)**
```python
@pytest.mark.vllm
def test_real_model_inference():
    """Test real model inference"""
    # Din vLLM-test här
    pass
```
