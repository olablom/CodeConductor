# GitHub Actions Workflows

## Overview
Detta dokument beskriver de olika GitHub Actions workflows som körs i CodeConductor-projektet.

## Workflows

### 1. Simple Test (`simple-test.yml`)
**Syfte:** Grundläggande tester som alltid ska köras
**Triggers:** PR och push till main
**Körs på:** Ubuntu latest
**Vad den gör:**
- Installerar dependencies
- Testar att CodeConductor kan importeras
- Kör enkla pytest-tester med max 3 fel

### 2. Smoke (`smoke.yml`)
**Syfte:** Snabba tester på Windows
**Triggers:** PR och push till main
**Körs på:** Windows latest
**Vad den gör:**
- Installerar dependencies
- Testar att CodeConductor kan importeras
- Kör specifika tester om de finns

### 3. CI (`ci.yml`)
**Syfte:** Kontinuerlig integration
**Triggers:** PR och push till main
**Körs på:** Ubuntu latest
**Vad den gör:**
- Installerar dependencies
- Testar att CodeConductor kan importeras
- Kör tester med max 5 fel

### 4. Test (`test.yml`)
**Syfte:** Omfattande tester på Linux
**Triggers:** PR och push till main
**Körs på:** Ubuntu latest
**Vad den gör:**
- Installerar dependencies
- Kör linting (black, flake8)
- Kör alla tester med coverage
- Kör benchmarks (om de finns)

### 5. Test Windows (`test-windows.yml`)
**Syfte:** Omfattande tester på Windows
**Triggers:** PR och push till main
**Körs på:** Windows latest
**Vad den gör:**
- Installerar dependencies
- Kör alla tester med coverage
- Hanterar Windows-specifika skip

## Rekommendationer

### För snabba tester:
- Använd **Simple Test** eller **Smoke**

### För fullständig validering:
- Använd **Test** (Linux) och **Test Windows**

### För kontinuerlig integration:
- Använd **CI**

## Felsökning

### Vanliga problem:
1. **Dependencies misslyckas:** Kontrollera `requirements.txt`
2. **Tester misslyckas:** Kör lokalt med `pytest tests/`
3. **Windows-specifika problem:** Vissa tester skippas på Windows (vLLM)

### Loggar:
- Alla workflows laddar upp artifacts
- Kontrollera "Actions" fliken på GitHub
- Ladda ner test-rapporter för detaljerad information
