#!/usr/bin/env python3
"""
Debug script to identify hanging issues in CodeConductor ensemble engine
"""

import asyncio
import logging
import time
from ensemble.ensemble_engine import EnsembleEngine
from ensemble.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_model_discovery():
    """Test model discovery with timeouts."""
    print("🔍 Testing model discovery...")
    
    manager = ModelManager()
    
    try:
        # Test with shorter timeout
        models = await asyncio.wait_for(manager.list_models(), timeout=15.0)
        print(f"✅ Found {len(models)} models")
        for model in models:
            print(f"  - {model.id} ({model.provider})")
        return models
    except asyncio.TimeoutError:
        print("❌ Model discovery timed out")
        return []
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        return []

async def test_health_checks(models):
    """Test health checks with timeouts."""
    print("🏥 Testing health checks...")
    
    manager = ModelManager()
    
    for model in models:
        try:
            # Test with shorter timeout
            is_healthy = await asyncio.wait_for(
                manager.check_health(model), timeout=5.0
            )
            status = "✅" if is_healthy else "❌"
            print(f"  {status} {model.id}")
        except asyncio.TimeoutError:
            print(f"  ⏰ {model.id} - health check timed out")
        except Exception as e:
            print(f"  ❌ {model.id} - health check failed: {e}")

async def test_single_model_query(model_id, provider):
    """Test querying a single model."""
    print(f"🎯 Testing single model query: {model_id}")
    
    from ensemble.query_dispatcher import QueryDispatcher
    
    test_prompt = "What is 2 + 2? Answer with just the number."
    
    try:
        async with QueryDispatcher() as dispatcher:
            # Create a mock ModelInfo
            from ensemble.model_manager import ModelInfo
            model_info = ModelInfo(
                id=model_id,
                name=model_id,
                provider=provider,
                endpoint="http://localhost:1234/v1" if provider == "lm_studio" else "http://localhost:11434",
                is_available=True
            )
            
            # Test with timeout
            result = await asyncio.wait_for(
                dispatcher.dispatch_parallel([model_info], test_prompt),
                timeout=30.0
            )
            
            print(f"  ✅ Query completed: {result}")
            return result
            
    except asyncio.TimeoutError:
        print(f"  ⏰ Query timed out for {model_id}")
        return None
    except Exception as e:
        print(f"  ❌ Query failed for {model_id}: {e}")
        return None

async def test_ensemble_initialization():
    """Test ensemble engine initialization with timeouts."""
    print("🚀 Testing ensemble engine initialization...")
    
    try:
        engine = EnsembleEngine(min_confidence=0.6)
        
        # Test initialization with timeout
        success = await asyncio.wait_for(engine.initialize(), timeout=20.0)
        
        if success:
            print("✅ Ensemble engine initialized successfully")
            return engine
        else:
            print("❌ Ensemble engine initialization failed")
            return None
            
    except asyncio.TimeoutError:
        print("⏰ Ensemble engine initialization timed out")
        return None
    except Exception as e:
        print(f"❌ Ensemble engine initialization failed: {e}")
        return None

async def test_ensemble_request(engine):
    """Test ensemble request with timeouts."""
    print("🎯 Testing ensemble request...")
    
    test_task = "Create a simple Python function that prints 'Hello, World!'"
    
    try:
        # Test with timeout
        result = await asyncio.wait_for(
            engine.process_request_with_fallback(test_task),
            timeout=60.0
        )
        
        print(f"✅ Ensemble request completed: {result}")
        return result
        
    except asyncio.TimeoutError:
        print("⏰ Ensemble request timed out")
        return None
    except Exception as e:
        print(f"❌ Ensemble request failed: {e}")
        return None

async def main():
    """Main diagnostic function."""
    print("🔧 CodeConductor Hanging Issue Diagnostic")
    print("=" * 50)
    
    # Step 1: Test model discovery
    models = await test_model_discovery()
    
    if not models:
        print("❌ No models found. Check if Ollama or LM Studio is running.")
        return
    
    # Step 2: Test health checks
    await test_health_checks(models)
    
    # Step 3: Test single model queries
    for model in models[:2]:  # Test first 2 models
        await test_single_model_query(model.id, model.provider)
    
    # Step 4: Test ensemble initialization
    engine = await test_ensemble_initialization()
    
    if engine:
        # Step 5: Test ensemble request
        await test_ensemble_request(engine)
    
    print("\n🔧 Diagnostic complete!")

if __name__ == "__main__":
    asyncio.run(main()) 