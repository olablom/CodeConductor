#!/usr/bin/env python3
"""
Test script for Phone Validator functionality
Tests the ensemble system with the phone validator task
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ensemble.model_manager import ModelManager
from ensemble.query_dispatcher import QueryDispatcher
from ensemble.consensus_calculator import ConsensusCalculator
from generators.prompt_generator import PromptGenerator

async def test_phone_validator():
    """Test the Phone Validator task with the ensemble system"""
    
    print("🧪 Testing Phone Validator functionality...")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing components...")
    model_manager = ModelManager()
    query_dispatcher = QueryDispatcher()  # QueryDispatcher creates its own ModelManager
    consensus_calculator = ConsensusCalculator()
    prompt_generator = PromptGenerator()
    
    # Phone Validator task (same as in the Streamlit app)
    task = "Create a function to validate Swedish phone numbers"
    print(f"2. Task: {task}")
    
    # Check available models
    print("3. Checking available models...")
    models = await model_manager.list_models()
    print(f"   Found {len(models)} models: {models}")
    
    if not models:
        print("❌ No models found. Please ensure LM Studio or Ollama is running.")
        return False
    
    # Generate prompt
    print("4. Generating prompt...")
    try:
        prompt = prompt_generator.generate_code_generation_prompt(task)
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Prompt preview: {prompt[:200]}...")
    except Exception as e:
        print(f"❌ Error generating prompt: {e}")
        return False
    
    # Run ensemble processing
    print("5. Running ensemble processing...")
    try:
        results = await query_dispatcher.dispatch_with_fallback(task, min_models=2)
        
        if not results:
            print("❌ No models responded")
            return False
        
        print(f"   Got responses from {len(results)} models")
        
        # Format results for consensus
        formatted_results = []
        for model_id, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                # Extract response content from model response
                response_content = ""
                if "choices" in result and result["choices"]:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        response_content = choice["message"]["content"]
                    elif "text" in choice:
                        response_content = choice["text"]
                elif "response" in result:
                    response_content = result["response"]
                
                if response_content:
                    formatted_results.append({
                        "model_id": model_id,
                        "success": True,
                        "response": response_content,
                        "response_time": result.get("response_time", 0),
                    })
                    print(f"   ✅ {model_id}: {len(response_content)} chars")
                else:
                    print(f"   ❌ {model_id}: No content in response")
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                print(f"   ❌ {model_id}: {error_msg}")
        
        if not formatted_results:
            print("❌ No successful responses")
            return False
        
        # Calculate consensus
        print("6. Calculating consensus...")
        consensus = consensus_calculator.calculate_consensus(formatted_results)
        
        print("7. Results:")
        print(f"   Consensus confidence: {consensus.get('confidence', 0):.2f}")
        print(f"   Best response length: {len(consensus.get('best_response', ''))} characters")
        
        # Show a preview of the generated code
        best_response = consensus.get('best_response', '')
        if best_response:
            print("\n📱 Generated Phone Validator Code Preview:")
            print("-" * 40)
            lines = best_response.split('\n')[:20]  # Show first 20 lines
            for line in lines:
                print(f"   {line}")
            if len(best_response.split('\n')) > 20:
                print("   ...")
        
        print("\n✅ Phone Validator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during ensemble processing: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_phone_validator())
    sys.exit(0 if success else 1) 