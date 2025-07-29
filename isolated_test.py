#!/usr/bin/env python3
"""
Isolated test for CodeConductor → LM Studio communication
Tests ONLY the communication layer without Streamlit/Ensemble complexity
"""

import requests
import json
import time
from typing import Dict, Any

class IsolatedLMStudioTest:
    def __init__(self):
        self.base_url = "http://127.0.0.1:1234"
        self.timeout = 30
        
    def test_basic_connection(self) -> bool:
        """Test if LM Studio is reachable"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            print(f"✅ Connection OK: {response.status_code}")
            
            # Print available models
            if response.status_code == 200:
                models = response.json()
                print(f"📦 Available models: {len(models.get('data', []))}")
                for model in models.get('data', []):
                    print(f"   - {model.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def test_simple_generation(self, model_name: str) -> Dict[str, Any]:
        """Test simple code generation with specific model"""
        print(f"\n🧪 Testing generation with: {model_name}")
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": "Create a simple Python function that prints 'Hello, World!'"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stream": False
        }
        
        print(f"📤 Sending request...")
        print(f"   URL: {self.base_url}/v1/chat/completions")
        print(f"   Payload: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            print(f"⏱️  Response time: {duration:.2f}s")
            print(f"📤 Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the generated content
                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    print(f"✅ Generated content ({len(content)} chars):")
                    print(f"---\n{content}\n---")
                    
                    return {
                        'success': True,
                        'content': content,
                        'duration': duration,
                        'status_code': response.status_code,
                        'full_response': data
                    }
                else:
                    print(f"❌ No choices in response: {data}")
                    return {'success': False, 'error': 'No choices in response', 'data': data}
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return {
                    'success': False, 
                    'error': f'HTTP {response.status_code}',
                    'response_text': response.text
                }
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out after {self.timeout}s")
            return {'success': False, 'error': 'Timeout'}
            
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_codeconductor_format(self, model_name: str) -> Dict[str, Any]:
        """Test with the exact format CodeConductor uses"""
        print(f"\n🎯 Testing CodeConductor format with: {model_name}")
        
        # This is the exact prompt format from CodeConductor
        codeconductor_prompt = """## Task: Create a Python function that prints "Hello, World!"

### Approach
Default approach

### Requirements
- Implement requested functionality

### Constraints
- Use type hints
- Include docstrings
- Follow PEP 8
- Handle errors gracefully

### Output Format
Please provide the code in the following format:

```python
# Your implementation here
```

```test_test_implementation.py
# Your test cases here
```

Make sure all tests pass and the code follows the specified standards."""

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": codeconductor_prompt}],
            "max_tokens": 2048,
            "temperature": 0.25,
            "stream": False
        }
        
        return self.test_simple_generation_with_payload(payload)
    
    def test_simple_generation_with_payload(self, payload: Dict) -> Dict[str, Any]:
        """Helper method to test with custom payload"""
        print(f"📤 Custom payload test...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )
            
            duration = time.time() - start_time
            print(f"⏱️  Response time: {duration:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    print(f"✅ SUCCESS! Generated {len(content)} characters")
                    print(f"📝 First 200 chars: {content[:200]}...")
                    return {'success': True, 'content': content, 'duration': duration}
                    
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return {'success': False, 'error': response.text}
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Run all tests"""
    print("🚀 Starting Isolated LM Studio Communication Test")
    print("=" * 60)
    
    tester = IsolatedLMStudioTest()
    
    # Test 1: Basic connection
    print("\n📡 TEST 1: Basic Connection")
    if not tester.test_basic_connection():
        print("❌ Basic connection failed - stopping tests")
        return
    
    # Test 2: Simple generation with each model
    test_models = [
        "mistral-7b-instruct-v0.1",
        "codellama-7b-instruct"
    ]
    
    results = {}
    
    for model in test_models:
        print(f"\n🧪 TEST 2: Simple Generation - {model}")
        result = tester.test_simple_generation(model)
        results[f"simple_{model}"] = result
        
        if result.get('success'):
            print(f"✅ {model}: SUCCESS in {result.get('duration', 0):.2f}s")
        else:
            print(f"❌ {model}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Test 3: CodeConductor format
    for model in test_models:
        print(f"\n🎯 TEST 3: CodeConductor Format - {model}")
        result = tester.test_codeconductor_format(model)
        results[f"codeconductor_{model}"] = result
        
        if result.get('success'):
            print(f"✅ {model}: SUCCESS in {result.get('duration', 0):.2f}s")
        else:
            print(f"❌ {model}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results.values() if r.get('success'))
    total_tests = len(results)
    
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("💡 LM Studio communication works perfectly")
        print("🔍 The problem is likely in CodeConductor's integration layer")
    elif successful_tests > 0:
        print("\n⚠️  PARTIAL SUCCESS")
        print("💡 Some models work, others don't - investigate specific failures")
    else:
        print("\n💥 ALL TESTS FAILED")
        print("💡 Problem is likely in LM Studio setup or network connectivity")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Optional: Save results to file for further analysis
    with open("lm_studio_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: lm_studio_test_results.json")