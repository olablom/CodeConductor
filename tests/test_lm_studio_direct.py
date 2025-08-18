#!/usr/bin/env python3
"""
Direct LM Studio Test - Isolate the server issue

This tests the LM Studio server directly without any CodeConductor layers.
"""

import asyncio
import time

import aiohttp


class DirectLMStudioTester:
    """Direct tester for LM Studio server"""

    def __init__(self):
        self.base_url = "http://localhost:1234"
        self.model = "meta-llama-3.1-8b-instruct"

    async def test_server_connection(self):
        """Test basic server connectivity"""
        print("🔍 Testing LM Studio server connectivity...")

        try:
            async with aiohttp.ClientSession() as session:
                # Test basic endpoint
                async with session.get(f"{self.base_url}/v1/models", timeout=10) as response:
                    if response.status == 200:
                        print("✅ Server is running and responding")
                        return True
                    else:
                        print(f"❌ Server responded with status: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    async def test_simple_completion(self, prompt: str, timeout: float = 30.0):
        """Test a simple completion request"""
        print("\n🧪 Testing simple completion...")
        print(f"📝 Prompt: {prompt}")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    execution_time = time.time() - start_time

                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        print(f"✅ Success! Response in {execution_time:.1f}s")
                        print(f"📄 Response length: {len(content)} characters")
                        print(f"📝 First 200 chars: {content[:200]}...")

                        return {
                            "success": True,
                            "execution_time": execution_time,
                            "response_length": len(content),
                            "content": content,
                        }
                    else:
                        error_text = await response.text()
                        print(f"❌ Server error: {response.status}")
                        print(f"📄 Error details: {error_text}")

                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "execution_time": execution_time,
                        }

        except TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏰ Request timed out after {execution_time:.1f}s")
            return {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Request failed: {e}")
            return {"success": False, "error": str(e), "execution_time": execution_time}

    async def test_code_generation(self):
        """Test code generation specifically"""
        print("\n🧪 Testing code generation...")

        test_cases = [
            "Write a Python function to calculate fibonacci numbers",
            "Create a simple Python class for a Todo item",
            "Write a Python function to reverse a string",
        ]

        results = []

        for i, prompt in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/3: {prompt}")
            result = await self.test_simple_completion(prompt, timeout=60.0)
            results.append(result)

            if result["success"]:
                print(f"✅ Test {i} passed!")
            else:
                print(f"❌ Test {i} failed: {result.get('error', 'Unknown error')}")

        return results

    async def run_diagnostic(self):
        """Run full diagnostic"""
        print("🔧 LM Studio Server Diagnostic")
        print("=" * 50)

        # Test 1: Server connectivity
        server_ok = await self.test_server_connection()

        if not server_ok:
            print("\n❌ SERVER ISSUE DETECTED!")
            print("   - LM Studio server is not running")
            print("   - Or server is not accessible on localhost:1234")
            print("   - Check if LM Studio is started and models are loaded")
            return

        # Test 2: Simple completion
        print("\n🧪 Testing simple completion...")
        simple_result = await self.test_simple_completion("Hello, how are you?", timeout=30.0)

        if not simple_result["success"]:
            print("\n❌ COMPLETION ISSUE DETECTED!")
            print(f"   - Error: {simple_result.get('error', 'Unknown')}")
            print("   - Model may not be loaded")
            print("   - Or server is overloaded")
            return

        # Test 3: Code generation
        print("\n🧪 Testing code generation...")
        code_results = await self.test_code_generation()

        # Summary
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 50)

        successful_tests = sum(1 for r in code_results if r["success"])
        total_tests = len(code_results)

        print(f"Server connectivity: {'✅ OK' if server_ok else '❌ FAILED'}")
        print(f"Simple completion: {'✅ OK' if simple_result['success'] else '❌ FAILED'}")
        print(f"Code generation: {successful_tests}/{total_tests} tests passed")

        if server_ok and simple_result["success"] and successful_tests > 0:
            print("\n🎉 SERVER IS WORKING!")
            print("   - LM Studio server is functional")
            print("   - Model is responding")
            print("   - Issue is in CodeConductor layer")
        else:
            print("\n⚠️  SERVER ISSUES DETECTED!")
            print("   - Check LM Studio configuration")
            print("   - Ensure models are loaded")
            print("   - Restart LM Studio if needed")


async def main():
    """Run diagnostic"""
    tester = DirectLMStudioTester()
    await tester.run_diagnostic()


if __name__ == "__main__":
    asyncio.run(main())
