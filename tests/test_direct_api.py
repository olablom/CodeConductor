#!/usr/bin/env python3
"""
Direct API test for LM Studio

Tests direct API calls to verify the model is working.
"""

import asyncio
import json

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_direct_api():
    """Test direct API call to LM Studio"""

    print("🧪 Testing Direct LM Studio API")
    print("=" * 40)

    url = "http://localhost:1234/v1/chat/completions"

    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "Hello! Please respond with a simple greeting."}
        ],
        "max_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    print(f"📝 Sending request to: {url}")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                print(f"📊 Response status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    print(f"📊 Response data: {json.dumps(data, indent=2)}")

                    if "choices" in data and data["choices"]:
                        content = (
                            data["choices"][0].get("message", {}).get("content", "")
                        )
                        print(f"✅ Content: {content}")
                        return True
                    else:
                        print("❌ No content in response")
                        return False
                else:
                    print(f"❌ Error status: {response.status}")
                    return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_direct_api())
    if success:
        print("\n🎉 Direct API test PASSED!")
    else:
        print("\n💥 Direct API test FAILED!")
