#!/usr/bin/env python3
"""
Test file for Simple API Smoke Testing
"""

import pytest
import requests
import subprocess
import time
import json
from pathlib import Path

class TestSimpleAPI:
    """Test class for Simple API."""
    
    @pytest.fixture(autouse=True)
    def setup_api(self):
        """Setup API for testing."""
        # Start API in background
        self.api_process = subprocess.Popen([
            'python', 'simple_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for API to start
        time.sleep(3)
        
        yield
        
        # Cleanup
        self.api_process.terminate()
        self.api_process.wait()
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = requests.get("http://localhost:8000/")
        assert response.status_code == 200
        assert "Simple API is running" in response.json()["message"]
    
    def test_items_endpoint(self):
        """Test items endpoint."""
        response = requests.get("http://localhost:8000/items")
        assert response.status_code == 200
        assert "items" in response.json()
    
    def test_create_item(self):
        """Test creating an item."""
        item_data = {
            "name": "Test Item",
            "description": "A test item",
            "price": 10.99
        }
        
        response = requests.post(
            "http://localhost:8000/items",
            json=item_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Item"
        assert data["id"] == 1
    
    def test_get_item(self):
        """Test getting a specific item."""
        # First create an item
        item_data = {"name": "Test Item", "price": 5.99}
        create_response = requests.post(
            "http://localhost:8000/items",
            json=item_data
        )
        item_id = create_response.json()["id"]
        
        # Then get it
        response = requests.get(f"http://localhost:8000/items/{item_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Test Item"
    
    def test_health_endpoint(self):
        """Test health endpoint (should be added by pipeline)."""
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data

def test_api_file_structure():
    """Test that API file has correct structure."""
    with open('simple_api.py', 'r') as f:
        content = f.read()
    
    # Check for health endpoint
    assert "@app.get(\"/health\")" in content
    assert "def health_check()" in content
    assert "status" in content
    assert "healthy" in content

if __name__ == "__main__":
    pytest.main([__file__]) 