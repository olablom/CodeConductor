#!/usr/bin/env python3
"""
Debug script for ProjectAnalyzer
"""

import ast
from pathlib import Path
from analysis.project_analyzer import ProjectAnalyzer

def debug_ast_parsing():
    """Debug AST parsing of test_fastapi_app.py"""
    print("🔍 Debugging AST parsing...")
    
    try:
        with open('test_fastapi_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 File length: {len(content)} characters")
        print(f"🔍 Contains '@app.get': {'@app.get' in content}")
        print(f"🔍 Contains 'def ': {'def ' in content}")
        
        # Parse AST
        tree = ast.parse(content)
        print(f"✅ AST parsing successful")
        
        # Find all function definitions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"📝 Found {len(functions)} function definitions")
        
        # Show first few function names
        for i, func in enumerate(functions[:5]):
            print(f"  {i+1}. {func.name}")
            print(f"     Decorators: {len(func.decorator_list)}")
            
            # Check decorators
            for j, decorator in enumerate(func.decorator_list):
                print(f"       Decorator {j+1}: {type(decorator).__name__}")
                if isinstance(decorator, ast.Call):
                    if hasattr(decorator.func, 'attr'):
                        print(f"         Method: {decorator.func.attr}")
                    if hasattr(decorator.func, 'value') and hasattr(decorator.func.value, 'id'):
                        print(f"         Object: {decorator.func.value.id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        return False

def debug_project_analyzer():
    """Debug ProjectAnalyzer step by step"""
    print("\n🔧 Debugging ProjectAnalyzer...")
    
    analyzer = ProjectAnalyzer()
    
    # Test scanning
    try:
        routes = analyzer.scan_fastapi_routes('.')
        print(f"✅ Scan completed: {len(routes)} routes found")
        print(f"📁 Files scanned: {len(analyzer.python_files)}")
        
        # Show some routes
        for i, route in enumerate(routes[:5]):
            print(f"  Route {i+1}: {route['method']} {route['path']} - {route['function']}")
            
        return routes
        
    except Exception as e:
        print(f"❌ Error in ProjectAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("🚀 Starting ProjectAnalyzer debug...")
    
    # Step 1: Debug AST parsing
    ast_ok = debug_ast_parsing()
    
    if ast_ok:
        # Step 2: Debug ProjectAnalyzer
        routes = debug_project_analyzer()
        
        if routes:
            print(f"\n🎉 SUCCESS! Found {len(routes)} routes")
        else:
            print("\n❌ No routes found")
    else:
        print("\n❌ AST parsing failed") 