#!/usr/bin/env python3
"""
Project Analyzer for CodeConductor MVP.

This module provides comprehensive project analysis including:
- FastAPI route scanning
- Database schema introspection
- Code quality analysis
- AI-powered recommendations
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RouteInfo:
    """Information about a FastAPI route"""
    method: str
    path: str
    function: str
    file: str
    parameters: Optional[List[str]] = None
    response_model: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    columns: List[Dict[str, str]]
    primary_key: Optional[str] = None
    foreign_keys: Optional[List[Dict[str, str]]] = None


class ProjectAnalyzer:
    """
    Analyzes codebases for FastAPI routes, database schemas, and code quality.
    
    This is the MVP version focusing on basic scanning capabilities.
    Future versions will include AI-powered analysis and recommendations.
    """
    
    def __init__(self):
        self.project_path: Optional[Path] = None
        self.python_files: List[Path] = []
        self.routes: List[RouteInfo] = []
        self.schema: Dict[str, Any] = {"tables": []}
        
    def scan_fastapi_routes(self, project_path: str) -> List[Dict[str, Any]]:
        """
        Scan a project for FastAPI routes.
        
        Args:
            project_path: Path to the project root directory
            
        Returns:
            List of route information dictionaries
        """
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
            
        logger.info(f"🔍 Scanning FastAPI routes in: {project_path}")
        
        # Find all Python files (exclude venv, site-packages, etc.)
        self.python_files = []
        for file_path in self.project_path.rglob("*.py"):
            # Skip virtual environments and external packages
            if any(exclude in str(file_path) for exclude in [
                "venv", "site-packages", "node_modules", "__pycache__", 
                ".git", ".pytest_cache", ".mypy_cache"
            ]):
                continue
            self.python_files.append(file_path)
        
        logger.info(f"📁 Found {len(self.python_files)} Python files (filtered)")
        
        # Scan each file for FastAPI routes
        for file_path in self.python_files:
            try:
                self._scan_file_for_routes(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Error scanning {file_path}: {e}")
                
        # Convert to dictionary format for JSON serialization
        routes_dict = []
        for route in self.routes:
            routes_dict.append({
                "method": route.method,
                "path": route.path,
                "function": route.function,
                "file": str(route.file),
                "parameters": route.parameters,
                "response_model": route.response_model
            })
            
        logger.info(f"✅ Found {len(routes_dict)} FastAPI routes")
        return routes_dict
    
    def _scan_file_for_routes(self, file_path: Path):
        """Scan a single Python file for FastAPI routes"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the Python file
            tree = ast.parse(content)
            
            # Look for FastAPI route decorators
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has route decorators
                    for decorator in node.decorator_list:
                        if self._is_route_decorator(decorator):
                            route_info = self._extract_route_info(node, decorator, file_path)
                            if route_info:
                                self.routes.append(route_info)
                                
        except Exception as e:
            logger.warning(f"⚠️ Error parsing {file_path}: {e}")
    
    def _is_route_decorator(self, decorator: ast.expr) -> bool:
        """Check if a decorator is a FastAPI route decorator"""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                # Check for @app.get, @app.post, etc.
                if decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                    # Additional check: make sure it's likely a FastAPI app
                    if hasattr(decorator.func, 'value'):
                        if isinstance(decorator.func.value, ast.Name):
                            # Common FastAPI app variable names
                            if decorator.func.value.id in ['app', 'api', 'router', 'fastapi_app']:
                                return True
            elif isinstance(decorator.func, ast.Name):
                # Check for @get, @post, etc. (direct imports)
                if decorator.func.id in ['get', 'post', 'put', 'delete', 'patch']:
                    return True
        return False
    
    def _extract_route_info(self, func_node: ast.FunctionDef, decorator: ast.Call, file_path: Path) -> Optional[RouteInfo]:
        """Extract route information from a function and its decorator"""
        try:
            # Get HTTP method
            if isinstance(decorator.func, ast.Attribute):
                method = decorator.func.attr.upper()
            else:
                method = decorator.func.id.upper()
            
            # Get path from decorator arguments
            path = "/"
            if decorator.args:
                path_arg = decorator.args[0]
                if isinstance(path_arg, ast.Constant):
                    path = path_arg.value
                elif isinstance(path_arg, ast.Str):  # Python < 3.8
                    path = path_arg.s
            
            # Get function parameters
            parameters = []
            for arg in func_node.args.args:
                if arg.arg != 'self':
                    parameters.append(arg.arg)
            
            return RouteInfo(
                method=method,
                path=path,
                function=func_node.name,
                file=file_path.relative_to(self.project_path),
                parameters=parameters if parameters else None
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting route info: {e}")
            return None
    
    def introspect_postgresql(self, db_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Introspect PostgreSQL database schema.
        
        Args:
            db_url: Database connection URL (optional for MVP)
            
        Returns:
            Database schema information
        """
        logger.info("🗄️ Starting PostgreSQL introspection")
        
        # For MVP, return mock schema data
        # In production, this would connect to the actual database
        mock_schema = {
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "id", "type": "INTEGER PRIMARY KEY"},
                        {"name": "email", "type": "VARCHAR(255) UNIQUE"},
                        {"name": "password_hash", "type": "VARCHAR(255)"},
                        {"name": "created_at", "type": "TIMESTAMP"}
                    ]
                },
                {
                    "name": "posts", 
                    "columns": [
                        {"name": "id", "type": "INTEGER PRIMARY KEY"},
                        {"name": "user_id", "type": "INTEGER REFERENCES users(id)"},
                        {"name": "title", "type": "VARCHAR(255)"},
                        {"name": "content", "type": "TEXT"},
                        {"name": "created_at", "type": "TIMESTAMP"}
                    ]
                }
            ],
            "total_tables": 2,
            "total_columns": 9
        }
        
        self.schema = mock_schema
        logger.info(f"✅ Introspected {mock_schema['total_tables']} tables")
        return mock_schema
    
    def generate_report(self, routes: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive project analysis report.
        
        Args:
            routes: List of FastAPI routes
            schema: Database schema information
            
        Returns:
            Complete analysis report
        """
        logger.info("📊 Generating project analysis report")
        
        # Basic metrics
        total_routes = len(routes)
        total_tables = len(schema.get("tables", []))
        total_files = len(self.python_files)
        
        # Route analysis
        route_methods = {}
        for route in routes:
            method = route["method"]
            route_methods[method] = route_methods.get(method, 0) + 1
        
        # Generate AI recommendations (mock for MVP)
        ai_recommendations = self._generate_ai_recommendations(routes, schema)
        
        report = {
            "project_path": str(self.project_path) if self.project_path else None,
            "files_analyzed": total_files,
            "routes": routes,
            "schema": schema,
            "metrics": {
                "total_routes": total_routes,
                "total_tables": total_tables,
                "route_methods": route_methods,
                "files_analyzed": total_files
            },
            "ai_recommendations": ai_recommendations,
            "analysis_timestamp": str(Path().cwd()),
            "version": "MVP-1.0"
        }
        
        logger.info(f"✅ Generated report with {total_routes} routes, {total_tables} tables")
        return report
    
    def _generate_ai_recommendations(self, routes: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate AI-powered recommendations (mock for MVP)"""
        recommendations = []
        
        # Route-based recommendations
        if len(routes) == 0:
            recommendations.append({
                "category": "Architecture",
                "message": "No FastAPI routes found. Consider adding API endpoints for your application."
            })
        
        if len(routes) > 20:
            recommendations.append({
                "category": "Maintainability", 
                "message": "Large number of routes detected. Consider organizing into separate router modules."
            })
        
        # Security recommendations
        auth_routes = [r for r in routes if "auth" in r["path"].lower() or "login" in r["path"].lower()]
        if len(auth_routes) == 0:
            recommendations.append({
                "category": "Security",
                "message": "No authentication routes found. Consider implementing user authentication."
            })
        
        # Database recommendations
        if len(schema.get("tables", [])) == 0:
            recommendations.append({
                "category": "Database",
                "message": "No database tables found. Consider adding database models for data persistence."
            })
        
        # Performance recommendations
        if len(routes) > 10:
            recommendations.append({
                "category": "Performance",
                "message": "Consider implementing caching for frequently accessed endpoints."
            })
        
        return recommendations
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics (placeholder for future implementation)"""
        return {
            "complexity": "medium",
            "maintainability": "good",
            "test_coverage": "unknown",
            "documentation": "partial"
        }
    
    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """
        Export analysis report in specified format.
        
        Args:
            report: Analysis report
            format: Export format ("json" or "csv")
            
        Returns:
            Exported report as string
        """
        if format.lower() == "json":
            return json.dumps(report, indent=2)
        elif format.lower() == "csv":
            # Simple CSV export for routes and tables
            lines = ["Type,Name,Details"]
            
            for route in report.get("routes", []):
                lines.append(f"Route,{route['method']} {route['path']},{route['function']}")
            
            for table in report.get("schema", {}).get("tables", []):
                lines.append(f"Table,{table['name']},{len(table['columns'])} columns")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions for easy usage
def analyze_project(project_path: str) -> Dict[str, Any]:
    """Convenience function to analyze a project"""
    analyzer = ProjectAnalyzer()
    routes = analyzer.scan_fastapi_routes(project_path)
    schema = analyzer.introspect_postgresql()
    return analyzer.generate_report(routes, schema)


def scan_routes_only(project_path: str) -> List[Dict[str, Any]]:
    """Convenience function to scan only FastAPI routes"""
    analyzer = ProjectAnalyzer()
    return analyzer.scan_fastapi_routes(project_path) 