"""
Context module for CodeConductor MVP.

This module provides RAG (Retrieval-Augmented Generation) capabilities including:
- Vector database management
- Document retrieval
- Context-aware code generation
"""

from .rag_system import RAGSystem

__all__ = ["RAGSystem"]
