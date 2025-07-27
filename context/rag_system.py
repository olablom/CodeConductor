"""
RAG (Retrieval-Augmented Generation) System for CodeConductor

This module provides context-aware code generation by retrieving relevant
documentation, code examples, and patterns based on task descriptions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system for CodeConductor.
    
    Provides relevant context to LLM ensemble based on task descriptions
    by searching through project documentation, code examples, and patterns.
    """
    
    def __init__(self, project_root: str = ".", vector_db_path: str = "vector_db"):
        """
        Initialize RAG system.
        
        Args:
            project_root: Root directory of the project
            vector_db_path: Path to store vector database
        """
        self.project_root = Path(project_root)
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model (using local model for privacy/cost)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        try:
            if (self.vector_db_path / "chroma.sqlite3").exists():
                logger.info("Loading existing vector database...")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_db_path),
                    embedding_function=self.embedding_model
                )
            else:
                logger.info("Creating new vector database...")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_db_path),
                    embedding_function=self.embedding_model
                )
                # Initial indexing of project files
                self._index_project_files()
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None
    
    def _index_project_files(self):
        """Index project files for context retrieval."""
        if not self.vector_store:
            return
        
        documents = []
        
        # Index Python files
        for py_file in self.project_root.rglob("*.py"):
            if "venv" not in str(py_file) and "node_modules" not in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(py_file),
                            "type": "python_code",
                            "filename": py_file.name
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")
        
        # Index README and documentation
        for doc_file in self.project_root.glob("*.md"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(doc_file),
                        "type": "documentation",
                        "filename": doc_file.name
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Could not read {doc_file}: {e}")
        
        # Index JSON files (patterns, configs)
        for json_file in self.project_root.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert JSON to readable text
                content = json.dumps(data, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(json_file),
                        "type": "json_data",
                        "filename": json_file.name
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")
        
        if documents:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            
            logger.info(f"Indexed {len(split_docs)} document chunks")
    
    def retrieve_context(self, task_description: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a task description.
        
        Args:
            task_description: Description of the coding task
            k: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.vector_store:
            logger.warning("Vector store not available, returning empty context")
            return []
        
        try:
            # Search for relevant documents
            results = self.vector_store.similarity_search_with_relevance_scores(
                task_description, k=k
            )
            
            # Format results
            context_docs = []
            for doc, score in results:
                context_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                })
            
            logger.info(f"Retrieved {len(context_docs)} relevant documents")
            return context_docs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def add_pattern_to_context(self, pattern: Dict[str, Any]):
        """
        Add a successful pattern to the context database.
        
        Args:
            pattern: Pattern data from learning system
        """
        if not self.vector_store:
            return
        
        try:
            # Create document from pattern
            content = f"""
Task: {pattern.get('task_description', '')}
Prompt: {pattern.get('prompt', '')}
Code: {pattern.get('code', '')}
Validation Score: {pattern.get('validation', {}).get('score', 0)}
Model Used: {pattern.get('model_used', 'Unknown')}
Rating: {pattern.get('user_rating', 0)}/5
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "learning_pattern",
                    "type": "successful_pattern",
                    "task": pattern.get('task_description', ''),
                    "score": pattern.get('validation', {}).get('score', 0),
                    "rating": pattern.get('user_rating', 0),
                    "model": pattern.get('model_used', 'Unknown'),
                    "timestamp": pattern.get('timestamp', '')
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            self.vector_store.persist()
            
            logger.info("Added pattern to context database")
            
        except Exception as e:
            logger.error(f"Error adding pattern to context: {e}")
    
    def format_context_for_prompt(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context for inclusion in LLM prompts.
        
        Args:
            context_docs: Retrieved context documents
            
        Returns:
            Formatted context string
        """
        if not context_docs:
            return ""
        
        context_parts = ["## Relevant Context:"]
        
        for i, doc in enumerate(context_docs, 1):
            metadata = doc["metadata"]
            content = doc["content"][:500]  # Limit content length
            
            context_parts.append(f"""
### Context {i} (Score: {doc['relevance_score']:.3f})
**Source:** {metadata.get('filename', metadata.get('source', 'Unknown'))}
**Type:** {metadata.get('type', 'Unknown')}

{content}
...
""")
        
        return "\n".join(context_parts)
    
    def get_context_summary(self, task_description: str) -> Dict[str, Any]:
        """
        Get a summary of available context for a task.
        
        Args:
            task_description: Task description
            
        Returns:
            Summary of context availability
        """
        context_docs = self.retrieve_context(task_description, k=3)
        
        return {
            "context_available": len(context_docs) > 0,
            "context_count": len(context_docs),
            "avg_relevance": sum(doc["relevance_score"] for doc in context_docs) / len(context_docs) if context_docs else 0,
            "context_types": list(set(doc["metadata"].get("type", "Unknown") for doc in context_docs))
        }


# Global instance for easy access
rag_system = RAGSystem() 