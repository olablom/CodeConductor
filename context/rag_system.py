"""
RAG (Retrieval-Augmented Generation) System for CodeConductor

This module provides context-aware code generation by retrieving relevant
documentation, code examples, and patterns based on task descriptions.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from urllib.parse import quote_plus

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
            model_kwargs={"device": "cpu"},  # Use CPU for compatibility
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
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
                    embedding_function=self.embedding_model,
                )
            else:
                logger.info("Creating new vector database...")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_db_path),
                    embedding_function=self.embedding_model,
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
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(py_file),
                            "type": "python_code",
                            "filename": py_file.name,
                        },
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")

        # Index README and documentation
        for doc_file in self.project_root.glob("*.md"):
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(doc_file),
                        "type": "documentation",
                        "filename": doc_file.name,
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Could not read {doc_file}: {e}")

        # Index JSON files (patterns, configs)
        for json_file in self.project_root.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert JSON to readable text
                content = json.dumps(data, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(json_file),
                        "type": "json_data",
                        "filename": json_file.name,
                    },
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

    def retrieve_context(
        self, task_description: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a task description.

        Args:
            task_description: Description of the coding task
            k: Number of relevant documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        context_docs = []

        # Retrieve from local vectorstore
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    task_description, k=k
                )

                for doc, score in results:
                    context_docs.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance_score": score,
                            "source": "local",
                        }
                    )

                logger.info(f"Retrieved {len(context_docs)} local documents")
            except Exception as e:
                logger.error(f"Error retrieving local context: {e}")

        # Fetch external context
        external_context = self.fetch_external_context(task_description)
        for i, content in enumerate(external_context):
            context_docs.append(
                {
                    "content": content,
                    "metadata": {
                        "source": "external",
                        "type": "external_context",
                        "filename": f"external_{i + 1}",
                    },
                    "relevance_score": 0.8,  # Default high score for external content
                    "source": "external",
                }
            )

        logger.info(f"Total context: {len(context_docs)} documents (local + external)")
        return context_docs

    def fetch_external_context(
        self, query: str, source: str = "stackoverflow", max_results: int = 3
    ) -> List[str]:
        """
        Fetch external context from various sources.

        Args:
            query: Search query
            source: Source to fetch from ("stackoverflow", "github", "docs")
            max_results: Maximum number of results to fetch

        Returns:
            List of context strings
        """
        try:
            if source == "stackoverflow":
                return self._fetch_stackoverflow_context(query, max_results)
            elif source == "github":
                return self._fetch_github_context(query, max_results)
            elif source == "docs":
                return self._fetch_documentation_context(query, max_results)
            else:
                logger.warning(f"Unknown external source: {source}")
                return []
        except Exception as e:
            logger.error(f"Error fetching external context from {source}: {e}")
            return []

    def _fetch_stackoverflow_context(
        self, query: str, max_results: int = 3
    ) -> List[str]:
        """Fetch context from Stack Overflow API."""
        try:
            # Use Stack Exchange API
            encoded_query = quote_plus(query)
            url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={encoded_query}&site=stackoverflow&filter=withbody&pagesize={max_results}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("items", []):
                    title = item.get("title", "")
                    body = item.get("body", "")
                    # Clean HTML tags (simple approach)
                    import re

                    body = re.sub(r"<[^>]+>", "", body)
                    body = body[:500]  # Limit length

                    result = f"Stack Overflow: {title}\n\n{body}..."
                    results.append(result)

                logger.info(f"Fetched {len(results)} Stack Overflow results")
                return results
            else:
                logger.warning(
                    f"Stack Overflow API returned status {response.status_code}"
                )
                return []

        except Exception as e:
            logger.error(f"Error fetching Stack Overflow context: {e}")
            return []

    def _fetch_github_context(self, query: str, max_results: int = 3) -> List[str]:
        """Fetch context from GitHub (placeholder for future implementation)."""
        # TODO: Implement GitHub API integration
        logger.info("GitHub context fetching not yet implemented")
        return []

    def _fetch_documentation_context(
        self, query: str, max_results: int = 3
    ) -> List[str]:
        """Fetch context from documentation sites (placeholder)."""
        # TODO: Implement documentation site scraping
        logger.info("Documentation context fetching not yet implemented")
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
Task: {pattern.get("task_description", "")}
Prompt: {pattern.get("prompt", "")}
Code: {pattern.get("code", "")}
Validation Score: {pattern.get("validation", {}).get("score", 0)}
Model Used: {pattern.get("model_used", "Unknown")}
Rating: {pattern.get("user_rating", 0)}/5
"""

            doc = Document(
                page_content=content,
                metadata={
                    "source": "learning_pattern",
                    "type": "successful_pattern",
                    "task": pattern.get("task_description", ""),
                    "score": pattern.get("validation", {}).get("score", 0),
                    "rating": pattern.get("user_rating", 0),
                    "model": pattern.get("model_used", "Unknown"),
                    "timestamp": pattern.get("timestamp", ""),
                },
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
### Context {i} (Score: {doc["relevance_score"]:.3f})
**Source:** {metadata.get("filename", metadata.get("source", "Unknown"))}
**Type:** {metadata.get("type", "Unknown")}

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
            "avg_relevance": sum(doc["relevance_score"] for doc in context_docs)
            / len(context_docs)
            if context_docs
            else 0,
            "context_types": list(
                set(doc["metadata"].get("type", "Unknown") for doc in context_docs)
            ),
        }

    def augment_prompt(
        self, task_description: str, include_external: bool = True
    ) -> str:
        """
        Augment a task description with relevant context.

        Args:
            task_description: Original task description
            include_external: Whether to include external context

        Returns:
            Augmented prompt with context
        """
        # Retrieve context
        context_docs = self.retrieve_context(task_description, k=5)

        if not context_docs:
            return task_description

        # Format context
        context_text = self.format_context_for_prompt(context_docs)

        # Create augmented prompt
        augmented_prompt = f"""{task_description}

## Relevant Context:
{context_text}

Please use the above context to generate high-quality, contextually appropriate code."""

        return augmented_prompt


# Global instance for easy access
rag_system = RAGSystem()
