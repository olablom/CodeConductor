# --- HARD CPU-ONLY GUARD (måste ligga allra först) ---
import logging  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from urllib.parse import quote_plus  # noqa: E402

import requests  # noqa: E402

if (
    os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
    or os.getenv("CC_GPU_DISABLED", "0") == "1"
    or os.getenv("CC_ULTRA_MOCK", "0") == "1"
):
    # Tvinga RAG till mock-läge
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VLLM_NO_CUDA", "1")
    os.environ["CC_GPU_DISABLED"] = "1"
    os.environ["CC_TESTING_MODE"] = "1"
    os.environ["CC_ULTRA_MOCK"] = "1"
# ------------------------------------------------------

"""
RAG (Retrieval-Augmented Generation) System for CodeConductor

This module provides context-aware code generation by retrieving relevant
documentation, code examples, and patterns based on task descriptions.
"""

# Graceful fallback for optional dependencies
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    RAG_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"RAG dependencies not available: {e}")
    logger.warning(
        "RAG functionality will be disabled. Install with: pip install sentence-transformers"
    )
    RAG_AVAILABLE = False

    # Create dummy classes for graceful fallback
    class Chroma:
        def __init__(self, *args, **kwargs):
            pass

        def add_documents(self, *args, **kwargs):
            pass

        def persist(self):
            pass

        def similarity_search_with_relevance_scores(self, *args, **kwargs):
            return []

    class HuggingFaceEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, docs):
            return []

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


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

        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("RAG system forced to mock mode for testing")
            self.embedding_model = None
            self.text_splitter = None
            self.vector_store = None
            return

        # Check if RAG dependencies are available
        if not RAG_AVAILABLE:
            logger.warning(
                "RAG system initialized in disabled mode - dependencies not available"
            )
            self.embedding_model = None
            self.text_splitter = None
            self.vector_store = None
            return

        # Initialize embedding model (using local model for privacy/cost)
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},  # Use CPU for compatibility
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

        # Initialize text splitter
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            self.text_splitter = None

        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("Vector store initialization skipped - test mode")
            self.vector_store = None
            return

        if not RAG_AVAILABLE or not self.embedding_model:
            logger.warning("Vector store initialization skipped - RAG not available")
            self.vector_store = None
            return

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
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("Project indexing skipped - test mode")
            return

        if not self.vector_store or not self.text_splitter:
            logger.warning("Project indexing skipped - RAG not available")
            return

        documents = []

        # Index Python files
        for py_file in self.project_root.rglob("*.py"):
            if "venv" not in str(py_file) and "node_modules" not in str(py_file):
                try:
                    with open(py_file, encoding="utf-8") as f:
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
                    logger.warning(f"Failed to index {py_file}: {e}")

        # Index markdown files
        for md_file in self.project_root.rglob("*.md"):
            if "venv" not in str(md_file) and "node_modules" not in str(md_file):
                try:
                    with open(md_file, encoding="utf-8") as f:
                        content = f.read()

                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(md_file),
                            "type": "documentation",
                            "filename": md_file.name,
                        },
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to index {md_file}: {e}")

        # Add documents to vector store
        if documents:
            try:
                chunks = self.text_splitter.split_documents(documents)
                self.vector_store.add_documents(chunks)
                self.vector_store.persist()
                logger.info(f"Indexed {len(documents)} files into vector store")
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}")
        else:
            logger.warning("No documents found to index")

    def retrieve_context(
        self, task_description: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant context for a task description.

        Args:
            task_description: Description of the coding task
            k: Number of relevant documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("Context retrieval returning mock data - test mode")
            return [
                {
                    "content": f"[MOCK] Test context for: {task_description}",
                    "metadata": {
                        "source": "mock",
                        "type": "test_data",
                        "filename": "mock_context.py",
                    },
                    "relevance_score": 0.85,
                    "source": "mock",
                }
                for _ in range(min(k, 3))
            ]

        context_docs = []

        # Retrieve from local vectorstore
        if self.vector_store and RAG_AVAILABLE:
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
        else:
            logger.info("Local context retrieval skipped - RAG not available")

        # Fetch external context (respect ALLOW_NET)
        try:
            if os.getenv("ALLOW_NET", "0").strip() in {"1", "true", "yes"}:
                external_context = self.fetch_external_context(task_description)
            else:
                external_context = []
        except Exception:
            external_context = []

        for i, content in enumerate(external_context):
            context_docs.append(
                {
                    "content": content,
                    "metadata": {
                        "source": "external",
                        "type": "external_context",
                        "filename": f"external_{i + 1}",
                    },
                    "relevance_score": 0.8,
                    "source": "external",
                }
            )

        logger.info(f"Total context: {len(context_docs)} documents (local + external)")
        return context_docs

    def fetch_external_context(
        self, query: str, source: str = "stackoverflow", max_results: int = 3
    ) -> list[str]:
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
    ) -> list[str]:
        """Fetch context from Stack Overflow API with optional key and disk cache."""
        try:
            import hashlib
            import json as _json
            import random as _random
            import re
            import time as _time
            from pathlib import Path as _Path

            # Env controls
            timeout_s = float(os.getenv("NET_TIMEOUT_S", "10") or "10")
            max_retries = max(0, int(os.getenv("NET_MAX_RETRIES", "2") or "2"))
            cache_ttl = max(
                0, int(os.getenv("NET_CACHE_TTL_SECONDS", "3600") or "3600")
            )
            pagesize = max(
                1, int(os.getenv("SO_PAGESIZE", str(max_results)) or str(max_results))
            )
            stack_key = (os.getenv("STACKEXCHANGE_KEY") or "").strip()
            cache_dir = _Path(os.getenv("NET_CACHE_DIR", "artifacts/net_cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Build URL
            encoded_query = quote_plus(query)
            base = (
                f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
                f"&q={encoded_query}&site=stackoverflow&filter=withbody&pagesize={pagesize}"
            )
            if stack_key:
                base += f"&key={quote_plus(stack_key)}"

            # Cache key
            ck = hashlib.sha256(base.encode("utf-8")).hexdigest()
            cpath = cache_dir / f"so_{ck}.json"

            # Try cache first
            if cache_ttl > 0 and cpath.exists():
                try:
                    stat = cpath.stat()
                    if (_time.time() - stat.st_mtime) <= cache_ttl:
                        cached = _json.loads(cpath.read_text(encoding="utf-8"))
                        if isinstance(cached, list):
                            return cached
                except Exception:
                    pass

            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(base, timeout=timeout_s)
                    if response.status_code == 200:
                        data = response.json()
                        results = []
                        for item in data.get("items", []):
                            title = item.get("title", "")
                            body = item.get("body", "")
                            body = re.sub(r"<[^>]+>", "", body)
                            body = body[:500]
                            results.append(f"Stack Overflow: {title}\n\n{body}...")
                        # Write cache
                        try:
                            cpath.write_text(
                                _json.dumps(results, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        logger.info(f"Fetched {len(results)} Stack Overflow results")
                        return results
                    elif response.status_code in (429, 502, 503):
                        # backoff with jitter
                        delay = min(
                            60.0, (1.0 * (2**attempt)) + _random.uniform(0.0, 0.5)
                        )
                        _time.sleep(delay)
                        last_error = RuntimeError(f"HTTP {response.status_code}")
                    else:
                        logger.warning(
                            f"Stack Overflow API returned status {response.status_code}"
                        )
                        return []
                except Exception as e:
                    last_error = e
                    delay = min(10.0, 0.5 * (2**attempt) + _random.uniform(0.0, 0.25))
                    _time.sleep(delay)
            if last_error:
                logger.warning(f"Stack Overflow fetch failed: {last_error}")
            return []
        except Exception as e:
            logger.error(f"Error fetching Stack Overflow context: {e}")
            return []

    def _fetch_github_context(self, query: str, max_results: int = 3) -> list[str]:
        """Fetch context from GitHub (placeholder for future implementation)."""
        # TODO: Implement GitHub API integration
        logger.info("GitHub context fetching not yet implemented")
        return []

    def _fetch_documentation_context(
        self, query: str, max_results: int = 3
    ) -> list[str]:
        """Fetch context from documentation sites (placeholder)."""
        # TODO: Implement documentation site scraping
        logger.info("Documentation context fetching not yet implemented")
        return []

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any] = None):
        """
        Add a document to the vector database.

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata dictionary
        """
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info(f"Document addition skipped - test mode (would add: {doc_id})")
            return

        if not self.vector_store or not RAG_AVAILABLE:
            logger.warning("Document addition skipped - RAG not available")
            return

        try:
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata or {"id": doc_id, "type": "document"},
            )

            # Add to vector store
            self.vector_store.add_documents([doc])
            # Fix persist method call
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
            elif hasattr(self.vector_store, "_persist"):
                self.vector_store._persist()

            logger.info(f"Added document '{doc_id}' to context database")

        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for relevant documents in the vector database.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documents with metadata
        """
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("Search returning mock data - test mode")
            return [
                {
                    "id": f"mock_{i+1}",
                    "content": f"[MOCK] Test result for query: {query}",
                    "metadata": {
                        "source": "mock",
                        "type": "test_data",
                        "filename": f"mock_result_{i+1}.py",
                    },
                    "relevance_score": 0.8 - (i * 0.1),
                }
                for i in range(min(top_k, 3))
            ]

        if not self.vector_store:
            logger.warning("Search skipped - RAG not available")
            return []

        try:
            # Search vector store
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )

            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        "id": doc.metadata.get("id", "unknown"),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": score,
                    }
                )

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def search_external(self, query: str, max_results: int = 3) -> list[str]:
        """
        Search external sources (Stack Overflow, etc.).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of external content strings
        """
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("External search returning mock data - test mode")
            return [
                f"[MOCK] External result for query: {query} - Result {i+1}"
                for i in range(min(max_results, 2))
            ]

        try:
            return self.fetch_external_context(query, max_results=max_results)
        except Exception as e:
            logger.error(f"Error searching external sources: {e}")
            return []

    def get_context(self, task_description: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Get context for a task description (alias for retrieve_context).

        Args:
            task_description: Description of the coding task
            k: Number of relevant documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        return self.retrieve_context(task_description, k)

    def add_pattern_to_context(self, pattern: dict[str, Any]):
        """
        Add a successful pattern to the context database.

        Args:
            pattern: Pattern data from learning system
        """
        # FORCE MOCK MODE IN TEST ENVIRONMENT
        if (
            os.getenv("CC_HARD_CPU_ONLY", "0") == "1"
            or os.getenv("CC_GPU_DISABLED", "0") == "1"
            or os.getenv("CC_ULTRA_MOCK", "0") == "1"
        ):
            logger.info("Pattern addition skipped - test mode")
            return

        if not self.vector_store or not RAG_AVAILABLE:
            logger.info("Pattern addition skipped - RAG not available")
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
            # Fix persist method call
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
            elif hasattr(self.vector_store, "_persist"):
                self.vector_store._persist()

            logger.info("Added pattern to context database")

        except Exception as e:
            logger.error(f"Error adding pattern to context: {e}")

    def format_context_for_prompt(self, context_docs: list[dict[str, Any]]) -> str:
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

            context_parts.append(
                f"""
### Context {i} (Score: {doc["relevance_score"]:.3f})
**Source:** {metadata.get("filename", metadata.get("source", "Unknown"))}
**Type:** {metadata.get("type", "Unknown")}

{content}
...
"""
            )

        return "\n".join(context_parts)

    def get_context_summary(self, task_description: str) -> dict[str, Any]:
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
            "avg_relevance": (
                sum(doc["relevance_score"] for doc in context_docs) / len(context_docs)
                if context_docs
                else 0
            ),
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
