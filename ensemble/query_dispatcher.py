"""
Query Dispatcher for LLM Ensemble

Handles parallel query execution with timeout and error handling.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from .model_manager import ModelInfo

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    model_id: str
    response: str
    success: bool
    response_time: float
    error: Optional[str] = None

class QueryDispatcher:
    """Dispatches queries to multiple LLM models in parallel."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def dispatch_parallel(
        self, 
        models: Dict[str, ModelInfo], 
        prompt: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Dispatch query to multiple models in parallel."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        tasks = []
        for model_id, model in models.items():
            task = self._query_model(model_id, model, prompt, task_context)
            tasks.append(task)
        
        # Execute all queries in parallel with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout after {self.timeout}s")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = [QueryResult(
                model_id="timeout",
                response="",
                success=False,
                response_time=self.timeout,
                error="Query timeout"
            )]
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(QueryResult(
                    model_id="error",
                    response="",
                    success=False,
                    response_time=0.0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _query_model(
        self, 
        model_id: str, 
        model: ModelInfo, 
        prompt: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Query a single model."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if "ollama" in model_id:
                response = await self._query_ollama(model, prompt, task_context)
            elif "lm_studio" in model_id:
                response = await self._query_lm_studio(model, prompt, task_context)
            else:
                raise ValueError(f"Unknown model type: {model_id}")
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return QueryResult(
                model_id=model_id,
                response=response,
                success=True,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Query failed for {model_id}: {e}")
            
            return QueryResult(
                model_id=model_id,
                response="",
                success=False,
                response_time=response_time,
                error=str(e)
            )
    
    async def _query_ollama(
        self, 
        model: ModelInfo, 
        prompt: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Query Ollama model."""
        # Format prompt for code generation
        formatted_prompt = self._format_code_prompt(prompt, task_context)
        
        payload = {
            "model": model.name,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for code generation
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        async with self.session.post(model.endpoint, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama API error: {response.status}")
            
            data = await response.json()
            return data.get('response', '')
    
    async def _query_lm_studio(
        self, 
        model: ModelInfo, 
        prompt: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Query LM Studio model."""
        # Format prompt for code generation
        formatted_prompt = self._format_code_prompt(prompt, task_context)
        
        payload = {
            "model": model.name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert software developer. Generate clean, efficient, and well-documented code."
                },
                {
                    "role": "user", 
                    "content": formatted_prompt
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        async with self.session.post(model.endpoint, json=payload) as response:
            if response.status != 200:
                raise Exception(f"LM Studio API error: {response.status}")
            
            data = await response.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def _format_code_prompt(self, prompt: str, task_context: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt for code generation tasks."""
        formatted = f"""You are an expert software developer. Your task is to:

{prompt}

Please provide:
1. A clear approach to solve this problem
2. The specific files that need to be created or modified
3. Any dependencies that need to be added
4. An assessment of complexity (low/medium/high)

Respond in JSON format:
{{
    "approach": "Description of solution approach",
    "files_needed": ["file1.py", "file2.py"],
    "dependencies": ["package1", "package2"],
    "complexity": "low|medium|high",
    "reasoning": "Brief explanation of your approach"
}}"""

        if task_context:
            formatted += f"\n\nContext:\n{json.dumps(task_context, indent=2)}"
        
        return formatted 