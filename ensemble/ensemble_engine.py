"""
Main Ensemble Engine for CodeConductor

Orchestrates multiple LLMs for consensus-based code generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .model_manager import ModelManager
from .query_dispatcher import QueryDispatcher, QueryResult
from .consensus_calculator import ConsensusCalculator, ConsensusResult

logger = logging.getLogger(__name__)

@dataclass
class EnsembleRequest:
    task_description: str
    context: Optional[Dict[str, Any]] = None
    min_models: int = 2
    timeout: float = 30.0

@dataclass
class EnsembleResponse:
    consensus: Dict[str, Any]
    confidence: float
    disagreements: List[str]
    model_responses: List[QueryResult]
    execution_time: float

class EnsembleEngine:
    """Main ensemble engine that coordinates multiple LLMs."""
    
    def __init__(self, min_confidence: float = 0.7):
        self.model_manager = ModelManager()
        self.consensus_calculator = ConsensusCalculator(min_confidence)
        self.min_confidence = min_confidence
        
    async def initialize(self) -> bool:
        """Initialize the ensemble engine and discover models."""
        logger.info("Initializing Ensemble Engine...")
        
        try:
            # Discover available models
            models = await self.model_manager.discover_models()
            
            if not models:
                logger.error("No models discovered")
                return False
            
            # Perform health checks on discovered models
            health_tasks = []
            for model_id in models.keys():
                task = self.model_manager.health_check(model_id)
                health_tasks.append(task)
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            online_models = 0
            for model_id, result in zip(models.keys(), health_results):
                if isinstance(result, bool) and result:
                    online_models += 1
                    logger.info(f"Model {model_id} is online")
                else:
                    logger.warning(f"Model {model_id} failed health check")
            
            logger.info(f"Ensemble Engine initialized with {online_models} online models")
            return online_models >= 2
            
        except Exception as e:
            logger.error(f"Failed to initialize Ensemble Engine: {e}")
            return False
    
    async def process_request(self, request: EnsembleRequest) -> EnsembleResponse:
        """Process a task request through the ensemble."""
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Processing ensemble request: {request.task_description[:50]}...")
        
        try:
            # Get available models
            available_model_ids = self.model_manager.get_available_models(request.min_models)
            
            if len(available_model_ids) < request.min_models:
                raise Exception(f"Insufficient models available: {len(available_model_ids)} < {request.min_models}")
            
            # Get model objects
            models = {
                model_id: self.model_manager.models[model_id]
                for model_id in available_model_ids
            }
            
            # Dispatch queries in parallel
            async with QueryDispatcher(timeout=request.timeout) as dispatcher:
                results = await dispatcher.dispatch_parallel(
                    models, 
                    request.task_description, 
                    request.context
                )
            
            # Calculate consensus
            consensus_result = self.consensus_calculator.calculate_consensus(results)
            
            # Update model statistics
            for result in results:
                if result.success:
                    self.model_manager.update_model_stats(
                        result.model_id, 
                        True, 
                        result.response_time
                    )
                else:
                    self.model_manager.update_model_stats(
                        result.model_id, 
                        False, 
                        result.response_time
                    )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            response = EnsembleResponse(
                consensus=consensus_result.consensus,
                confidence=consensus_result.confidence,
                disagreements=consensus_result.disagreements,
                model_responses=results,
                execution_time=execution_time
            )
            
            logger.info(f"Ensemble request completed in {execution_time:.2f}s with confidence {consensus_result.confidence:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Ensemble request failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return EnsembleResponse(
                consensus={},
                confidence=0.0,
                disagreements=[f"Request failed: {str(e)}"],
                model_responses=[],
                execution_time=execution_time
            )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "total_models": len(self.model_manager.models),
            "online_models": 0,
            "models": {}
        }
        
        for model_id, model in self.model_manager.models.items():
            model_status = {
                "name": model.name,
                "status": model.status.value,
                "response_time": model.response_time,
                "success_rate": model.success_rate,
                "capabilities": model.capabilities
            }
            
            status["models"][model_id] = model_status
            
            if model.status.value == "online":
                status["online_models"] += 1
        
        return status
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all models."""
        health_tasks = []
        model_ids = list(self.model_manager.models.keys())
        
        for model_id in model_ids:
            task = self.model_manager.health_check(model_id)
            health_tasks.append(task)
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        health_status = {}
        for model_id, result in zip(model_ids, results):
            if isinstance(result, bool):
                health_status[model_id] = result
            else:
                health_status[model_id] = False
        
        return health_status 