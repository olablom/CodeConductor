"""
vLLM Integration for CodeConductor

High-performance inference engine with AWQ quantization support for RTX 5090.
Provides optimized code generation with ensemble consensus capabilities.
Platform-specific - excluded from coverage on Windows
"""

import asyncio
import logging
from typing import Any

from vllm import LLM, SamplingParams  # pragma: no cover

logger = logging.getLogger(__name__)


class VLLMEngine:
    """High-performance vLLM inference engine for CodeConductor."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        quantization: str = "awq",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        """
        Initialize vLLM engine for RTX 5090.

        Args:
            model_name: HuggingFace model identifier
            quantization: Quantization method (awq, gptq, etc.)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory usage (0.0-1.0)
            trust_remote_code: Trust custom model code
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype

        self.llm: LLM | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the vLLM engine asynchronously."""
        if self._initialized:
            return

        try:
            logger.info(f"Initializing vLLM engine with model: {self.model_name}")

            # Configure sampling parameters for code generation
            self.sampling_params = SamplingParams(
                temperature=0.1,  # Low temperature for deterministic code
                top_p=0.95,
                max_tokens=2048,
                stop=["```", "\n\n\n", "END"],
            )

            # Initialize LLM with AWQ quantization for RTX 5090
            self.llm = LLM(
                model=self.model_name,
                quantization=self.quantization,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
            )

            self._initialized = True
            logger.info("vLLM engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise

    async def generate_code(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """
        Generate code using vLLM engine.

        Args:
            prompt: Input prompt for code generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Tokens to stop generation

        Returns:
            Generated code string
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Update sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                stop=stop_tokens or ["```", "\n\n\n", "END"],
            )

            # Generate code
            outputs = self.llm.generate([prompt], sampling_params)

            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                logger.info(f"Generated {len(generated_text)} characters of code")
                return generated_text
            else:
                logger.warning("No output generated from vLLM")
                return ""

        except Exception as e:
            logger.error(f"Error generating code with vLLM: {e}")
            raise

    async def generate_with_consensus(
        self,
        prompt: str,
        num_models: int = 3,
        temperature_range: tuple = (0.1, 0.3, 0.5),
    ) -> dict[str, Any]:
        """
        Generate code with ensemble consensus using multiple temperature settings.

        Args:
            prompt: Input prompt
            num_models: Number of consensus models
            temperature_range: Temperature values for diversity

        Returns:
            Dictionary with generated code and consensus metrics
        """
        if not self._initialized:
            await self.initialize()

        results = []

        # Generate with different temperatures for diversity
        for temp in temperature_range:
            try:
                sampling_params = SamplingParams(
                    temperature=temp,
                    top_p=0.95,
                    max_tokens=2048,
                    stop=["```", "\n\n\n", "END"],
                )

                outputs = self.llm.generate([prompt], sampling_params)
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text
                    results.append(
                        {
                            "temperature": temp,
                            "code": generated_text,
                            "length": len(generated_text),
                        }
                    )

            except Exception as e:
                logger.error(f"Error in consensus generation with temp {temp}: {e}")

        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(results)

        return {
            "generations": results,
            "consensus_metrics": consensus_metrics,
            "best_generation": self._select_best_generation(results),
        }

    def _calculate_consensus_metrics(self, results: list[dict]) -> dict[str, Any]:
        """Calculate consensus metrics for generated code."""
        if not results:
            return {}

        lengths = [r["length"] for r in results]
        avg_length = sum(lengths) / len(lengths)

        # Simple similarity metric (can be enhanced with CodeBLEU)
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                sim = self._calculate_similarity(results[i]["code"], results[j]["code"])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        return {
            "num_generations": len(results),
            "average_length": avg_length,
            "average_similarity": avg_similarity,
            "length_variance": self._calculate_variance(lengths),
        }

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets."""
        # Simple token-based similarity (placeholder for CodeBLEU)
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / len(values)

    def _select_best_generation(self, results: list[dict]) -> dict | None:
        """Select the best generation based on quality metrics."""
        if not results:
            return None

        # Simple selection based on length and temperature
        # (can be enhanced with more sophisticated metrics)
        best_score = 0
        best_result = None

        for result in results:
            # Prefer medium temperature and reasonable length
            temp_score = 1.0 - abs(result["temperature"] - 0.2)
            length_score = min(result["length"] / 1000, 1.0)  # Cap at 1000 chars

            total_score = temp_score * 0.6 + length_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_result = result

        return best_result

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized or not self.llm:
            return {"error": "Model not initialized"}

        try:
            return {
                "model_name": self.model_name,
                "quantization": self.quantization,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "device": str(self.llm.llm_engine.device),
                "model_config": str(self.llm.llm_engine.model_config),
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}

    async def cleanup(self) -> None:
        """Clean up vLLM resources."""
        if self.llm:
            try:
                # vLLM handles cleanup automatically
                self.llm = None
                self._initialized = False
                logger.info("vLLM engine cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up vLLM: {e}")


# Factory function for creating vLLM engines
async def create_vllm_engine(
    model_name: str = "microsoft/DialoGPT-medium", **kwargs
) -> VLLMEngine:
    """Create and initialize a vLLM engine."""
    engine = VLLMEngine(model_name=model_name, **kwargs)
    await engine.initialize()
    return engine


# Example usage
async def main():  # pragma: no cover
    """Example usage of vLLM integration."""
    engine = await create_vllm_engine(
        model_name="microsoft/DialoGPT-medium", quantization="awq", max_model_len=4096
    )

    prompt = """Write a Python function to calculate the factorial of a number:

def factorial(n):
"""

    try:
        # Single generation
        code = await engine.generate_code(prompt)
        print("Generated code:", code)

        # Consensus generation
        consensus_result = await engine.generate_with_consensus(prompt)
        print("Consensus result:", consensus_result)

    finally:
        await engine.cleanup()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
