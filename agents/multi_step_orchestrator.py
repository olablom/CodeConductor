#!/usr/bin/env python3
"""
Multi-Step Reasoning Orchestrator
Combines contextual bandits with predictive workflow planning
"""

import logging
import requests
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.orchestrator import AgentOrchestrator, DiscussionRound

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in a multi-step workflow."""

    step_id: int
    agent_type: str
    task_description: str
    dependencies: List[int]  # Step IDs this depends on
    expected_duration: float
    success_criteria: List[str]
    fallback_agents: List[str]


@dataclass
class WorkflowPlan:
    """Complete multi-step workflow plan."""

    steps: List[WorkflowStep]
    total_estimated_duration: float
    critical_path: List[int]
    risk_assessment: Dict[str, float]
    confidence_score: float


class MultiStepOrchestrator(AgentOrchestrator):
    """
    Cutting-edge orchestrator that combines:
    1. Multi-step reasoning with chain-of-thought planning
    2. Contextual bandits for adaptive agent selection
    3. Predictive workflow optimization
    4. Real-time adaptation to failures
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        gpu_service_url: str = "http://localhost:8009",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the multi-step reasoning orchestrator."""
        super().__init__(agents, config)

        self.gpu_service_url = gpu_service_url
        self.workflow_history: List[WorkflowPlan] = []

        # Multi-step specific configuration
        multi_step_config = {
            "enable_chain_of_thought": True,
            "max_planning_depth": 5,
            "enable_predictive_planning": True,
            "enable_adaptive_execution": True,
            "enable_workflow_optimization": True,
            "context_window_size": 10,
        }

        if config:
            multi_step_config.update(config)

        self.multi_step_config = multi_step_config

        # Available agent types for contextual selection
        self.agent_types = [
            "architect_agent",
            "security_agent",
            "review_agent",
            "codegen_agent",
            "policy_agent",
            "reward_agent",
            "qlearning_agent",
            "test_agent",
        ]

        logger.info(
            f"Multi-Step Orchestrator initialized with {len(self.agents)} agents"
        )
        logger.info(f"GPU Service URL: {self.gpu_service_url}")

    def plan_workflow(self, task_context: Dict[str, Any]) -> WorkflowPlan:
        """
        Create a multi-step workflow plan using chain-of-thought reasoning.

        Args:
            task_context: Task context and requirements

        Returns:
            Complete workflow plan with steps and dependencies
        """
        logger.info("Planning multi-step workflow with chain-of-thought reasoning")

        # Step 1: Context analysis
        context_features = self._extract_context_features(task_context)
        complexity_score = self._assess_task_complexity(context_features)

        # Step 2: Chain-of-thought decomposition
        decomposition = self._decompose_task_chain_of_thought(
            task_context, complexity_score
        )

        # Step 3: Create workflow steps
        steps = self._create_workflow_steps(decomposition, context_features)

        # Step 4: Optimize workflow
        optimized_steps = self._optimize_workflow(steps, context_features)

        # Step 5: Calculate metrics
        total_duration = sum(step.expected_duration for step in optimized_steps)
        critical_path = self._calculate_critical_path(optimized_steps)
        risk_assessment = self._assess_workflow_risks(optimized_steps, context_features)
        confidence_score = self._calculate_plan_confidence(
            optimized_steps, context_features
        )

        workflow_plan = WorkflowPlan(
            steps=optimized_steps,
            total_estimated_duration=total_duration,
            critical_path=critical_path,
            risk_assessment=risk_assessment,
            confidence_score=confidence_score,
        )

        logger.info(
            f"Workflow planned: {len(steps)} steps, {total_duration:.1f}s estimated, confidence: {confidence_score:.2f}"
        )
        return workflow_plan

    def execute_workflow(
        self, workflow_plan: WorkflowPlan, task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the multi-step workflow with adaptive execution.

        Args:
            workflow_plan: Planned workflow
            task_context: Original task context

        Returns:
            Execution results with adaptation metadata
        """
        logger.info(f"Executing workflow with {len(workflow_plan.steps)} steps")

        execution_results = {
            "workflow_id": f"wf_{len(self.workflow_history)}",
            "steps_executed": [],
            "adaptations_made": [],
            "total_duration": 0.0,
            "success_rate": 0.0,
            "final_result": None,
        }

        completed_steps = {}
        failed_steps = {}

        # Execute steps in dependency order
        for step in workflow_plan.steps:
            logger.info(f"Executing step {step.step_id}: {step.agent_type}")

            # Check dependencies
            if not self._check_dependencies(step, completed_steps):
                logger.warning(f"Step {step.step_id} dependencies not met, skipping")
                failed_steps[step.step_id] = step
                continue

            # Execute step with contextual bandit selection
            step_result = self._execute_step_with_adaptation(
                step, task_context, completed_steps
            )

            if step_result["success"]:
                completed_steps[step.step_id] = step_result
                execution_results["steps_executed"].append(step_result)
            else:
                failed_steps[step.step_id] = step
                # Try fallback agents
                fallback_result = self._try_fallback_agents(
                    step, task_context, completed_steps
                )
                if fallback_result["success"]:
                    completed_steps[step.step_id] = fallback_result
                    execution_results["steps_executed"].append(fallback_result)
                    execution_results["adaptations_made"].append(
                        {
                            "step_id": step.step_id,
                            "original_agent": step.agent_type,
                            "fallback_agent": fallback_result["agent_used"],
                        }
                    )
                else:
                    logger.error(f"Step {step.step_id} failed with all agents")

            execution_results["total_duration"] += step_result.get("duration", 0.0)

        # Calculate success rate
        total_steps = len(workflow_plan.steps)
        successful_steps = len(execution_results["steps_executed"])
        execution_results["success_rate"] = (
            successful_steps / total_steps if total_steps > 0 else 0.0
        )

        # Generate final result
        execution_results["final_result"] = self._synthesize_final_result(
            completed_steps, task_context
        )

        logger.info(
            f"Workflow completed: {successful_steps}/{total_steps} steps successful, {execution_results['total_duration']:.1f}s total"
        )

        return execution_results

    def _extract_context_features(self, task_context: Dict[str, Any]) -> List[float]:
        """Extract comprehensive context features for multi-step planning."""
        features = [
            # Task complexity features
            self._calculate_complexity_score(task_context.get("description", "")),
            self._calculate_urgency_score(task_context),
            self._calculate_team_size_score(task_context),
            self._calculate_deadline_score(task_context),
            self._calculate_domain_expertise_score(task_context),
            self._calculate_code_quality_score(task_context),
            self._calculate_testing_required_score(task_context),
            self._calculate_documentation_needed_score(task_context),
            self._calculate_security_level_score(task_context),
            self._calculate_performance_priority_score(task_context),
            # Multi-step specific features
            self._calculate_workflow_complexity_score(task_context),
            self._calculate_interdependency_score(task_context),
            self._calculate_risk_tolerance_score(task_context),
            self._calculate_adaptation_needs_score(task_context),
        ]

        return features

    def _assess_task_complexity(self, context_features: List[float]) -> float:
        """Assess overall task complexity for workflow planning."""
        # Weight different complexity factors
        weights = [0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05]
        complexity = sum(f * w for f, w in zip(context_features[:11], weights))
        return min(1.0, max(0.0, complexity))

    def _decompose_task_chain_of_thought(
        self, task_context: Dict[str, Any], complexity_score: float
    ) -> List[Dict[str, Any]]:
        """Decompose task using chain-of-thought reasoning with adaptive planning."""
        description = task_context.get("description", "").lower()
        title = task_context.get("title", "").lower()

        # Enhanced complexity analysis
        task_analysis = self._analyze_task_semantics(
            description, title, complexity_score
        )

        # Chain-of-thought decomposition based on task type
        decomposition = []

        if task_analysis["task_type"] == "microservices":
            # High complexity: microservices architecture
            decomposition = [
                {
                    "phase": "architecture",
                    "description": "Design microservices architecture",
                    "agents": ["architect_agent"],
                },
                {
                    "phase": "security",
                    "description": "Security and API design",
                    "agents": ["policy_agent"],
                },
                {
                    "phase": "implementation",
                    "description": "Generate microservices code",
                    "agents": ["codegen_agent"],
                },
                {
                    "phase": "review",
                    "description": "Code review and validation",
                    "agents": ["review_agent"],
                },
                {
                    "phase": "testing",
                    "description": "Generate tests",
                    "agents": ["codegen_agent"],
                },
                {
                    "phase": "validation",
                    "description": "Final validation",
                    "agents": ["policy_agent"],
                },
            ]
        elif task_analysis["task_type"] == "api":
            # Medium-high complexity: API development
            decomposition = [
                {
                    "phase": "design",
                    "description": "Design API structure",
                    "agents": ["architect_agent"],
                },
                {
                    "phase": "implementation",
                    "description": "Generate API code",
                    "agents": ["codegen_agent"],
                },
                {
                    "phase": "review",
                    "description": "Review and validate",
                    "agents": ["review_agent", "policy_agent"],
                },
            ]
        elif task_analysis["task_type"] == "simple_app":
            # Low-medium complexity: simple applications
            decomposition = [
                {
                    "phase": "planning",
                    "description": "Plan implementation",
                    "agents": ["architect_agent"],
                },
                {
                    "phase": "implementation",
                    "description": "Generate code",
                    "agents": ["codegen_agent"],
                },
                {
                    "phase": "validation",
                    "description": "Basic validation",
                    "agents": ["policy_agent"],
                },
            ]
        elif task_analysis["task_type"] == "utility":
            # Low complexity: utilities and simple functions
            decomposition = [
                {
                    "phase": "implementation",
                    "description": "Generate simple solution",
                    "agents": ["codegen_agent"],
                },
                {
                    "phase": "validation",
                    "description": "Basic validation",
                    "agents": ["policy_agent"],
                },
            ]
        else:
            # Default fallback based on complexity score
            if complexity_score > 0.7:
                decomposition = [
                    {
                        "phase": "analysis",
                        "description": "Analyze requirements",
                        "agents": ["architect_agent"],
                    },
                    {
                        "phase": "implementation",
                        "description": "Generate code",
                        "agents": ["codegen_agent"],
                    },
                    {
                        "phase": "review",
                        "description": "Review and validate",
                        "agents": ["review_agent", "policy_agent"],
                    },
                ]
            elif complexity_score > 0.4:
                decomposition = [
                    {
                        "phase": "implementation",
                        "description": "Generate code",
                        "agents": ["codegen_agent"],
                    },
                    {
                        "phase": "validation",
                        "description": "Validate result",
                        "agents": ["policy_agent"],
                    },
                ]
            else:
                decomposition = [
                    {
                        "phase": "implementation",
                        "description": "Generate simple solution",
                        "agents": ["codegen_agent"],
                    }
                ]

        logger.info(
            f"Task analysis: {task_analysis['task_type']}, planned {len(decomposition)} steps"
        )
        return decomposition

    def _analyze_task_semantics(
        self, description: str, title: str, complexity_score: float
    ) -> Dict[str, Any]:
        """Analyze task semantics to determine optimal workflow."""
        text = f"{title} {description}".lower()

        # Task type detection
        if any(
            keyword in text
            for keyword in [
                "microservices",
                "microservice",
                "distributed",
                "service-oriented",
            ]
        ):
            task_type = "microservices"
            expected_steps = 5
            agents_needed = [
                "architect_agent",
                "policy_agent",
                "codegen_agent",
                "review_agent",
            ]
        elif any(
            keyword in text
            for keyword in ["api", "rest", "fastapi", "flask", "endpoint", "http"]
        ):
            task_type = "api"
            expected_steps = 3
            agents_needed = [
                "architect_agent",
                "codegen_agent",
                "review_agent",
                "policy_agent",
            ]
        elif any(
            keyword in text
            for keyword in ["calculator", "simple", "basic", "utility", "function"]
        ):
            task_type = "utility"
            expected_steps = 1
            agents_needed = ["codegen_agent", "policy_agent"]
        elif any(
            keyword in text for keyword in ["application", "app", "system", "platform"]
        ):
            task_type = "simple_app"
            expected_steps = 3
            agents_needed = ["architect_agent", "codegen_agent", "policy_agent"]
        else:
            task_type = "unknown"
            expected_steps = max(2, int(complexity_score * 5))
            agents_needed = ["codegen_agent", "policy_agent"]

        return {
            "task_type": task_type,
            "expected_steps": expected_steps,
            "agents_needed": agents_needed,
            "complexity_score": complexity_score,
            "semantic_analysis": {
                "has_architecture": "architect" in text or "design" in text,
                "has_security": "security" in text or "auth" in text,
                "has_api": "api" in text or "endpoint" in text,
                "is_simple": "simple" in text or "basic" in text,
            },
        }

    def _create_workflow_steps(
        self, decomposition: List[Dict[str, Any]], context_features: List[float]
    ) -> List[WorkflowStep]:
        """Create workflow steps from decomposition."""
        steps = []

        for i, phase in enumerate(decomposition):
            # Use contextual bandit to select optimal agent
            selected_agent = self._select_agent_contextually(
                phase["agents"], context_features
            )

            step = WorkflowStep(
                step_id=i + 1,
                agent_type=selected_agent,
                task_description=phase["description"],
                dependencies=[j + 1 for j in range(i)],  # All previous steps
                expected_duration=self._estimate_step_duration(
                    selected_agent, context_features
                ),
                success_criteria=self._define_success_criteria(selected_agent),
                fallback_agents=[
                    agent for agent in phase["agents"] if agent != selected_agent
                ],
            )
            steps.append(step)

        return steps

    def _select_agent_contextually(
        self, available_agents: List[str], context_features: List[float]
    ) -> str:
        """Use contextual bandit to select optimal agent for current context."""
        try:
            # Prepare request for contextual bandit
            request_data = {
                "arms": available_agents,
                "features": context_features,
                "epsilon": 0.1,  # Exploration rate
            }

            # Call GPU contextual bandit service
            response = requests.post(
                f"{self.gpu_service_url}/gpu/bandits/choose",
                json=request_data,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                return result["selected_arm"]
            else:
                # Fallback to first available agent
                return available_agents[0] if available_agents else "codegen_agent"

        except Exception as e:
            logger.warning(f"Contextual bandit selection failed: {e}, using fallback")
            return available_agents[0] if available_agents else "codegen_agent"

    def _optimize_workflow(
        self, steps: List[WorkflowStep], context_features: List[float]
    ) -> List[WorkflowStep]:
        """Optimize workflow for efficiency and reliability."""
        # Simple optimization: parallelize independent steps
        optimized_steps = []

        for i, step in enumerate(steps):
            # Check if step can be parallelized
            if i > 0:
                # Only depend on critical path steps
                critical_dependencies = [j + 1 for j in range(i) if j < len(steps) // 2]
                step.dependencies = critical_dependencies

            optimized_steps.append(step)

        return optimized_steps

    def _execute_step_with_adaptation(
        self,
        step: WorkflowStep,
        task_context: Dict[str, Any],
        completed_steps: Dict[int, Any],
    ) -> Dict[str, Any]:
        """Execute a single step with adaptive behavior."""
        # Find the agent
        agent = self._find_agent_by_type(step.agent_type)

        if not agent:
            return {"success": False, "error": f"Agent {step.agent_type} not found"}

        try:
            # Execute the step
            analysis = agent.analyze(task_context)
            proposal = agent.propose(analysis, task_context)

            return {
                "success": True,
                "step_id": step.step_id,
                "agent_used": step.agent_type,
                "result": proposal,
                "duration": step.expected_duration,
            }

        except Exception as e:
            logger.error(f"Step {step.step_id} execution failed: {e}")
            return {"success": False, "error": str(e), "step_id": step.step_id}

    def _try_fallback_agents(
        self,
        step: WorkflowStep,
        task_context: Dict[str, Any],
        completed_steps: Dict[int, Any],
    ) -> Dict[str, Any]:
        """Try fallback agents if primary agent fails."""
        for fallback_agent_type in step.fallback_agents:
            logger.info(f"Trying fallback agent: {fallback_agent_type}")

            agent = self._find_agent_by_type(fallback_agent_type)
            if agent:
                try:
                    analysis = agent.analyze(task_context)
                    proposal = agent.propose(analysis, task_context)

                    return {
                        "success": True,
                        "step_id": step.step_id,
                        "agent_used": fallback_agent_type,
                        "result": proposal,
                        "duration": step.expected_duration,
                        "was_fallback": True,
                    }

                except Exception as e:
                    logger.warning(
                        f"Fallback agent {fallback_agent_type} also failed: {e}"
                    )
                    continue

        return {"success": False, "error": "All agents failed", "step_id": step.step_id}

    def _find_agent_by_type(self, agent_type: str) -> Optional[BaseAgent]:
        """Find agent by type name."""
        logger.info(f"Looking for agent type: {agent_type}")
        for agent in self.agents:
            agent_name_lower = agent.name.lower().replace("_", "")
            agent_type_clean = agent_type.replace("_", "")
            logger.info(f"Checking agent: {agent.name} (clean: {agent_name_lower})")
            if agent_type_clean in agent_name_lower:
                logger.info(f"Found agent: {agent.name} for type: {agent_type}")
                return agent
        logger.warning(f"No agent found for type: {agent_type}")
        return None

    def _check_dependencies(
        self, step: WorkflowStep, completed_steps: Dict[int, Any]
    ) -> bool:
        """Check if step dependencies are met."""
        for dep_id in step.dependencies:
            if dep_id not in completed_steps:
                return False
        return True

    def _calculate_critical_path(self, steps: List[WorkflowStep]) -> List[int]:
        """Calculate critical path through workflow."""
        # Simple critical path: steps with longest duration
        durations = [(i + 1, step.expected_duration) for i, step in enumerate(steps)]
        durations.sort(key=lambda x: x[1], reverse=True)

        # Return top 50% of steps as critical path
        critical_count = max(1, len(steps) // 2)
        return [step_id for step_id, _ in durations[:critical_count]]

    def _assess_workflow_risks(
        self, steps: List[WorkflowStep], context_features: List[float]
    ) -> Dict[str, float]:
        """Assess risks in the workflow."""
        return {
            "dependency_risk": 0.2,
            "agent_failure_risk": 0.15,
            "complexity_risk": context_features[0] * 0.3,
            "time_risk": 0.1,
        }

    def _calculate_plan_confidence(
        self, steps: List[WorkflowStep], context_features: List[float]
    ) -> float:
        """Calculate confidence in the workflow plan."""
        # Base confidence on number of steps and complexity
        base_confidence = 0.8
        complexity_penalty = context_features[0] * 0.2
        step_penalty = len(steps) * 0.05

        confidence = base_confidence - complexity_penalty - step_penalty
        return max(0.1, min(1.0, confidence))

    def _estimate_step_duration(
        self, agent_type: str, context_features: List[float]
    ) -> float:
        """Estimate duration for a step."""
        base_durations = {
            "architect_agent": 30.0,
            "security_agent": 20.0,
            "review_agent": 15.0,
            "codegen_agent": 25.0,
            "policy_agent": 10.0,
            "reward_agent": 5.0,
            "test_agent": 20.0,
        }

        base_duration = base_durations.get(agent_type, 15.0)
        complexity_multiplier = 1.0 + context_features[0] * 0.5

        return base_duration * complexity_multiplier

    def _define_success_criteria(self, agent_type: str) -> List[str]:
        """Define success criteria for a step."""
        criteria_map = {
            "architect_agent": ["Architecture designed", "Components identified"],
            "security_agent": ["Security reviewed", "Vulnerabilities identified"],
            "review_agent": ["Code reviewed", "Issues found"],
            "codegen_agent": ["Code generated", "Tests included"],
            "policy_agent": ["Policy compliant", "Safety checked"],
            "reward_agent": ["Reward calculated", "Score > 0.7"],
            "test_agent": ["Tests generated", "Coverage > 80%"],
        }

        return criteria_map.get(agent_type, ["Task completed"])

    def _synthesize_final_result(
        self, completed_steps: Dict[int, Any], task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final result from completed steps."""
        # Combine results from all completed steps
        combined_result = {
            "workflow_completed": True,
            "steps_completed": len(completed_steps),
            "final_code": None,
            "architecture": None,
            "review_results": [],
            "policy_results": [],
            "test_results": [],
            "reward_score": 0.0,
        }

        for step_result in completed_steps.values():
            if step_result.get("success"):
                result = step_result.get("result", {})
                agent_type = step_result.get("agent_used", "")

                if "codegen_agent" in agent_type and "code" in result:
                    combined_result["final_code"] = result["code"]
                elif "architect_agent" in agent_type:
                    combined_result["architecture"] = result
                elif "review_agent" in agent_type:
                    combined_result["review_results"].append(result)
                elif "policy_agent" in agent_type:
                    combined_result["policy_results"].append(result)
                elif "test_agent" in agent_type:
                    combined_result["test_results"].append(result)
                elif "reward_agent" in agent_type:
                    combined_result["reward_score"] = result.get("total_reward", 0.0)

        return combined_result

    # Feature extraction helper methods (reused from GPUOrchestrator)
    def _calculate_complexity_score(self, description: str) -> float:
        """Calculate complexity score from description."""
        complexity_keywords = [
            "complex",
            "advanced",
            "sophisticated",
            "enterprise",
            "scalable",
        ]
        score = sum(
            1 for keyword in complexity_keywords if keyword in description.lower()
        )
        return min(score / len(complexity_keywords), 1.0)

    def _calculate_urgency_score(self, context: Dict[str, Any]) -> float:
        """Calculate urgency score."""
        urgency = context.get("urgency", "medium")
        urgency_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        return urgency_map.get(urgency.lower(), 0.5)

    def _calculate_team_size_score(self, context: Dict[str, Any]) -> float:
        """Calculate team size score."""
        team_size = context.get("team_size", "small")
        size_map = {"small": 0.2, "medium": 0.5, "large": 0.8, "enterprise": 1.0}
        return size_map.get(team_size.lower(), 0.5)

    def _calculate_deadline_score(self, context: Dict[str, Any]) -> float:
        """Calculate deadline pressure score."""
        deadline = context.get("deadline", "flexible")
        deadline_map = {"flexible": 0.2, "moderate": 0.5, "tight": 0.8, "urgent": 1.0}
        return deadline_map.get(deadline.lower(), 0.5)

    def _calculate_domain_expertise_score(self, context: Dict[str, Any]) -> float:
        """Calculate domain expertise requirement score."""
        expertise = context.get("domain_expertise", "general")
        expertise_map = {
            "general": 0.2,
            "moderate": 0.5,
            "specialized": 0.8,
            "expert": 1.0,
        }
        return expertise_map.get(expertise.lower(), 0.5)

    def _calculate_code_quality_score(self, context: Dict[str, Any]) -> float:
        """Calculate code quality requirement score."""
        quality = context.get("code_quality", "standard")
        quality_map = {"basic": 0.2, "standard": 0.5, "high": 0.8, "enterprise": 1.0}
        return quality_map.get(quality.lower(), 0.5)

    def _calculate_testing_required_score(self, context: Dict[str, Any]) -> float:
        """Calculate testing requirement score."""
        testing = context.get("testing_required", "basic")
        testing_map = {
            "none": 0.0,
            "basic": 0.3,
            "comprehensive": 0.7,
            "enterprise": 1.0,
        }
        return testing_map.get(testing.lower(), 0.3)

    def _calculate_documentation_needed_score(self, context: Dict[str, Any]) -> float:
        """Calculate documentation requirement score."""
        docs = context.get("documentation_needed", "basic")
        docs_map = {"none": 0.0, "basic": 0.3, "comprehensive": 0.7, "enterprise": 1.0}
        return docs_map.get(docs.lower(), 0.3)

    def _calculate_security_level_score(self, context: Dict[str, Any]) -> float:
        """Calculate security requirement score."""
        security = context.get("security_level", "standard")
        security_map = {"basic": 0.2, "standard": 0.5, "high": 0.8, "enterprise": 1.0}
        return security_map.get(security.lower(), 0.5)

    def _calculate_performance_priority_score(self, context: Dict[str, Any]) -> float:
        """Calculate performance priority score."""
        performance = context.get("performance_priority", "balanced")
        perf_map = {"low": 0.2, "balanced": 0.5, "high": 0.8, "critical": 1.0}
        return perf_map.get(performance.lower(), 0.5)

    def _calculate_workflow_complexity_score(self, context: Dict[str, Any]) -> float:
        """Calculate workflow complexity score."""
        # Based on number of components and interdependencies
        return 0.6  # Default medium complexity

    def _calculate_interdependency_score(self, context: Dict[str, Any]) -> float:
        """Calculate interdependency score."""
        # How much steps depend on each other
        return 0.4  # Default medium interdependency

    def _calculate_risk_tolerance_score(self, context: Dict[str, Any]) -> float:
        """Calculate risk tolerance score."""
        risk_tolerance = context.get("risk_tolerance", "medium")
        risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        return risk_map.get(risk_tolerance.lower(), 0.5)

    def _calculate_adaptation_needs_score(self, context: Dict[str, Any]) -> float:
        """Calculate adaptation needs score."""
        # How much the workflow might need to adapt
        return 0.3  # Default low adaptation needs
