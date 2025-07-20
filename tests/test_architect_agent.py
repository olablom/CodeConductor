"""
Unit tests for ArchitectAgent
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.architect_agent import ArchitectAgent, ArchitectureComponent, SystemPattern


class TestArchitectAgent(unittest.TestCase):
    """Test cases for ArchitectAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = ArchitectAgent("TestArchitectAgent")
        self.sample_context = {
            "requirements": {
                "system_type": "web_application",
                "scale": "enterprise",
                "performance": "high",
                "security": "critical",
                "availability": "99.9%",
                "data_volume": "large",
                "user_count": "10000+",
            },
            "constraints": {
                "budget": "limited",
                "time_to_market": "fast",
                "team_size": "small",
                "existing_infrastructure": "cloud",
            },
            "existing_architecture": {
                "components": ["web_server", "database"],
                "patterns": ["monolithic"],
                "technologies": ["Python", "PostgreSQL"],
            },
        }

    def test_init(self):
        """Test ArchitectAgent initialization"""
        agent = ArchitectAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.architect_config)
        self.assertIsNotNone(agent.architecture_patterns)
        self.assertIsNotNone(agent.technology_stack)

    def test_init_with_config(self):
        """Test ArchitectAgent initialization with config"""
        config = {
            "architect": {"default_pattern": "microservices", "preferred_cloud": "AWS"}
        }
        agent = ArchitectAgent("TestAgent", config)
        self.assertEqual(
            agent.architect_config,
            {"default_pattern": "microservices", "preferred_cloud": "AWS"},
        )

    def test_analyze_basic(self):
        """Test basic analysis functionality"""
        result = self.agent.analyze(self.sample_context)

        self.assertIn("requirements_analysis", result)
        self.assertIn("current_architecture_assessment", result)
        self.assertIn("scalability_analysis", result)
        self.assertIn("security_analysis", result)
        self.assertIn("performance_analysis", result)
        self.assertIn("technology_compatibility", result)
        self.assertIn("risk_assessment", result)
        self.assertIn("migration_complexity", result)

    def test_analyze_with_error(self):
        """Test analysis with error handling"""
        with patch.object(
            self.agent, "_analyze_requirements", side_effect=Exception("Test error")
        ):
            result = self.agent.analyze(self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["migration_complexity"], "unknown")

    def test_propose_basic(self):
        """Test basic proposal functionality"""
        analysis = {
            "requirements_analysis": {"system_type": "web_application"},
            "current_architecture_assessment": {"pattern": "monolithic"},
            "scalability_analysis": {"current_scale": "limited"},
            "security_analysis": {"current_security": "basic"},
            "performance_analysis": {"current_performance": "adequate"},
            "migration_complexity": "medium",
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("architecture_design", result)
        self.assertIn("component_breakdown", result)
        self.assertIn("technology_stack", result)
        self.assertIn("deployment_strategy", result)
        self.assertIn("scalability_plan", result)
        self.assertIn("security_architecture", result)
        self.assertIn("migration_plan", result)
        self.assertIn("cost_estimation", result)

    def test_propose_with_error(self):
        """Test proposal with error handling"""
        with patch.object(
            self.agent, "_design_architecture", side_effect=Exception("Test error")
        ):
            analysis = {"migration_complexity": "medium"}
            result = self.agent.propose(analysis, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["architecture_design"], {})

    def test_review_basic(self):
        """Test basic review functionality"""
        proposal = {
            "architecture_design": {"pattern": "microservices"},
            "component_breakdown": ["api_gateway", "user_service"],
            "technology_stack": ["Python", "Docker"],
            "deployment_strategy": "containerized",
            "scalability_plan": ["horizontal_scaling"],
            "security_architecture": ["authentication", "authorization"],
            "migration_plan": ["phase1", "phase2"],
            "cost_estimation": {"monthly": 1000},
        }

        result = self.agent.review(proposal, self.sample_context)

        self.assertIn("architecture_quality", result)
        self.assertIn("scalability_assessment", result)
        self.assertIn("security_review", result)
        self.assertIn("performance_evaluation", result)
        self.assertIn("maintainability_analysis", result)
        self.assertIn("cost_benefit_analysis", result)
        self.assertIn("risk_evaluation", result)
        self.assertIn("approval_recommendation", result)
        self.assertIn("final_score", result)

    def test_review_with_error(self):
        """Test review with error handling"""
        with patch.object(
            self.agent,
            "_assess_architecture_quality",
            side_effect=Exception("Test error"),
        ):
            proposal = {"architecture_design": {}}
            result = self.agent.review(proposal, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["approval_recommendation"], "reject")

    def test_analyze_requirements(self):
        """Test requirements analysis"""
        requirements = self.agent._analyze_requirements(self.sample_context)

        self.assertIn("system_type", requirements)
        self.assertIn("scale", requirements)
        self.assertIn("performance", requirements)
        self.assertIn("security", requirements)
        self.assertIn("availability", requirements)
        self.assertIn("data_volume", requirements)
        self.assertIn("user_count", requirements)

    def test_assess_current_architecture(self):
        """Test current architecture assessment"""
        existing_architecture = {
            "components": ["web_server", "database"],
            "patterns": ["monolithic"],
            "technologies": ["Python", "PostgreSQL"],
        }

        assessment = self.agent._assess_current_architecture(existing_architecture)

        self.assertIn("pattern_analysis", assessment)
        self.assertIn("component_analysis", assessment)
        self.assertIn("technology_analysis", assessment)
        self.assertIn("strengths", assessment)
        self.assertIn("weaknesses", assessment)
        self.assertIn("migration_readiness", assessment)

    def test_analyze_scalability(self):
        """Test scalability analysis"""
        requirements = {
            "scale": "enterprise",
            "user_count": "10000+",
            "data_volume": "large",
        }
        current_architecture = {"pattern": "monolithic"}

        scalability = self.agent._analyze_scalability(
            requirements, current_architecture
        )

        self.assertIn("current_scale", scalability)
        self.assertIn("required_scale", scalability)
        self.assertIn("scaling_gaps", scalability)
        self.assertIn("scaling_opportunities", scalability)
        self.assertIn("scaling_strategy", scalability)

    def test_analyze_security_requirements(self):
        """Test security requirements analysis"""
        requirements = {
            "security": "critical",
            "data_volume": "large",
            "user_count": "10000+",
        }
        current_architecture = {"security": "basic"}

        security = self.agent._analyze_security_requirements(
            requirements, current_architecture
        )

        self.assertIn("current_security", security)
        self.assertIn("required_security", security)
        self.assertIn("security_gaps", security)
        self.assertIn("security_priorities", security)
        self.assertIn("compliance_requirements", security)

    def test_analyze_performance_requirements(self):
        """Test performance requirements analysis"""
        requirements = {
            "performance": "high",
            "availability": "99.9%",
            "user_count": "10000+",
        }
        current_architecture = {"performance": "adequate"}

        performance = self.agent._analyze_performance_requirements(
            requirements, current_architecture
        )

        self.assertIn("current_performance", performance)
        self.assertIn("required_performance", performance)
        self.assertIn("performance_bottlenecks", performance)
        self.assertIn("performance_optimizations", performance)
        self.assertIn("performance_metrics", performance)

    def test_assess_technology_compatibility(self):
        """Test technology compatibility assessment"""
        requirements = {"system_type": "web_application", "scale": "enterprise"}
        current_technologies = ["Python", "PostgreSQL"]

        compatibility = self.agent._assess_technology_compatibility(
            requirements, current_technologies
        )

        self.assertIn("compatible_technologies", compatibility)
        self.assertIn("incompatible_technologies", compatibility)
        self.assertIn("technology_recommendations", compatibility)
        self.assertIn("migration_effort", compatibility)

    def test_assess_risks(self):
        """Test risk assessment"""
        requirements = {"security": "critical", "availability": "99.9%"}
        constraints = {"budget": "limited", "time_to_market": "fast"}

        risks = self.agent._assess_risks(requirements, constraints)

        self.assertIn("technical_risks", risks)
        self.assertIn("business_risks", risks)
        self.assertIn("security_risks", risks)
        self.assertIn("operational_risks", risks)
        self.assertIn("risk_mitigation", risks)

    def test_calculate_migration_complexity(self):
        """Test migration complexity calculation"""
        current_architecture = {"pattern": "monolithic"}
        requirements = {"scale": "enterprise"}
        constraints = {"budget": "limited"}

        complexity = self.agent._calculate_migration_complexity(
            current_architecture, requirements, constraints
        )

        self.assertIn(complexity, ["low", "medium", "high", "unknown"])

    def test_design_architecture(self):
        """Test architecture design"""
        requirements = {
            "system_type": "web_application",
            "scale": "enterprise",
            "security": "critical",
        }
        analysis = {
            "scalability_analysis": {"scaling_strategy": "horizontal"},
            "security_analysis": {"security_priorities": ["authentication"]},
            "performance_analysis": {"performance_optimizations": ["caching"]},
        }

        design = self.agent._design_architecture(requirements, analysis)

        self.assertIn("pattern", design)
        self.assertIn("components", design)
        self.assertIn("layers", design)
        self.assertIn("interfaces", design)
        self.assertIn("data_flow", design)
        self.assertIn("deployment_model", design)

    def test_break_down_components(self):
        """Test component breakdown"""
        architecture_design = {
            "pattern": "microservices",
            "components": ["api_gateway", "user_service", "data_service"],
        }
        requirements = {"system_type": "web_application"}

        breakdown = self.agent._break_down_components(architecture_design, requirements)

        self.assertIsInstance(breakdown, list)
        self.assertGreater(len(breakdown), 0)
        for component in breakdown:
            self.assertIsInstance(component, ArchitectureComponent)
            self.assertIsInstance(component.name, str)
            self.assertIsInstance(component.responsibilities, list)
            self.assertIsInstance(component.technologies, list)

    def test_select_technology_stack(self):
        """Test technology stack selection"""
        requirements = {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
        }
        architecture_design = {"pattern": "microservices"}

        stack = self.agent._select_technology_stack(requirements, architecture_design)

        self.assertIn("languages", stack)
        self.assertIn("frameworks", stack)
        self.assertIn("databases", stack)
        self.assertIn("message_queues", stack)
        self.assertIn("monitoring", stack)
        self.assertIn("deployment", stack)

    def test_plan_deployment_strategy(self):
        """Test deployment strategy planning"""
        requirements = {"availability": "99.9%", "scale": "enterprise"}
        constraints = {"budget": "limited", "existing_infrastructure": "cloud"}
        architecture_design = {"pattern": "microservices"}

        strategy = self.agent._plan_deployment_strategy(
            requirements, constraints, architecture_design
        )

        self.assertIn("deployment_model", strategy)
        self.assertIn("infrastructure", strategy)
        self.assertIn("orchestration", strategy)
        self.assertIn("monitoring", strategy)
        self.assertIn("scaling_policy", strategy)
        self.assertIn("disaster_recovery", strategy)

    def test_design_scalability_plan(self):
        """Test scalability plan design"""
        requirements = {"scale": "enterprise", "user_count": "10000+"}
        architecture_design = {"pattern": "microservices"}

        plan = self.agent._design_scalability_plan(requirements, architecture_design)

        self.assertIn("scaling_strategy", plan)
        self.assertIn("scaling_triggers", plan)
        self.assertIn("resource_allocation", plan)
        self.assertIn("load_balancing", plan)
        self.assertIn("caching_strategy", plan)
        self.assertIn("database_scaling", plan)

    def test_design_security_architecture(self):
        """Test security architecture design"""
        requirements = {"security": "critical", "data_volume": "large"}
        architecture_design = {"pattern": "microservices"}

        security = self.agent._design_security_architecture(
            requirements, architecture_design
        )

        self.assertIn("authentication", security)
        self.assertIn("authorization", security)
        self.assertIn("data_protection", security)
        self.assertIn("network_security", security)
        self.assertIn("monitoring", security)
        self.assertIn("compliance", security)

    def test_create_migration_plan(self):
        """Test migration plan creation"""
        current_architecture = {"pattern": "monolithic"}
        new_architecture = {"pattern": "microservices"}
        constraints = {"time_to_market": "fast"}

        plan = self.agent._create_migration_plan(
            current_architecture, new_architecture, constraints
        )

        self.assertIn("phases", plan)
        self.assertIn("timeline", plan)
        self.assertIn("dependencies", plan)
        self.assertIn("rollback_strategy", plan)
        self.assertIn("testing_strategy", plan)
        self.assertIn("risk_mitigation", plan)

    def test_estimate_costs(self):
        """Test cost estimation"""
        requirements = {"scale": "enterprise", "availability": "99.9%"}
        architecture_design = {"pattern": "microservices"}
        constraints = {"budget": "limited"}

        costs = self.agent._estimate_costs(
            requirements, architecture_design, constraints
        )

        self.assertIn("infrastructure_costs", costs)
        self.assertIn("development_costs", costs)
        self.assertIn("operational_costs", costs)
        self.assertIn("maintenance_costs", costs)
        self.assertIn("total_monthly", costs)
        self.assertIn("total_yearly", costs)

    def test_assess_architecture_quality(self):
        """Test architecture quality assessment"""
        architecture_design = {
            "pattern": "microservices",
            "components": ["api_gateway", "user_service"],
            "layers": ["presentation", "business", "data"],
        }
        requirements = {"scale": "enterprise"}

        quality = self.agent._assess_architecture_quality(
            architecture_design, requirements
        )

        self.assertIn("modularity_score", quality)
        self.assertIn("cohesion_score", quality)
        self.assertIn("coupling_score", quality)
        self.assertIn("complexity_score", quality)
        self.assertIn("maintainability_score", quality)
        self.assertIn("overall_quality", quality)

    def test_assess_scalability(self):
        """Test scalability assessment"""
        architecture_design = {"pattern": "microservices"}
        scalability_plan = {
            "scaling_strategy": "horizontal",
            "load_balancing": "round_robin",
        }
        requirements = {"scale": "enterprise"}

        assessment = self.agent._assess_scalability(
            architecture_design, scalability_plan, requirements
        )

        self.assertIn("scalability_score", assessment)
        self.assertIn("bottlenecks", assessment)
        self.assertIn("scaling_capabilities", assessment)
        self.assertIn("performance_under_load", assessment)
        self.assertIn("recommendations", assessment)

    def test_review_security(self):
        """Test security review"""
        security_architecture = {
            "authentication": "OAuth2",
            "authorization": "RBAC",
            "data_protection": "encryption",
        }
        requirements = {"security": "critical"}

        review = self.agent._review_security(security_architecture, requirements)

        self.assertIn("security_score", review)
        self.assertIn("vulnerabilities", review)
        self.assertIn("compliance_status", review)
        self.assertIn("security_gaps", review)
        self.assertIn("recommendations", review)

    def test_evaluate_performance(self):
        """Test performance evaluation"""
        architecture_design = {"pattern": "microservices"}
        performance_plan = {"caching_strategy": "Redis", "load_balancing": "HAProxy"}
        requirements = {"performance": "high"}

        evaluation = self.agent._evaluate_performance(
            architecture_design, performance_plan, requirements
        )

        self.assertIn("performance_score", evaluation)
        self.assertIn("latency_analysis", evaluation)
        self.assertIn("throughput_analysis", evaluation)
        self.assertIn("bottlenecks", evaluation)
        self.assertIn("optimization_opportunities", evaluation)

    def test_analyze_maintainability(self):
        """Test maintainability analysis"""
        architecture_design = {"pattern": "microservices"}
        component_breakdown = [
            ArchitectureComponent("api_gateway", ["routing"], ["Python"]),
            ArchitectureComponent("user_service", ["user_management"], ["Python"]),
        ]

        analysis = self.agent._analyze_maintainability(
            architecture_design, component_breakdown
        )

        self.assertIn("maintainability_score", analysis)
        self.assertIn("complexity_analysis", analysis)
        self.assertIn("documentation_requirements", analysis)
        self.assertIn("testing_requirements", analysis)
        self.assertIn("deployment_complexity", analysis)

    def test_analyze_cost_benefit(self):
        """Test cost-benefit analysis"""
        costs = {"total_monthly": 5000, "total_yearly": 60000}
        benefits = {
            "performance_improvement": "50%",
            "scalability_improvement": "100%",
            "maintenance_reduction": "30%",
        }
        requirements = {"scale": "enterprise"}

        analysis = self.agent._analyze_cost_benefit(costs, benefits, requirements)

        self.assertIn("roi_analysis", analysis)
        self.assertIn("payback_period", analysis)
        self.assertIn("cost_effectiveness", analysis)
        self.assertIn("benefit_quantification", analysis)
        self.assertIn("recommendation", analysis)

    def test_evaluate_risks(self):
        """Test risk evaluation"""
        architecture_design = {"pattern": "microservices"}
        migration_plan = {
            "phases": ["phase1", "phase2"],
            "rollback_strategy": "blue_green",
        }
        constraints = {"budget": "limited"}

        evaluation = self.agent._evaluate_risks(
            architecture_design, migration_plan, constraints
        )

        self.assertIn("technical_risks", evaluation)
        self.assertIn("business_risks", evaluation)
        self.assertIn("operational_risks", evaluation)
        self.assertIn("mitigation_strategies", evaluation)
        self.assertIn("risk_score", evaluation)

    def test_make_approval_recommendation(self):
        """Test approval recommendation"""
        architecture_quality = {"overall_quality": 0.8}
        scalability_assessment = {"scalability_score": 0.9}
        security_review = {"security_score": 0.9}
        performance_evaluation = {"performance_score": 0.8}
        maintainability_analysis = {"maintainability_score": 0.7}
        cost_benefit_analysis = {"roi_analysis": "positive"}
        risk_evaluation = {"risk_score": "low"}

        recommendation = self.agent._make_approval_recommendation(
            architecture_quality,
            scalability_assessment,
            security_review,
            performance_evaluation,
            maintainability_analysis,
            cost_benefit_analysis,
            risk_evaluation,
        )

        self.assertIn(recommendation, ["approve", "approve_with_caution", "reject"])

    def test_calculate_final_score(self):
        """Test final score calculation"""
        architecture_quality = {"overall_quality": 0.8}
        scalability_assessment = {"scalability_score": 0.9}
        security_review = {"security_score": 0.9}
        performance_evaluation = {"performance_score": 0.8}
        maintainability_analysis = {"maintainability_score": 0.7}
        cost_benefit_analysis = {"roi_analysis": "positive"}
        risk_evaluation = {"risk_score": "low"}

        score = self.agent._calculate_final_score(
            architecture_quality,
            scalability_assessment,
            security_review,
            performance_evaluation,
            maintainability_analysis,
            cost_benefit_analysis,
            risk_evaluation,
        )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_identify_architecture_patterns(self):
        """Test architecture pattern identification"""
        requirements = {
            "system_type": "web_application",
            "scale": "enterprise",
            "performance": "high",
        }

        patterns = self.agent._identify_architecture_patterns(requirements)

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        for pattern in patterns:
            self.assertIsInstance(pattern, SystemPattern)
            self.assertIn(
                pattern.category,
                ["monolithic", "microservices", "event_driven", "layered"],
            )

    def test_select_appropriate_pattern(self):
        """Test pattern selection"""
        requirements = {
            "system_type": "web_application",
            "scale": "enterprise",
            "team_size": "large",
        }
        patterns = [
            SystemPattern(
                "microservices", "Distributed services", "high", "enterprise"
            ),
            SystemPattern("monolithic", "Single application", "low", "small"),
        ]

        selected_pattern = self.agent._select_appropriate_pattern(
            requirements, patterns
        )

        self.assertIsInstance(selected_pattern, SystemPattern)
        self.assertEqual(selected_pattern.name, "microservices")

    def test_design_microservices_architecture(self):
        """Test microservices architecture design"""
        requirements = {"system_type": "web_application", "scale": "enterprise"}

        design = self.agent._design_microservices_architecture(requirements)

        self.assertEqual(design["pattern"], "microservices")
        self.assertIn("api_gateway", design["components"])
        self.assertIn("service_discovery", design["components"])
        self.assertIn("load_balancer", design["components"])

    def test_design_monolithic_architecture(self):
        """Test monolithic architecture design"""
        requirements = {"system_type": "web_application", "scale": "small"}

        design = self.agent._design_monolithic_architecture(requirements)

        self.assertEqual(design["pattern"], "monolithic")
        self.assertIn("web_layer", design["layers"])
        self.assertIn("business_layer", design["layers"])
        self.assertIn("data_layer", design["layers"])

    def test_design_event_driven_architecture(self):
        """Test event-driven architecture design"""
        requirements = {"system_type": "data_processing", "scale": "enterprise"}

        design = self.agent._design_event_driven_architecture(requirements)

        self.assertEqual(design["pattern"], "event_driven")
        self.assertIn("event_bus", design["components"])
        self.assertIn("event_producers", design["components"])
        self.assertIn("event_consumers", design["components"])

    def test_calculate_complexity_metrics(self):
        """Test complexity metrics calculation"""
        architecture_design = {
            "pattern": "microservices",
            "components": [
                "api_gateway",
                "user_service",
                "data_service",
                "notification_service",
            ],
        }

        metrics = self.agent._calculate_complexity_metrics(architecture_design)

        self.assertIn("component_count", metrics)
        self.assertIn("interaction_complexity", metrics)
        self.assertIn("deployment_complexity", metrics)
        self.assertIn("operational_complexity", metrics)

        self.assertIsInstance(metrics["component_count"], int)
        self.assertIsInstance(metrics["interaction_complexity"], str)
        self.assertIsInstance(metrics["deployment_complexity"], str)
        self.assertIsInstance(metrics["operational_complexity"], str)


class TestArchitectureComponent(unittest.TestCase):
    """Test cases for ArchitectureComponent dataclass"""

    def test_architecture_component_creation(self):
        """Test ArchitectureComponent creation"""
        component = ArchitectureComponent(
            name="api_gateway",
            responsibilities=["routing", "authentication"],
            technologies=["Python", "FastAPI"],
        )

        self.assertEqual(component.name, "api_gateway")
        self.assertEqual(component.responsibilities, ["routing", "authentication"])
        self.assertEqual(component.technologies, ["Python", "FastAPI"])


class TestSystemPattern(unittest.TestCase):
    """Test cases for SystemPattern dataclass"""

    def test_system_pattern_creation(self):
        """Test SystemPattern creation"""
        pattern = SystemPattern(
            name="microservices",
            description="Distributed services architecture",
            complexity="high",
            scale="enterprise",
        )

        self.assertEqual(pattern.name, "microservices")
        self.assertEqual(pattern.description, "Distributed services architecture")
        self.assertEqual(pattern.complexity, "high")
        self.assertEqual(pattern.scale, "enterprise")


if __name__ == "__main__":
    unittest.main()
