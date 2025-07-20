#!/usr/bin/env python3
"""
Multi-Agent Discussion Demo for CodeConductor

This demo showcases the multi-agent discussion system with:
- CodeGenAgent: Specialized in code generation and templates
- ArchitectAgent: Specialized in system architecture and design
- ReviewAgent: Specialized in code review and quality assessment
- AgentOrchestrator: Coordinates the discussion and reaches consensus

The demo simulates a real-world scenario where agents collaborate to:
1. Analyze requirements for a new web application
2. Propose architectural and implementation solutions
3. Review and approve the final solution
"""

import sys
import os

import time
from typing import Dict, Any

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import AgentOrchestrator
from agents.codegen_agent import CodeGenAgent
from agents.architect_agent import ArchitectAgent
from agents.review_agent import ReviewAgent


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section"""
    print(f"\n--- {title} ---")


def print_agent_output(agent_name: str, output: Dict[str, Any], phase: str):
    """Print formatted agent output"""
    print(f"\n🤖 {agent_name} - {phase.upper()}")
    print("-" * 60)

    if "error" in output:
        print(f"❌ Error: {output['error']}")
        return

    # Print key highlights based on phase and agent
    if phase == "analysis":
        if agent_name == "CodeGenAgent":
            print(
                f"📋 Requirements Analysis: {output.get('requirements_analysis', {}).get('functionality', 'N/A')}"
            )
            print(
                f"🎯 Generation Feasibility: {output.get('generation_feasibility', 'N/A')}"
            )
            print(f"⏱️  Estimated Effort: {output.get('estimated_effort', 'N/A')}")
        elif agent_name == "ArchitectAgent":
            print(
                f"🏗️  Current Architecture: {output.get('current_architecture_assessment', {}).get('pattern_analysis', 'N/A')}"
            )
            print(
                f"📈 Scalability Analysis: {output.get('scalability_analysis', {}).get('scaling_strategy', 'N/A')}"
            )
            print(
                f"🔒 Security Analysis: {output.get('security_analysis', {}).get('security_priorities', 'N/A')}"
            )
        elif agent_name == "ReviewAgent":
            print(f"📊 Code Quality Score: {output.get('code_quality_score', 0):.2f}")
            print(f"🔍 Issues Found: {len(output.get('issues_found', []))}")
            print(f"💡 Recommendations: {len(output.get('recommendations', []))}")

    elif phase == "proposal":
        if agent_name == "CodeGenAgent":
            print(f"📝 Code Templates: {len(output.get('code_templates', []))}")
            print(
                f"📋 Implementation Plan: {len(output.get('implementation_plan', []))} steps"
            )
            print(f"🔧 Dependencies: {len(output.get('dependencies', []))}")
        elif agent_name == "ArchitectAgent":
            print(
                f"🏗️  Architecture Pattern: {output.get('architecture_design', {}).get('pattern', 'N/A')}"
            )
            print(f"🧩 Components: {len(output.get('component_breakdown', []))}")
            print(
                f"💰 Cost Estimation: ${output.get('cost_estimation', {}).get('total_monthly', 0)}/month"
            )
        elif agent_name == "ReviewAgent":
            print(
                f"📋 Improvement Plan: {len(output.get('improvement_plan', []))} categories"
            )
            print(
                f"🔧 Refactoring Suggestions: {len(output.get('refactoring_suggestions', []))}"
            )
            print(
                f"⚡ Performance Optimizations: {len(output.get('performance_optimizations', []))}"
            )

    elif phase == "review":
        if agent_name == "CodeGenAgent":
            print(
                f"✅ Code Quality Assessment: {output.get('code_quality_assessment', {}).get('readability_score', 0):.2f}"
            )
            print(
                f"🔒 Security Analysis: {output.get('security_analysis', {}).get('security_score', 0):.2f}"
            )
            print(f"📊 Final Score: {output.get('final_score', 0):.2f}")
        elif agent_name == "ArchitectAgent":
            print(
                f"🏗️  Architecture Quality: {output.get('architecture_quality', {}).get('overall_quality', 0):.2f}"
            )
            print(
                f"📈 Scalability Assessment: {output.get('scalability_assessment', {}).get('scalability_score', 0):.2f}"
            )
            print(f"📊 Final Score: {output.get('final_score', 0):.2f}")
        elif agent_name == "ReviewAgent":
            print(
                f"🔍 Code Quality Assessment: {output.get('code_quality_assessment', {}).get('readability_score', 0):.2f}"
            )
            print(
                f"🔒 Security Review: {output.get('security_analysis', {}).get('security_score', 0):.2f}"
            )
            print(f"📊 Final Score: {output.get('final_score', 0):.2f}")


def print_discussion_summary(result: Dict[str, Any]):
    """Print discussion summary"""
    print_header("DISCUSSION SUMMARY")

    print(
        f"🎯 Consensus Reached: {'✅ Yes' if result['consensus_reached'] else '❌ No'}"
    )
    print(f"📋 Final Decision: {result['final_decision']}")
    print(f"📊 Consensus Score: {result['consensus_score']:.2f}")
    print(f"🔄 Discussion Rounds: {len(result['discussion_rounds'])}")

    print_section("Agent Contributions")
    for agent, contributions in result["agent_contributions"].items():
        print(f"  🤖 {agent}: {contributions} contributions")

    if result["discussion_rounds"]:
        print_section("Discussion Flow")
        for i, round_data in enumerate(result["discussion_rounds"], 1):
            print(
                f"  📍 Round {i}: {'Consensus reached' if round_data.consensus_reached else 'No consensus'}"
            )
            if round_data.final_decision:
                print(f"     Decision: {round_data.final_decision}")

    if "errors" in result and result["errors"]:
        print_section("Errors Encountered")
        for error in result["errors"]:
            print(f"  ❌ {error['agent']} ({error['phase']}): {error['error']}")


def create_sample_context() -> Dict[str, Any]:
    """Create a sample context for the demo"""
    return {
        "requirements": {
            "functionality": "Create a scalable e-commerce platform",
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
            "message_queue": "RabbitMQ",
            "input_validation": True,
            "error_handling": True,
            "documentation": True,
            "testing": True,
            "monitoring": True,
            "logging": True,
            "security": "critical",
            "performance": "high",
            "availability": "99.9%",
            "scalability": "enterprise",
            "user_count": "10000+",
            "data_volume": "large",
        },
        "constraints": {
            "budget": "limited",
            "time_to_market": "fast",
            "team_size": "small",
            "existing_infrastructure": "cloud",
            "compliance": ["GDPR", "PCI-DSS"],
            "performance_critical": True,
            "security_critical": True,
            "maintainability": "high",
        },
        "existing_code": '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

app = FastAPI(title="E-commerce API")

class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None

@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Get product by ID."""
    # TODO: Implement database query
    return {"id": product_id, "name": "Sample Product", "price": 99.99}

@app.post("/products/")
def create_product(product: Product):
    """Create a new product."""
    # TODO: Implement database insert
    return product
''',
        "existing_architecture": {
            "pattern": "monolithic",
            "components": ["web_server", "database"],
            "technologies": ["Python", "FastAPI", "PostgreSQL"],
            "deployment": "single_server",
            "scaling": "vertical",
            "security": "basic",
            "monitoring": "minimal",
        },
    }


def run_demo():
    """Run the multi-agent discussion demo"""
    print_header("MULTI-AGENT DISCUSSION DEMO")
    print("CodeConductor - Phase 3: Multi-agent Discussion System")
    print("\nThis demo showcases how specialized agents collaborate to:")
    print("1. Analyze requirements and existing code")
    print("2. Propose architectural and implementation solutions")
    print("3. Review and approve the final solution")
    print("4. Reach consensus through orchestrated discussion")

    # Create sample context
    context = create_sample_context()

    print_section("Initial Context")
    print(f"📋 Functionality: {context['requirements']['functionality']}")
    print(
        f"🔧 Technology Stack: {context['requirements']['language']} + {context['requirements']['framework']}"
    )
    print(
        f"📊 Scale: {context['requirements']['scalability']} ({context['requirements']['user_count']} users)"
    )
    print(f"🔒 Security: {context['requirements']['security']}")
    print(f"💰 Budget: {context['constraints']['budget']}")

    # Initialize agents
    print_section("Initializing Agents")

    codegen_agent = CodeGenAgent(
        "CodeGenAgent",
        {
            "codegen": {
                "default_language": "Python",
                "code_style": "pep8",
                "framework_preferences": ["FastAPI", "Django", "Flask"],
            }
        },
    )

    architect_agent = ArchitectAgent(
        "ArchitectAgent",
        {
            "architect": {
                "default_pattern": "microservices",
                "preferred_cloud": "AWS",
                "scaling_strategy": "horizontal",
            }
        },
    )

    review_agent = ReviewAgent(
        "ReviewAgent",
        {
            "review": {
                "quality_threshold": 0.8,
                "security_focus": True,
                "performance_focus": True,
            }
        },
    )

    agents = [codegen_agent, architect_agent, review_agent]

    print("🤖 CodeGenAgent: Specialized in code generation and templates")
    print("🏗️  ArchitectAgent: Specialized in system architecture and design")
    print("🔍 ReviewAgent: Specialized in code review and quality assessment")

    # Initialize orchestrator
    print_section("Initializing Orchestrator")
    orchestrator = AgentOrchestrator(
        agents=agents,
        config={
            "consensus_strategy": "weighted_majority",
            "max_rounds": 3,
            "consensus_threshold": 0.7,
        },
    )

    print(f"🎯 Consensus Strategy: {orchestrator.config['consensus_strategy']}")
    print(f"🔄 Max Rounds: {orchestrator.config['max_rounds']}")
    print(f"📊 Consensus Threshold: {orchestrator.config['consensus_threshold']}")

    # Run the discussion
    print_header("STARTING MULTI-AGENT DISCUSSION")
    print("The agents will now analyze, propose, and review the solution...")

    start_time = time.time()
    result = orchestrator.run_discussion(context)
    end_time = time.time()

    print(f"\n⏱️  Discussion completed in {end_time - start_time:.2f} seconds")

    # Print detailed results
    print_header("DETAILED RESULTS")

    # Print each round's results
    for i, round_data in enumerate(result["discussion_rounds"], 1):
        print_section(f"Round {i} Results")

        # Analysis phase
        print("\n📊 ANALYSIS PHASE:")
        for agent_name, analysis in round_data.analysis_results.items():
            print_agent_output(agent_name, analysis, "analysis")

        # Proposal phase
        print("\n💡 PROPOSAL PHASE:")
        for agent_name, proposal in round_data.proposal_results.items():
            print_agent_output(agent_name, proposal, "proposal")

        # Review phase (if consensus not reached)
        if not round_data.consensus_reached:
            print("\n🔍 REVIEW PHASE:")
            # Simulate review phase results
            for agent_name in ["CodeGenAgent", "ArchitectAgent", "ReviewAgent"]:
                review_result = {
                    "code_quality_assessment": {"readability_score": 0.8},
                    "security_analysis": {"security_score": 0.9},
                    "final_score": 0.85,
                    "approval_recommendation": "approve",
                }
                print_agent_output(agent_name, review_result, "review")

    # Print final summary
    print_discussion_summary(result)

    # Print recommendations
    print_header("RECOMMENDATIONS")

    if result["consensus_reached"]:
        print("✅ The agents have reached consensus! Here are the key recommendations:")

        # Extract key recommendations from the final round
        final_round = result["discussion_rounds"][-1]

        print_section("Architecture Recommendations")
        for agent_name, proposal in final_round.proposal_results.items():
            if agent_name == "ArchitectAgent" and "architecture_design" in proposal:
                arch_design = proposal["architecture_design"]
                print(f"🏗️  Pattern: {arch_design.get('pattern', 'N/A')}")
                print(f"🧩 Components: {', '.join(arch_design.get('components', []))}")
                print(
                    f"💰 Estimated Cost: ${proposal.get('cost_estimation', {}).get('total_monthly', 0)}/month"
                )

        print_section("Implementation Recommendations")
        for agent_name, proposal in final_round.proposal_results.items():
            if agent_name == "CodeGenAgent" and "implementation_plan" in proposal:
                plan = proposal["implementation_plan"]
                print(f"📋 Implementation Steps: {len(plan)}")
                for i, step in enumerate(plan[:3], 1):  # Show first 3 steps
                    print(f"  {i}. {step}")
                if len(plan) > 3:
                    print(f"  ... and {len(plan) - 3} more steps")

        print_section("Quality Recommendations")
        for agent_name, proposal in final_round.proposal_results.items():
            if agent_name == "ReviewAgent" and "improvement_plan" in proposal:
                improvements = proposal["improvement_plan"]
                print(f"🔧 Improvement Categories: {len(improvements)}")
                for improvement in improvements[:3]:  # Show first 3
                    print(
                        f"  • {improvement.get('category', 'N/A')}: {improvement.get('action', 'N/A')}"
                    )

    else:
        print("❌ The agents could not reach consensus. Here are the key issues:")

        # Analyze why consensus wasn't reached
        print_section("Consensus Issues")
        print("The agents may have different priorities or conflicting requirements.")
        print("Consider:")
        print("  • Adjusting requirements or constraints")
        print("  • Providing more detailed context")
        print("  • Using a different consensus strategy")
        print("  • Increasing the number of discussion rounds")

    print_header("DEMO COMPLETED")
    print(
        "This demo showcases the power of multi-agent collaboration in software development."
    )
    print(
        "The agents work together to provide comprehensive analysis, proposals, and reviews."
    )
    print("The orchestrator ensures effective coordination and consensus building.")


def run_quick_demo():
    """Run a quick version of the demo for testing"""
    print_header("QUICK MULTI-AGENT DISCUSSION DEMO")

    # Simplified context
    context = {
        "requirements": {
            "functionality": "Simple REST API",
            "language": "Python",
            "framework": "FastAPI",
        },
        "constraints": {"budget": "limited", "time_to_market": "fast"},
    }

    # Initialize agents
    agents = [
        CodeGenAgent("CodeGenAgent"),
        ArchitectAgent("ArchitectAgent"),
        ReviewAgent("ReviewAgent"),
    ]

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        agents=agents, config={"consensus_strategy": "majority", "max_rounds": 1}
    )

    print("Running quick discussion...")
    result = orchestrator.run_discussion(context)

    print(f"\nConsensus: {result['consensus_reached']}")
    if result["consensus"]:
        print(f"Decision: {result['consensus'].get('approval', 'N/A')}")
    else:
        print("Decision: No consensus reached")
    print(f"Rounds: {result['discussion_rounds']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Discussion Demo")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick demo for testing"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_demo()
    else:
        run_demo()
