"""
CodeConductor CLI - Command Line Interface

Provides easy access to CodeConductor functionality from the command line.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from codeconductor.app import CodeConductorApp
from codeconductor.dashboard import ValidationDashboard
from codeconductor.logger import ValidationLogger


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CodeConductor - Personal AI Development Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codeconductor app          # Start the main Streamlit app
  codeconductor dashboard    # Start the validation dashboard
  codeconductor test         # Run automated tests
  codeconductor validate     # Validate code quality
        """
    )
    
    parser.add_argument(
        "command",
        choices=["app", "dashboard", "test", "validate", "version"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for web interface (default: 8501)"
    )
    
    parser.add_argument(
        "--host", "-H",
        default="localhost",
        help="Host for web interface (default: localhost)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.command == "version":
        from codeconductor import __version__
        print(f"CodeConductor v{__version__}")
        return
    
    elif args.command == "app":
        print("🚀 Starting CodeConductor App...")
        print(f"📱 Web interface: http://{args.host}:{args.port}")
        print("🛑 Press Ctrl+C to stop")
        
        # Set environment variables for Streamlit
        os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        app = CodeConductorApp()
        app.run()
    
    elif args.command == "dashboard":
        print("📊 Starting Validation Dashboard...")
        print(f"📱 Web interface: http://{args.host}:{args.port}")
        print("🛑 Press Ctrl+C to stop")
        
        # Set environment variables for Streamlit
        os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        dashboard = ValidationDashboard()
        dashboard.run()
    
    elif args.command == "test":
        print("🧪 Running automated tests...")
        import pytest
        
        # Run tests
        test_args = [
            "tests/",
            "-v",
            "--tb=short",
            "--disable-warnings"
        ]
        
        if args.debug:
            test_args.append("--pdb")
        
        exit_code = pytest.main(test_args)
        sys.exit(exit_code)
    
    elif args.command == "validate":
        print("✅ Running code validation...")
        
        # Import validation system
        from codeconductor.feedback.validation_system import validate_cursor_output
        
        # Example validation
        test_code = """
def example_function():
    return "Hello, World!"
        """
        
        result = validate_cursor_output(test_code)
        print(f"Validation result: {result}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 