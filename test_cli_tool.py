#!/usr/bin/env python3
"""
Basic CLI tool for testing purposes.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Simple CLI Tool for Testing")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Read input file
    try:
        with open(args.input, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process content (convert to uppercase for testing)
    processed_content = content.upper()
    
    # Write output if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(processed_content)
            if args.verbose:
                print(f"Output written to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(processed_content)
    
    sys.exit(0)

if __name__ == "__main__":
    main() 