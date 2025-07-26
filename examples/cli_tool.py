#!/usr/bin/env python3
"""
Simple CLI Tool for Smoke Testing
A basic command-line tool that we can enhance with the pipeline
"""

import argparse
import sys
import json
from pathlib import Path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Simple CLI Tool for Testing")
    parser.add_argument("--input", "-i", help="Input file to process")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("CLI Tool started")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
    
    # Simple processing
    if args.input:
        try:
            with open(args.input, 'r') as f:
                data = f.read()
            
            # Process data (simple example)
            processed_data = data.upper()
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(processed_data)
                print(f"Processed data written to {args.output}")
            else:
                print(processed_data)
                
        except FileNotFoundError:
            print(f"Error: File {args.input} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)
    else:
        print("No input file specified. Use --help for usage.")
        sys.exit(1)

if __name__ == "__main__":
    main() 