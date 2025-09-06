#!/usr/bin/env python3
"""
Simple runner script for the RNN Cross-Language Limitation Demonstration
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['torch', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to run the demonstration"""
    print("RNN Cross-Language Limitation Demonstration")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\nAll dependencies are satisfied!")
    print("Running the demonstration...\n")
    
    try:
        # Import and run the main demonstration
        from rnn_language_limitation_demo import main as run_demo
        run_demo()
    except Exception as e:
        print(f"Error running demonstration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
