"""
RAG Application Runner

This script provides a comprehensive interface to run the RAG application
with different modes and options.
"""

import sys
import argparse
from pathlib import Path
import logging

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rag_app.log'),
            logging.StreamHandler()
        ]
    )

def run_full_evaluation():
    """Run the complete RAG application with evaluation."""
    print("🚀 Running Full RAG Application Evaluation...")
    
    try:
        from main import main
        main()
    except Exception as e:
        print(f"❌ Full evaluation failed: {str(e)}")
        return False
    
    return True

def run_interactive_demo():
    """Run interactive demo."""
    print("🤖 Running Interactive Demo...")
    
    try:
        from demo import interactive_demo
        interactive_demo()
    except Exception as e:
        print(f"❌ Interactive demo failed: {str(e)}")
        return False
    
    return True

def run_quick_demo():
    """Run quick demo with predefined questions."""
    print("⚡ Running Quick Demo...")
    
    try:
        from demo import quick_demo
        quick_demo()
    except Exception as e:
        print(f"❌ Quick demo failed: {str(e)}")
        return False
    
    return True

def run_tests():
    """Run component tests."""
    print("🧪 Running Component Tests...")
    
    try:
        from test import main as test_main
        test_main()
    except Exception as e:
        print(f"❌ Tests failed: {str(e)}")
        return False
    
    return True

def setup_environment():
    """Setup the environment."""
    print("🔧 Setting up Environment...")
    
    try:
        from setup_environment import main as setup_main
        setup_main()
    except Exception as e:
        print(f"❌ Environment setup failed: {str(e)}")
        return False
    
    return True

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='RAG Application Runner')
    
    parser.add_argument('mode', choices=[
        'full', 'interactive', 'quick', 'test', 'setup'
    ], help='Mode to run the application')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--no-setup', action='store_true',
                       help='Skip environment setup check')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if setup is needed
    if not args.no_setup:
        if not Path('vector_db').exists() or not Path('data').exists():
            print("⚠️ Environment not fully set up. Running setup...")
            if not setup_environment():
                print("❌ Setup failed. Exiting.")
                sys.exit(1)
    
    # Run selected mode
    success = False
    
    if args.mode == 'full':
        success = run_full_evaluation()
    elif args.mode == 'interactive':
        success = run_interactive_demo()
    elif args.mode == 'quick':
        success = run_quick_demo()
    elif args.mode == 'test':
        success = run_tests()
    elif args.mode == 'setup':
        success = setup_environment()
    
    if success:
        print("✅ Operation completed successfully!")
    else:
        print("❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
