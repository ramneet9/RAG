"""
API Key Setup Script

This script helps users configure API keys for the RAG application.
"""

import os
import sys
from pathlib import Path

def setup_api_keys():
    """Interactive API key setup."""
    print("🔑 RAG Application API Key Setup")
    print("=" * 40)
    
    # Choose provider
    print("Choose your API provider:")
    print("1. OpenAI (GPT models + embeddings)")
    print("2. Anthropic (Claude models)")
    print("3. Hugging Face (Open source models)")
    print("4. Cohere (Business models)")
    print("5. Mixed (OpenAI embeddings + Anthropic responses)")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        setup_openai()
    elif choice == "2":
        setup_anthropic()
    elif choice == "3":
        setup_huggingface()
    elif choice == "4":
        setup_cohere()
    elif choice == "5":
        setup_mixed()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n✅ API keys configured!")
    print("You can now run the RAG application with: python main.py")

def setup_openai():
    """Setup OpenAI configuration."""
    print("\n🔵 OpenAI Setup")
    print("Get your API key from: https://platform.openai.com/")
    
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    # Update config
    update_config({
        "API_PROVIDER": "openai",
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL": "gpt-3.5-turbo",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
    })
    
    print("✅ OpenAI configuration saved!")

def setup_anthropic():
    """Setup Anthropic configuration."""
    print("\n🟣 Anthropic Setup")
    print("Get your API key from: https://console.anthropic.com/")
    
    api_key = input("Enter your Anthropic API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    # Update config
    update_config({
        "API_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229"
    })
    
    print("✅ Anthropic configuration saved!")
    print("⚠️ Note: Anthropic doesn't provide embeddings. You'll need OpenAI for embeddings.")

def setup_huggingface():
    """Setup Hugging Face configuration."""
    print("\n🟠 Hugging Face Setup")
    print("Get your API key from: https://huggingface.co/settings/tokens")
    
    api_key = input("Enter your Hugging Face API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    # Update config
    update_config({
        "API_PROVIDER": "huggingface",
        "HUGGINGFACE_API_KEY": api_key,
        "HUGGINGFACE_MODEL": "microsoft/DialoGPT-medium",
        "HUGGINGFACE_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    })
    
    print("✅ Hugging Face configuration saved!")

def setup_cohere():
    """Setup Cohere configuration."""
    print("\n🔴 Cohere Setup")
    print("Get your API key from: https://cohere.com/")
    
    api_key = input("Enter your Cohere API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    # Update config
    update_config({
        "API_PROVIDER": "cohere",
        "COHERE_API_KEY": api_key,
        "COHERE_MODEL": "command",
        "COHERE_EMBEDDING_MODEL": "embed-english-v3.0"
    })
    
    print("✅ Cohere configuration saved!")

def setup_mixed():
    """Setup mixed configuration (OpenAI + Anthropic)."""
    print("\n🔄 Mixed Setup (Recommended)")
    print("This uses OpenAI for embeddings and Anthropic for responses.")
    
    # OpenAI setup
    print("\nOpenAI Setup (for embeddings):")
    print("Get your API key from: https://platform.openai.com/")
    openai_key = input("Enter your OpenAI API key: ").strip()
    
    # Anthropic setup
    print("\nAnthropic Setup (for responses):")
    print("Get your API key from: https://console.anthropic.com/")
    anthropic_key = input("Enter your Anthropic API key: ").strip()
    
    if not openai_key or not anthropic_key:
        print("❌ Both API keys are required!")
        return
    
    # Update config
    update_config({
        "API_PROVIDER": "openai",  # Use OpenAI for embeddings
        "OPENAI_API_KEY": openai_key,
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "ANTHROPIC_API_KEY": anthropic_key,
        "ANTHROPIC_MODEL": "claude-3-sonnet-20240229"
    })
    
    print("✅ Mixed configuration saved!")

def update_config(updates):
    """Update config.py with new values."""
    config_path = Path("config.py")
    
    if not config_path.exists():
        print("❌ config.py not found!")
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update values
    for key, value in updates.items():
        # Find the line and update it
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(f"{key} ="):
                lines[i] = f'{key} = "{value}"'
                break
        content = '\n'.join(lines)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {len(updates)} configuration values")

def check_api_keys():
    """Check if API keys are configured."""
    print("🔍 Checking API key configuration...")
    
    config_path = Path("config.py")
    if not config_path.exists():
        print("❌ config.py not found!")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for empty API keys
    empty_keys = []
    if 'OPENAI_API_KEY = ""' in content:
        empty_keys.append("OpenAI")
    if 'ANTHROPIC_API_KEY = ""' in content:
        empty_keys.append("Anthropic")
    if 'HUGGINGFACE_API_KEY = ""' in content:
        empty_keys.append("Hugging Face")
    if 'COHERE_API_KEY = ""' in content:
        empty_keys.append("Cohere")
    
    if empty_keys:
        print(f"⚠️ Empty API keys found: {', '.join(empty_keys)}")
        print("Run this script to configure API keys.")
        return False
    else:
        print("✅ API keys appear to be configured!")
        return True

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_api_keys()
    else:
        setup_api_keys()

if __name__ == "__main__":
    main()
