"""
Perplexity API Setup Script

This script helps users configure Perplexity API for the RAG application.
"""

import os
import sys
from pathlib import Path

def setup_perplexity():
    """Setup Perplexity API configuration."""
    print("🔮 Perplexity API Setup for RAG Application")
    print("=" * 45)
    
    print("📋 Perplexity API Setup Steps:")
    print("1. Visit: https://www.perplexity.ai/settings/api")
    print("2. Sign up or log in to your account")
    print("3. Generate an API key")
    print("4. Copy the API key")
    print()
    
    api_key = input("Enter your Perplexity API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return False
    
    # Update config
    update_config({
        "PERPLEXITY_API_KEY": api_key,
        "API_PROVIDER": "perplexity",
        "PERPLEXITY_MODEL": "sonar-small-chat",
        "PERPLEXITY_API_BASE": "https://api.perplexity.ai"
    })
    
    print("✅ Perplexity configuration saved!")
    return True

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

def check_perplexity_key():
    """Check if Perplexity API key is configured."""
    print("🔍 Checking Perplexity API key configuration...")
    
    config_path = Path("config.py")
    if not config_path.exists():
        print("❌ config.py not found!")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for empty API key
    if 'PERPLEXITY_API_KEY = ""' in content:
        print("⚠️ Perplexity API key not configured")
        print("Run this script to configure API key.")
        return False
    else:
        print("✅ Perplexity API key appears to be configured!")
        return True

def test_perplexity_connection():
    """Test Perplexity API connection."""
    print("🧪 Testing Perplexity API connection...")
    
    try:
        from src.hybrid_llm_client import HybridLLMClient
        
        client = HybridLLMClient()
        
        if client.test_connection():
            print("✅ Perplexity API connection successful!")
            return True
        else:
            print("❌ Perplexity API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {str(e)}")
        return False

def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_perplexity_key()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_perplexity_connection()
    else:
        setup_perplexity()

if __name__ == "__main__":
    main()
