"""
Ollama Setup Script

This script helps users install and configure Ollama for the RAG application.
"""

import subprocess
import sys
import platform
import requests
import time
from pathlib import Path

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama is not installed")
        return False

def install_ollama():
    """Install Ollama based on the operating system."""
    system = platform.system().lower()
    
    print(f"🖥️ Detected OS: {system}")
    
    if system == "windows":
        print("📥 Installing Ollama for Windows...")
        print("Please visit: https://ollama.ai/download/windows")
        print("Download and run the installer.")
        input("Press Enter after installation is complete...")
        
    elif system == "darwin":  # macOS
        print("📥 Installing Ollama for macOS...")
        try:
            subprocess.run(["brew", "install", "ollama"], check=True)
            print("✅ Ollama installed via Homebrew")
        except subprocess.CalledProcessError:
            print("❌ Homebrew not found. Please install Ollama manually:")
            print("Visit: https://ollama.ai/download/mac")
            
    elif system == "linux":
        print("📥 Installing Ollama for Linux...")
        try:
            subprocess.run([
                "curl", "-fsSL", "https://ollama.ai/install.sh"
            ], check=True)
            print("✅ Ollama installed via install script")
        except subprocess.CalledProcessError:
            print("❌ Installation failed. Please install manually:")
            print("Visit: https://ollama.ai/download/linux")
    
    else:
        print(f"❌ Unsupported OS: {system}")
        return False
    
    return True

def start_ollama_service():
    """Start Ollama service."""
    print("🚀 Starting Ollama service...")
    
    try:
        # Start Ollama in background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        print("⏳ Waiting for Ollama service to start...")
        time.sleep(5)
        
        # Check if service is running
        if check_ollama_running():
            print("✅ Ollama service is running")
            return True
        else:
            print("❌ Ollama service failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Ollama: {str(e)}")
        return False

def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def pull_models():
    """Pull required models."""
    models = ["llama3", "mistral", "phi3"]
    
    print("📦 Pulling required models...")
    
    for model in models:
        print(f"📥 Pulling {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {model} pulled successfully")
            else:
                print(f"❌ Failed to pull {model}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout pulling {model}")
        except Exception as e:
            print(f"❌ Error pulling {model}: {str(e)}")

def list_available_models():
    """List available models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("📋 Available models:")
            print(result.stdout)
        else:
            print("❌ Failed to list models")
            
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")

def test_ollama():
    """Test Ollama functionality."""
    print("🧪 Testing Ollama...")
    
    try:
        result = subprocess.run([
            "ollama", "run", "llama3", "Hello, how are you?"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Ollama test successful")
            print(f"Response: {result.stdout[:100]}...")
            return True
        else:
            print(f"❌ Ollama test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Ollama test timeout")
        return False
    except Exception as e:
        print(f"❌ Ollama test error: {str(e)}")
        return False

def main():
    """Main setup function."""
    print("🦙 Ollama Setup for RAG Application")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n📥 Installing Ollama...")
        if not install_ollama():
            print("❌ Installation failed. Please install Ollama manually.")
            return
        
        # Check again after installation
        if not check_ollama_installed():
            print("❌ Ollama installation verification failed.")
            return
    
    # Start Ollama service
    if not check_ollama_running():
        if not start_ollama_service():
            print("❌ Failed to start Ollama service.")
            return
    
    # Pull models
    print("\n📦 Setting up models...")
    pull_models()
    
    # List models
    print("\n📋 Current models:")
    list_available_models()
    
    # Test Ollama
    print("\n🧪 Testing Ollama...")
    if test_ollama():
        print("\n🎉 Ollama setup completed successfully!")
        print("You can now run the RAG application.")
    else:
        print("\n⚠️ Ollama setup completed with warnings.")
        print("Please check the configuration manually.")

if __name__ == "__main__":
    main()
