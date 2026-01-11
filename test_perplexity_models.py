"""
Test script to find the correct Perplexity model name
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_BASE = "https://api.perplexity.ai"

# Common model name formats to try
MODELS_TO_TRY = [
    "sonar-small-chat",
    "sonar-small-online",
    "sonar-medium-chat",
    "sonar-large-chat",
    "llama-3.1-sonar-small-128k-chat",
    "llama-3.1-sonar-small-128k-online",
    "pplx-sonar-small-chat",
    "pplx-llama-3.1-sonar-small-128k-chat",
]

def test_model(model_name):
    """Test if a model name is valid."""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"\nTesting: {model_name}")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
        if response.status_code == 200:
            print(f"✅ VALID: {model_name}")
            return True
        else:
            error = response.json() if response.text else {}
            error_msg = error.get('error', {}).get('message', response.text)
            print(f"❌ INVALID: {model_name} - {error_msg}")
            
            # Check if there's a list of valid models in the error
            if 'permitted' in response.text.lower() or 'available' in response.text.lower():
                print("   (Check error message for list of valid models)")
            
            return False
    except Exception as e:
        print(f"❌ ERROR testing {model_name}: {str(e)}")
        return False

def check_api_key():
    """Check if API key is valid by making a simple request."""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Try with a very simple request
    payload = {
        "model": "test",  # Invalid model to test auth
        "messages": [{"role": "user", "content": "test"}]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 401:
            print("❌ API Key is INVALID (401 Unauthorized)")
            return False
        elif response.status_code == 400:
            print("✅ API Key is VALID (got 400 for invalid model, not 401)")
            return True
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking API key: {str(e)}")
        return None

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: PERPLEXITY_API_KEY not found in environment")
        exit(1)
    
    print("="*60)
    print("Perplexity API Model Tester")
    print("="*60)
    
    # First check if API key is valid
    print("\n1. Checking API Key...")
    key_valid = check_api_key()
    
    if key_valid is False:
        print("\n❌ Cannot proceed - API key is invalid")
        exit(1)
    
    print("\n2. Testing model names...\n")
    
    valid_models = []
    for model in MODELS_TO_TRY:
        if test_model(model):
            valid_models.append(model)
    
    print(f"\n{'='*60}")
    if valid_models:
        print(f"✅ Valid models found: {valid_models}")
        print(f"\nRecommended: Use '{valid_models[0]}' in config.py")
    else:
        print("❌ No valid models found.")
        print("\nPossible issues:")
        print("   1. Your API key might not have access to chat models")
        print("   2. Model names may have changed - check latest docs")
        print("   3. Visit: https://docs.perplexity.ai/getting-started/models")
        print("   4. Check your Perplexity account dashboard for available models")
        print("\n💡 Tip: The error message should list valid models. Check the full response above.")

