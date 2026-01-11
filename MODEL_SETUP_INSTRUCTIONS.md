# Perplexity Model Setup Instructions

## Current Issue

Your API key is **VALID** ✅, but the model names in the config are not recognized by the Perplexity API.

## How to Find the Correct Model Name

### Option 1: Check Perplexity Documentation
1. Visit: https://docs.perplexity.ai/getting-started/models
2. Look for the list of available models
3. Copy the exact model name (case-sensitive)

### Option 2: Check Your Perplexity Dashboard
1. Log into your Perplexity account
2. Go to API settings or dashboard
3. Check which models are available for your account tier

### Option 3: Use the Test Script
1. Update `test_perplexity_models.py` with model names from the docs
2. Run: `python test_perplexity_models.py`
3. It will test each model and tell you which ones work

## Update Config

Once you find the correct model name:

1. Open `config.py`
2. Find the line: `PERPLEXITY_MODEL = "sonar-small-chat"`
3. Replace with your correct model name
4. Save the file
5. Restart your Streamlit app

## Current Status

- ✅ API Key: Valid
- ❌ Model Name: Invalid (needs update)
- ✅ Fallback: Working (app will use keyword-based responses until model is fixed)

## Example Config Update

```python
# In config.py, change:
PERPLEXITY_MODEL = "your-correct-model-name-here"
```

## Testing

After updating the model name:
1. Restart Streamlit: `streamlit run streamlit_app.py`
2. Ask a question
3. Check logs for: `"Successfully received response from Perplexity API"`
4. If you see this, the model name is correct! ✅

## Fallback Behavior

Until you fix the model name, the app will:
- Still work and respond to questions
- Use a keyword-based fallback system
- Show responses based on context matching
- Log API errors (but continue functioning)

This ensures your app works while you find the correct model name.

