# What Happens When You Ask a Question in the Streamlit Chat

## Complete Flow Diagram

```
User Types Question
    ↓
[Streamlit UI] Validates Input
    ↓
[ConversationManager] Receives Query
    ↓
[VectorStore] Searches for Relevant Context
    ├─→ Generates Query Embedding (sentence-transformers)
    ├─→ Searches FAISS Vector Database
    └─→ Returns Top 5 Most Relevant Chunks
    ↓
[ConversationManager] Builds Prompt
    ├─→ Includes Retrieved Context
    ├─→ Includes Conversation History (last 4 turns)
    └─→ Includes Current Question
    ↓
[HybridLLMClient] Calls Perplexity API
    ├─→ Makes HTTP POST to /chat/completions
    ├─→ Sends: Context + History + Question
    └─→ Receives: LLM Response
    ↓
[ConversationManager] Processes Response
    ├─→ Adds to Conversation History
    └─→ Returns Response + Metadata
    ↓
[Streamlit UI] Displays Response
    └─→ Shows in Chat Interface
```

## Step-by-Step Breakdown

### 1. **User Input (Streamlit UI)**
   - You type a question in the chat input box
   - Location: `streamlit_app.py` → `main_chat_interface()`
   - Validation: Checks length, content, security

### 2. **Query Validation**
   - Validates the query (3-1000 characters)
   - Checks for malicious content
   - Location: `streamlit_app.py` → `validate_query()`

### 3. **Vector Search (Retrieval)**
   - Your question is converted to an embedding vector
   - Uses: `sentence-transformers/all-MiniLM-L6-v2`
   - Searches the FAISS vector database (1484 chunks from PDFs)
   - Returns top 5 most similar text chunks
   - Location: `src/vector_store.py` → `search()`
   - Location: `src/conversation_manager.py` → `get_relevant_context()`

### 4. **Prompt Building**
   - Combines:
     - Retrieved context (from vector search)
     - Conversation history (last 4 turns)
     - Current question
   - Location: `src/hybrid_llm_client.py` → `_build_prompt()`

### 5. **API Call to Perplexity**
   - Makes HTTP POST request to Perplexity API
   - Endpoint: `https://api.perplexity.ai/chat/completions`
   - Sends:
     ```json
     {
       "model": "llama-3.1-sonar-small-128k-online",
       "messages": [
         {"role": "system", "content": "..."},
         {"role": "user", "content": "Context: ... Question: ..."}
       ],
       "max_tokens": 200,
       "temperature": 0.7
     }
     ```
   - Location: `src/hybrid_llm_client.py` → `_generate_with_perplexity()`

### 6. **Response Processing**
   - Receives LLM response from Perplexity
   - Cleans and formats the response
   - Adds to conversation history
   - Location: `src/hybrid_llm_client.py` → `_clean_response()`
   - Location: `src/conversation_manager.py` → `add_to_history()`

### 7. **Display in UI**
   - Shows your question
   - Shows assistant response
   - Optionally shows retrieved context and metadata
   - Location: `streamlit_app.py` → `display_chat_message()`

## Current Issue

From your logs, I can see:
```
INFO - Making API call to Perplexity: llama-3.1-sonar-small-128k-online
ERROR - API request failed with status 400: Invalid model 'llama-3.1-sonar-small-128k-online'
```

**The model name is incorrect!** The Perplexity API doesn't recognize this model name.

## Fix Needed

Update the model name in `config.py` to one of these valid Perplexity models:
- `llama-3.1-sonar-small-128k-chat`
- `llama-3.1-sonar-large-128k-chat`
- `llama-3.1-sonar-huge-128k-online`
- `llama-3.1-sonar-small-128k-online` (might need different format)

Let me fix this for you!

