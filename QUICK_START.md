# Quick Start Guide - Streamlit UI

## Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   Create a `.env` file:
   ```
   PERPLEXITY_API_KEY=your_api_key_here
   ```

3. **Run Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   **Note:** If you see a message about `0.0.0.0`, ignore it. Always use `localhost` in your browser.

4. **Access the App**
   Open your browser to: `http://localhost:8501`
   
   **Important:** 
   - ✅ Use `http://localhost:8501` (correct)
   - ❌ Don't use `http://0.0.0.0:8501` (won't work)
   
   The `0.0.0.0` address is only for server bindings, not browser access.

## First Time Setup

1. **Configure API Key** (in sidebar)
   - Enter your Perplexity API key
   - Key should start with `pplx-`

2. **Initialize System** (in sidebar)
   - Click "Initialize System" button
   - Wait for initialization to complete

3. **Build Vector Database** (if needed)
   - Go to "PDF Management" tab
   - Click "Download Default PDFs"
   - Click "Rebuild Vector Database"
   - Wait for processing to complete

4. **Start Chatting**
   - Go to "Chat" tab
   - Ask questions about the research papers!

## Features

### Chat Interface
- Ask questions about the research papers
- View conversation history
- See retrieved context and metadata
- Clear chat history when needed

### PDF Management
- Upload new PDF files (up to 50MB)
- Download default research papers
- Rebuild vector database
- View existing PDFs

### System Status
- Real-time system health checks
- API key validation
- Vector database status
- Error monitoring

## Validation Features

The application includes comprehensive validation:

- ✅ **API Key Validation**: Format and presence checks
- ✅ **Query Validation**: Length, content, and security checks
- ✅ **File Upload Validation**: Size, type, and content checks
- ✅ **System Status Checks**: Before each operation
- ✅ **Error Handling**: Graceful error messages and logging

## Troubleshooting

### "API key not valid"
- Check that your API key starts with `pplx-`
- Verify the key is correctly set in `.env` or sidebar
- Ensure the key hasn't expired

### "Vector database not found"
- Go to PDF Management tab
- Download PDFs and rebuild vector database
- Wait for processing to complete

### "System not initialized"
- Click "Initialize System" in sidebar
- Check system status for errors
- Review logs in `logs/rag_app.log`

### Port already in use
```bash
# Find and kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

## Next Steps

- See `AWS_EC2_DEPLOYMENT.md` for production deployment
- Check `README.md` for full documentation
- Review `PERPLEXITY_SETUP_GUIDE.md` for API setup

