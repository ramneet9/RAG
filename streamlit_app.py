"""
Streamlit UI for RAG Application

A comprehensive web interface for the RAG application with full validation
and AWS EC2 deployment support.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Tuple
import time
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with dark mode support
st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A9EFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Status boxes - dark mode friendly */
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.5);
        color: #81C784;
    }
    .status-error {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid rgba(244, 67, 54, 0.5);
        color: #EF5350;
    }
    .status-warning {
        background-color: rgba(255, 193, 7, 0.2);
        border: 1px solid rgba(255, 193, 7, 0.5);
        color: #FFD54F;
    }
    
    /* Chat messages - dark mode optimized */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }
    .user-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6e 100%);
        border-left: 4px solid #4A9EFF;
        margin-left: 20%;
        color: #e0e0e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .assistant-message {
        background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
        border-left: 4px solid #66BB6A;
        margin-right: 20%;
        color: #f0f0f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Dark mode adjustments for Streamlit components */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* Sidebar dark mode */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    /* Button styling for dark mode */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
    }
    
    /* File uploader dark mode */
    .uploadedFile {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #4A9EFF;
    }
    
    /* Expander dark mode */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db_status' not in st.session_state:
    st.session_state.vector_db_status = None
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False


def validate_api_key(api_key: str) -> bool:
    """Validate Perplexity API key format."""
    if not api_key:
        return False
    # Perplexity API keys typically start with 'pplx-'
    return api_key.startswith('pplx-') and len(api_key) > 20


def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate user query input."""
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query) > 1000:
        return False, "Query is too long. Please keep it under 1000 characters."
    
    if len(query) < 3:
        return False, "Query is too short. Please provide a meaningful question."
    
    # Check for potentially malicious content
    dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onload=']
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return False, "Query contains invalid characters"
    
    return True, None


def validate_pdf_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded PDF file."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (50MB limit)
    max_size = 50 * 1024 * 1024  # 50MB
    if uploaded_file.size > max_size:
        return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f} MB) exceeds 50MB limit"
    
    # Check file type
    if uploaded_file.type != 'application/pdf':
        return False, "File must be a PDF"
    
    # Check if file has content
    if uploaded_file.size == 0:
        return False, "File is empty"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other issues."""
    import re
    # Remove path components
    filename = Path(filename).name
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    return filename


def check_system_status() -> Dict[str, Any]:
    """Check system status and return status dictionary."""
    status = {
        'api_key_configured': False,
        'api_key_valid': False,
        'vector_db_exists': False,
        'vector_db_loaded': False,
        'system_ready': False,
        'errors': []
    }
    
    try:
        # Check API key
        from config import PERPLEXITY_API_KEY
        if PERPLEXITY_API_KEY:
            status['api_key_configured'] = True
            status['api_key_valid'] = validate_api_key(PERPLEXITY_API_KEY)
            if not status['api_key_valid']:
                status['errors'].append("API key format is invalid")
        else:
            status['errors'].append("API key not configured")
        
        # Check vector database
        vector_db_path = Path("vector_db")
        index_path = vector_db_path / "faiss_index.bin"
        metadata_path = vector_db_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            status['vector_db_exists'] = True
            if st.session_state.vector_db_status == 'loaded':
                status['vector_db_loaded'] = True
        
        # System is ready if API key is valid and vector DB exists
        status['system_ready'] = (
            status['api_key_valid'] and 
            status['vector_db_exists']
        )
        
    except Exception as e:
        status['errors'].append(f"Error checking system status: {str(e)}")
        logger.error(f"Error checking system status: {str(e)}")
    
    return status


def initialize_system() -> bool:
    """Initialize the RAG system."""
    try:
        with st.spinner("Initializing RAG system..."):
            from src.pdf_processor import PDFProcessor
            from src.text_chunker import TextChunker
            from src.vector_store import VectorStore
            from src.hybrid_llm_client import HybridLLMClient
            from src.conversation_manager import ConversationManager
            from config import PDF_URLS
            
            # Initialize components
            pdf_processor = PDFProcessor()
            text_chunker = TextChunker()
            vector_store = VectorStore()
            llm_client = HybridLLMClient()
            conversation_manager = ConversationManager(llm_client, vector_store)
            
            # Check if vector database exists
            if vector_store.load_index():
                st.session_state.vector_db_status = 'loaded'
                logger.info("Loaded existing vector database")
            else:
                st.session_state.vector_db_status = 'not_found'
                logger.warning("Vector database not found")
            
            # Store in session state
            st.session_state.conversation_manager = conversation_manager
            st.session_state.system_initialized = True
            
            return True
            
    except Exception as e:
        error_msg = f"Failed to initialize system: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        return False


def setup_sidebar():
    """Setup sidebar with configuration and status."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Configuration
        st.subheader("API Key")
        api_key_input = st.text_input(
            "Perplexity API Key",
            value=os.getenv("PERPLEXITY_API_KEY", ""),
            type="password",
            help="Enter your Perplexity API key (starts with 'pplx-')"
        )
        
        if api_key_input:
            if validate_api_key(api_key_input):
                st.success("✓ API key format valid")
                # Update environment variable
                os.environ["PERPLEXITY_API_KEY"] = api_key_input
                st.session_state.api_key_valid = True
            else:
                st.error("✗ Invalid API key format")
                st.session_state.api_key_valid = False
        else:
            st.warning("API key not set")
            st.session_state.api_key_valid = False
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        status = check_system_status()
        
        if status['api_key_valid']:
            st.success("✓ API Key: Valid")
        else:
            st.error("✗ API Key: Invalid or Missing")
        
        if status['vector_db_exists']:
            st.success("✓ Vector DB: Found")
        else:
            st.warning("⚠ Vector DB: Not Found")
        
        if status['system_ready']:
            st.success("✓ System: Ready")
        else:
            st.error("✗ System: Not Ready")
        
        st.divider()
        
        # System Actions
        st.subheader("System Actions")
        
        if st.button("🔄 Initialize System", use_container_width=True):
            if not status['api_key_valid']:
                st.error("Please configure a valid API key first")
            else:
                if initialize_system():
                    st.success("System initialized successfully!")
                    st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.conversation_manager:
                st.session_state.conversation_manager.clear_history()
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("📊 Check System Health", use_container_width=True):
            status = check_system_status()
            st.json(status)
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ Information")
        st.info("""
        **RAG Application**
        
        This application uses:
        - Perplexity API for LLM
        - Sentence-Transformers for embeddings
        - FAISS for vector search
        
        **Usage:**
        1. Configure API key
        2. Initialize system
        3. Start chatting!
        """)


def display_chat_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Display a chat message."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            with st.expander("View Details"):
                if 'context' in metadata:
                    st.text_area("Retrieved Context", metadata['context'], height=100)
                if 'conversation_turns' in metadata:
                    st.metric("Conversation Turns", metadata['conversation_turns'])


def main_chat_interface():
    """Main chat interface."""
    st.markdown('<h1 class="main-header">🤖 RAG Application</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about the research papers in the knowledge base")
    
    # Check if system is initialized
    if not st.session_state.system_initialized:
        status = check_system_status()
        if not status['api_key_valid']:
            st.error("⚠️ Please configure a valid API key in the sidebar first")
            return
        if not status['vector_db_exists']:
            st.warning("⚠️ Vector database not found. Please initialize the system in the sidebar.")
            return
        
        # Auto-initialize if conditions are met
        if status['api_key_valid'] and status['vector_db_exists']:
            if initialize_system():
                st.success("System initialized! You can now start chatting.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to initialize system. Please check the logs.")
                return
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(
            message['role'],
            message['content'],
            message.get('metadata')
        )
    
    # Chat input
    user_query = st.chat_input("Ask a question about the research papers...")
    
    if user_query:
        # Validate input
        is_valid, error_msg = validate_query(user_query)
        if not is_valid:
            st.error(f"Invalid query: {error_msg}")
            return
        
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query,
            'metadata': None
        })
        
        # Display user message
        display_chat_message('user', user_query)
        
        # Generate response
        if st.session_state.conversation_manager:
            try:
                # Check system status before generating
                status = check_system_status()
                if not status['api_key_valid']:
                    st.error("API key is not valid. Please configure a valid API key in the sidebar.")
                    return
                
                if not status['vector_db_loaded']:
                    st.warning("Vector database may not be loaded. Response quality may be affected.")
                
                with st.spinner("Thinking..."):
                    result = st.session_state.conversation_manager.generate_response(user_query)
                    
                    response = result.get('response', 'I apologize, but I could not generate a response.')
                    
                    # Validate response
                    if not response or len(response.strip()) == 0:
                        response = "I apologize, but I could not generate a valid response. Please try again."
                    
                    # Check for errors in result
                    if 'error' in result:
                        st.warning(f"Warning: {result['error']}")
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'metadata': {
                            'context': result.get('context', ''),
                            'conversation_turns': result.get('conversation_turns', 0),
                            'timestamp': time.time()
                        }
                    })
                    
                    # Display assistant message
                    display_chat_message('assistant', response, {
                        'context': result.get('context', ''),
                        'conversation_turns': result.get('conversation_turns', 0)
                    })
                    
            except KeyboardInterrupt:
                st.warning("Request cancelled by user")
                logger.info("Request cancelled by user")
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)
                
                # Add error message to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "I apologize, but I encountered an error. Please try again or check the system status.",
                    'metadata': {'error': str(e), 'timestamp': time.time()}
                })
        else:
            st.error("System not initialized. Please initialize in the sidebar.")


def pdf_management_page():
    """PDF management page."""
    st.header("📄 PDF Management")
    
    from src.pdf_processor import PDFProcessor
    from src.text_chunker import TextChunker
    from src.vector_store import VectorStore
    from config import PDF_URLS
    
    # Check existing PDFs
    data_dir = Path("data")
    existing_pdfs = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Existing PDFs")
        if existing_pdfs:
            for pdf in existing_pdfs:
                st.text(f"✓ {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
        else:
            st.info("No PDFs found in data directory")
    
    with col2:
        st.subheader("Upload New PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            # Validate file
            is_valid, error_msg = validate_pdf_file(uploaded_file)
            if not is_valid:
                st.error(f"Upload failed: {error_msg}")
            else:
                try:
                    # Sanitize filename
                    safe_filename = sanitize_filename(uploaded_file.name)
                    
                    # Save file
                    data_dir.mkdir(exist_ok=True)
                    file_path = data_dir / safe_filename
                    
                    # Check if file already exists
                    if file_path.exists():
                        st.warning(f"File {safe_filename} already exists. Overwriting...")
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✓ Uploaded {safe_filename} ({uploaded_file.size / 1024:.1f} KB)")
                    logger.info(f"Uploaded PDF: {safe_filename}")
                except Exception as e:
                    error_msg = f"Error saving file: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
    
    st.divider()
    
    # Process PDFs and create vector database
    st.subheader("Vector Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Default PDFs", use_container_width=True):
            try:
                with st.spinner("Downloading PDFs..."):
                    pdf_processor = PDFProcessor()
                    downloaded = pdf_processor.download_pdfs(PDF_URLS)
                    if downloaded:
                        st.success(f"✓ Downloaded {len(downloaded)} PDFs")
                    else:
                        st.warning("No PDFs were downloaded")
            except Exception as e:
                st.error(f"Error downloading PDFs: {str(e)}")
                logger.error(f"Error downloading PDFs: {str(e)}")
    
    with col2:
        if st.button("🔄 Rebuild Vector Database", use_container_width=True):
            status = check_system_status()
            if not status['api_key_valid']:
                st.error("Please configure a valid API key first")
            else:
                try:
                    with st.spinner("Rebuilding vector database (this may take a while)..."):
                        pdf_processor = PDFProcessor()
                        text_chunker = TextChunker()
                        vector_store = VectorStore()
                        
                        # Extract texts
                        texts = pdf_processor.extract_texts()
                        if not texts:
                            st.error("No texts extracted from PDFs")
                        else:
                            # Chunk texts
                            chunks = text_chunker.chunk_texts(texts)
                            if not chunks:
                                st.error("No chunks created from texts")
                            else:
                                # Create index
                                vector_store.create_index(chunks)
                                st.success("✓ Vector database rebuilt successfully!")
                                st.session_state.vector_db_status = 'loaded'
                                st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding vector database: {str(e)}")
                    logger.error(f"Error rebuilding vector database: {str(e)}")
                    logger.error(traceback.format_exc())


def main():
    """Main application."""
    # Setup sidebar
    setup_sidebar()
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Chat", "📄 PDF Management"])
    
    with tab1:
        main_chat_interface()
    
    with tab2:
        pdf_management_page()


if __name__ == "__main__":
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("vector_db").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    main()

