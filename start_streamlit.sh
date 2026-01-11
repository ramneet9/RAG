#!/bin/bash
# Start Streamlit application on AWS EC2

# Activate virtual environment
source venv/bin/activate

# Use server config for deployment
if [ -f ".streamlit/config.server.toml" ]; then
    cp .streamlit/config.server.toml .streamlit/config.toml
fi

# Start Streamlit
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true

