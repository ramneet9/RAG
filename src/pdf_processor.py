"""
PDF Processing Module

Handles downloading and text extraction from PDF documents.
"""

import os
import requests
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF downloading and text extraction."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_pdfs(self, urls: List[str]) -> List[str]:
        """
        Download PDFs from URLs.
        
        Args:
            urls: List of PDF URLs to download
            
        Returns:
            List of local file paths
        """
        downloaded_files = []
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"Downloading PDF {i+1}/{len(urls)}: {url}")
                
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                filename = f"paper_{i+1}.pdf"
                filepath = self.data_dir / filename
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_files.append(str(filepath))
                logger.info(f"Downloaded: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {str(e)}")
                
        return downloaded_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
                text += "\n"  # Add page break
                
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            return ""
    
    def extract_texts(self) -> List[Dict[str, str]]:
        """
        Extract text from all PDFs in the data directory.
        
        Returns:
            List of dictionaries containing filename and extracted text
        """
        texts = []
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            logger.info(f"Extracting text from: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            
            if text.strip():
                texts.append({
                    "filename": pdf_file.name,
                    "text": text.strip()
                })
                logger.info(f"Extracted {len(text)} characters from {pdf_file.name}")
            else:
                logger.warning(f"No text extracted from {pdf_file.name}")
                
        return texts
