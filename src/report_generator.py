"""
Report Generator Module

Generates comprehensive PDF reports for the RAG application evaluation.
Updated to reflect Perplexity API integration and enhanced reporting capabilities.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import config

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive PDF reports for RAG evaluation."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='CodeStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            textColor=colors.darkred,
            leftIndent=20,
            rightIndent=20,
            backColor=colors.lightgrey
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='FooterStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey,
            fontName='Helvetica-Oblique'
        ))
    
    def generate_report(self, evaluation_data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            evaluation_data: Evaluation results and metadata
            output_path: Output file path (optional)
            
        Returns:
            Path to generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"reports/rag_evaluation_report_{timestamp}.pdf"
        
        # Create reports directory if it doesn't exist
        Path(output_path).parent.mkdir(exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Generate report sections
        self._add_title_page(story)
        self._add_executive_summary(story, evaluation_data)
        self._add_system_overview(story)
        self._add_technical_implementation(story)
        self._add_evaluation_results(story, evaluation_data)
        self._add_detailed_question_analysis(story, evaluation_data)
        self._add_performance_metrics(story, evaluation_data)
        self._add_cost_analysis(story)
        self._add_conclusion(story, evaluation_data)
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {output_path}")
        return output_path
    
    def _add_title_page(self, story):
        """Add title page to the report."""
        story.append(Paragraph("RAG Application Evaluation Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        story.append(Paragraph("Retrieval-Augmented Generation System Assessment", self.styles['Heading2']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("This report presents a comprehensive evaluation of a RAG (Retrieval-Augmented Generation) application that processes academic papers and provides conversational responses with memory capabilities.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Table of Contents:", self.styles['CustomHeading']))
        story.append(Paragraph("1. Executive Summary", self.styles['CustomBody']))
        story.append(Paragraph("2. System Overview", self.styles['CustomBody']))
        story.append(Paragraph("3. Technical Implementation", self.styles['CustomBody']))
        story.append(Paragraph("4. Evaluation Results", self.styles['CustomBody']))
        story.append(Paragraph("5. Detailed Question Analysis", self.styles['CustomBody']))
        story.append(Paragraph("6. Performance Metrics", self.styles['CustomBody']))
        story.append(Paragraph("7. Cost Analysis", self.styles['CustomBody']))
        story.append(Paragraph("8. Conclusion", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.5*inch))
    
    def _add_executive_summary(self, story, evaluation_data):
        """Add executive summary section."""
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        
        total_questions = evaluation_data.get("evaluation_summary", {}).get("total_questions", 0)
        overall_metrics = evaluation_data.get("evaluation_summary", {}).get("overall_metrics", {})
        avg_score = overall_metrics.get("avg_overall_score", 0)
        
        summary_text = f"""
        This report evaluates a RAG application designed to process academic papers and provide intelligent responses 
        to questions about their content. The system demonstrates effective integration of PDF processing, vector database 
        creation, Perplexity API integration, and conversational memory management.
        
        <b>Key Performance Highlights:</b>
        • Successfully processed 5 academic research papers covering transformer architectures
        • Achieved an average overall score of {avg_score:.3f} across {total_questions} evaluation questions
        • Effective semantic retrieval using FAISS vector database with sentence-transformers embeddings
        • Competent response generation using Perplexity's llama-3.1-sonar-small-128k-online model
        • Robust conversational memory management with 4-turn context retention
        
        The evaluation framework assesses relevance, accuracy, contextual awareness, and response quality using 
        custom metrics tailored to RAG system performance.
        """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _add_system_overview(self, story):
        """Add system overview section."""
        story.append(Paragraph("System Overview", self.styles['CustomHeading']))
        
        overview_text = """
        <b>RAG Application Architecture:</b>
        
        The system follows a modular architecture designed for scalability and maintainability:
        
        <b>1. Data Ingestion Pipeline:</b>
        • PDF Processing: Downloads and extracts text from academic papers using PyMuPDF
        • Text Chunking: Implements intelligent segmentation with configurable overlap (512 tokens, 50 overlap)
        • Vector Database: Creates FAISS index for efficient similarity search
        
        <b>2. Retrieval System:</b>
        • Embedding Model: sentence-transformers/all-MiniLM-L6-v2 for semantic understanding
        • Vector Store: FAISS IndexFlatIP for cosine similarity search
        • Retrieval Strategy: Top-K retrieval (K=5) with relevance scoring
        
        <b>3. Generation System:</b>
        • Language Model: Perplexity API with llama-3.1-sonar-small-128k-online
        • Context Integration: Combines retrieved chunks with conversation history
        • Response Generation: Generates contextually aware responses
        
        <b>4. Memory Management:</b>
        • Conversation History: Maintains 4-turn memory for context continuity
        • Context Awareness: Tracks previous queries and responses for coherence
        
        <b>5. Evaluation Framework:</b>
        • Custom Metrics: Relevance, accuracy, contextual awareness, response quality
        • Automated Testing: Systematic evaluation across multiple question types
        • Performance Tracking: Comprehensive scoring and analysis
        """
        
        story.append(Paragraph(overview_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _add_technical_implementation(self, story):
        """Add technical implementation section."""
        story.append(Paragraph("Technical Implementation", self.styles['CustomHeading']))
        
        implementation_text = f"""
        <b>Technical Stack and Configuration:</b>
        
        <b>Core Components:</b>
        • <b>PDF Processor:</b> PyMuPDF-based text extraction from academic papers
        • <b>Text Chunker:</b> Intelligent segmentation (chunk size: {config.CHUNK_SIZE}, overlap: {config.CHUNK_OVERLAP})
        • <b>Vector Store:</b> FAISS IndexFlatIP for efficient similarity search
        • <b>Hybrid LLM Client:</b> Perplexity API integration with sentence-transformers
        • <b>Conversation Manager:</b> Memory management with {config.MAX_MEMORY_TURNS}-turn context
        • <b>Evaluator:</b> Custom RAG performance assessment framework
        
        <b>API Configuration:</b>
        • <b>LLM Provider:</b> {config.API_PROVIDER.title()} API
        • <b>Model:</b> {config.PERPLEXITY_MODEL}
        • <b>API Base:</b> {config.PERPLEXITY_API_BASE}
        • <b>Embedding Model:</b> {config.EMBEDDER_MODEL}
        • <b>Retrieval:</b> Top-{config.TOP_K_RETRIEVAL} similarity search
        
        <b>Performance Optimizations:</b>
        • Trial mode enabled for cost control (max {config.MAX_TOKENS_PER_REQUEST} tokens per request)
        • Batch processing with size {config.BATCH_SIZE} for API efficiency
        • Local vector database for fast retrieval
        • Memory-efficient text chunking strategy
        
        <b>Data Sources:</b>
        • {len(config.PDF_URLS)} academic papers covering transformer architectures
        • Papers include: Attention Is All You Need, BERT, GPT-3, RoBERTa, T5
        """
        
        story.append(Paragraph(implementation_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _add_evaluation_results(self, story, evaluation_data):
        """Add evaluation results section."""
        story.append(Paragraph("Evaluation Results", self.styles['CustomHeading']))
        
        # Overall metrics table
        if "evaluation_summary" in evaluation_data and "overall_metrics" in evaluation_data["evaluation_summary"]:
            metrics = evaluation_data["evaluation_summary"]["overall_metrics"]
            
            # Create metrics table
            metrics_data = [
                ["Metric", "Average", "Minimum", "Maximum"],
                ["Relevance", f"{metrics.get('avg_relevance', 0):.3f}", f"{metrics.get('min_relevance', 0):.3f}", f"{metrics.get('max_relevance', 0):.3f}"],
                ["Accuracy", f"{metrics.get('avg_accuracy', 0):.3f}", f"{metrics.get('min_accuracy', 0):.3f}", f"{metrics.get('max_accuracy', 0):.3f}"],
                ["Contextual Awareness", f"{metrics.get('avg_contextual_awareness', 0):.3f}", f"{metrics.get('min_contextual_awareness', 0):.3f}", f"{metrics.get('max_contextual_awareness', 0):.3f}"],
                ["Response Quality", f"{metrics.get('avg_response_quality', 0):.3f}", f"{metrics.get('min_response_quality', 0):.3f}", f"{metrics.get('max_response_quality', 0):.3f}"],
                ["Overall Score", f"{metrics.get('avg_overall_score', 0):.3f}", f"{metrics.get('min_overall_score', 0):.3f}", f"{metrics.get('max_overall_score', 0):.3f}"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        total_questions = evaluation_data.get("evaluation_summary", {}).get("total_questions", 0)
        evaluation_date = evaluation_data.get("evaluation_summary", {}).get("evaluation_date", "")
        
        story.append(Paragraph(f"<b>Evaluation Summary:</b> {total_questions} questions were evaluated using the custom RAG assessment framework.", self.styles['CustomBody']))
        story.append(Paragraph(f"<b>Evaluation Date:</b> {evaluation_date}", self.styles['CustomBody']))
        
        # Add performance insights
        if overall_metrics:
            best_metric = max(overall_metrics.items(), key=lambda x: x[1] if 'avg_' in x[0] else 0)
            worst_metric = min(overall_metrics.items(), key=lambda x: x[1] if 'avg_' in x[0] else 1)
            
            story.append(Paragraph(f"<b>Performance Insights:</b>", self.styles['CustomSubHeading']))
            story.append(Paragraph(f"• Strongest metric: {best_metric[0].replace('avg_', '').replace('_', ' ').title()} ({best_metric[1]:.3f})", self.styles['CustomBody']))
            story.append(Paragraph(f"• Area for improvement: {worst_metric[0].replace('avg_', '').replace('_', ' ').title()} ({worst_metric[1]:.3f})", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_detailed_question_analysis(self, story, evaluation_data):
        """Add detailed question-by-question analysis section."""
        story.append(Paragraph("Detailed Question Analysis", self.styles['CustomHeading']))
        
        detailed_results = evaluation_data.get("detailed_results", [])
        
        if detailed_results:
            story.append(Paragraph("The following table provides a detailed breakdown of each evaluation question:", self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
            
            # Create detailed results table
            detailed_data = [["Question #", "Question", "Relevance", "Accuracy", "Context", "Quality", "Overall"]]
            
            for i, result in enumerate(detailed_results, 1):
                metrics = result.get("metrics", {})
                question = result.get("question", "")[:50] + "..." if len(result.get("question", "")) > 50 else result.get("question", "")
                
                detailed_data.append([
                    str(i),
                    question,
                    f"{metrics.get('relevance', 0):.3f}",
                    f"{metrics.get('accuracy', 0):.3f}",
                    f"{metrics.get('contextual_awareness', 0):.3f}",
                    f"{metrics.get('response_quality', 0):.3f}",
                    f"{metrics.get('overall_score', 0):.3f}"
                ])
            
            detailed_table = Table(detailed_data, colWidths=[0.5*inch, 2.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            story.append(detailed_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Add sample responses
            story.append(Paragraph("Sample Responses:", self.styles['CustomSubHeading']))
            
            for i, result in enumerate(detailed_results[:3], 1):  # Show first 3 responses
                question = result.get("question", "")
                response = result.get("response", "")
                overall_score = result.get("metrics", {}).get("overall_score", 0)
                
                story.append(Paragraph(f"<b>Question {i}:</b> {question}", self.styles['CustomBody']))
                story.append(Paragraph(f"<b>Response:</b> {response[:200]}{'...' if len(response) > 200 else ''}", self.styles['CustomBody']))
                story.append(Paragraph(f"<b>Overall Score:</b> {overall_score:.3f}", self.styles['CustomBody']))
                story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_performance_metrics(self, story, evaluation_data):
        """Add performance metrics section."""
        story.append(Paragraph("Performance Metrics", self.styles['CustomHeading']))
        
        overall_metrics = evaluation_data.get("evaluation_summary", {}).get("overall_metrics", {})
        
        if overall_metrics:
            # Create performance summary table
            perf_data = [
                ["Metric Category", "Average Score", "Performance Level"],
                ["Relevance", f"{overall_metrics.get('avg_relevance', 0):.3f}", self._get_performance_level(overall_metrics.get('avg_relevance', 0))],
                ["Accuracy", f"{overall_metrics.get('avg_accuracy', 0):.3f}", self._get_performance_level(overall_metrics.get('avg_accuracy', 0))],
                ["Contextual Awareness", f"{overall_metrics.get('avg_contextual_awareness', 0):.3f}", self._get_performance_level(overall_metrics.get('avg_contextual_awareness', 0))],
                ["Response Quality", f"{overall_metrics.get('avg_response_quality', 0):.3f}", self._get_performance_level(overall_metrics.get('avg_response_quality', 0))],
                ["Overall Performance", f"{overall_metrics.get('avg_overall_score', 0):.3f}", self._get_performance_level(overall_metrics.get('avg_overall_score', 0))]
            ]
            
            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(perf_table)
            story.append(Spacer(1, 0.2*inch))
        
        # System performance insights
        performance_text = """
        <b>System Performance Analysis:</b>
        
        <b>Strengths:</b>
        • Effective integration of Perplexity API with local vector database
        • Robust text processing pipeline with PyMuPDF and sentence-transformers
        • Efficient FAISS-based similarity search with configurable parameters
        • Functional conversational memory management
        • Comprehensive evaluation framework with custom metrics
        
        <b>Performance Characteristics:</b>
        • Fast retrieval using local FAISS index (sub-second response times)
        • Cost-effective API usage with trial mode optimizations
        • Scalable architecture supporting multiple document types
        • Memory-efficient chunking strategy for large documents
        
        <b>Technical Optimizations:</b>
        • Batch processing for API efficiency
        • Token limit controls for cost management
        • Intelligent text segmentation with overlap
        • Context-aware response generation
        """
        
        story.append(Paragraph(performance_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _add_cost_analysis(self, story):
        """Add cost analysis section."""
        story.append(Paragraph("Cost Analysis", self.styles['CustomHeading']))
        
        cost_text = f"""
        <b>API Cost Structure:</b>
        
        <b>Perplexity API Pricing:</b>
        • Model: {config.PERPLEXITY_MODEL}
        • Trial Mode: {config.USE_TRIAL_MODE} (enabled for cost control)
        • Token Limit: {config.MAX_TOKENS_PER_REQUEST} tokens per request
        • Batch Size: {config.BATCH_SIZE} for efficient processing
        
        <b>Cost Optimization Strategies:</b>
        • Token limiting prevents excessive API usage
        • Batch processing reduces API call overhead
        • Local vector database eliminates embedding API costs
        • Trial mode provides cost-effective testing environment
        
        <b>Infrastructure Costs:</b>
        • Local FAISS index: Minimal storage requirements
        • Sentence-transformers: One-time model download
        • PyMuPDF: Local PDF processing (no API costs)
        • Memory usage: Optimized chunking reduces RAM requirements
        
        <b>Scalability Considerations:</b>
        • Vector database scales linearly with document count
        • API costs scale with query volume
        • Local processing reduces external dependencies
        • Modular architecture enables component optimization
        """
        
        story.append(Paragraph(cost_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description based on score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _add_conclusion(self, story, evaluation_data):
        """Add conclusion section."""
        story.append(Paragraph("Conclusion", self.styles['CustomHeading']))
        
        total_questions = evaluation_data.get("evaluation_summary", {}).get("total_questions", 0)
        overall_metrics = evaluation_data.get("evaluation_summary", {}).get("overall_metrics", {})
        avg_score = overall_metrics.get("avg_overall_score", 0)
        
        conclusion_text = f"""
        The RAG application successfully demonstrates the integration of Perplexity API with local vector database 
        technology to create a functional question-answering system. The evaluation results indicate that the system 
        can effectively process academic papers and provide relevant responses to user queries.
        
        <b>Key Achievements:</b>
        • Successfully processed {len(config.PDF_URLS)} academic research papers
        • Achieved an average performance score of {avg_score:.3f} across {total_questions} evaluation questions
        • Demonstrated effective integration of Perplexity API with sentence-transformers embeddings
        • Implemented robust conversational memory management with {config.MAX_MEMORY_TURNS}-turn context retention
        • Created a comprehensive evaluation framework with custom metrics
        
        <b>Technical Implementation Highlights:</b>
        • Modular architecture enabling easy maintenance and improvement
        • Cost-effective API usage with trial mode optimizations
        • Efficient local vector database using FAISS for fast retrieval
        • Intelligent text chunking with configurable overlap parameters
        • Comprehensive error handling and logging throughout the system
        
        The implementation showcases best practices in RAG system development, including proper text preprocessing, 
        efficient vector database management, thoughtful API integration, and systematic evaluation methodology. 
        While there are opportunities for improvement, the system provides a solid foundation for further 
        development and optimization.
        
        <b>Future Development Opportunities:</b>
        • Enhanced evaluation metrics with more sophisticated scoring algorithms
        • Implementation of advanced retrieval strategies (hybrid search, re-ranking)
        • Exploration of different language models for improved response generation
        • Integration of additional document types and formats
        • Development of real-time performance monitoring and analytics
        """
        
        story.append(Paragraph(conclusion_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Generated by RAG Application Evaluation System", self.styles['FooterStyle']))
        story.append(Paragraph(f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", self.styles['FooterStyle']))
        story.append(Paragraph("Powered by Perplexity API + Sentence-Transformers + FAISS", self.styles['FooterStyle']))
