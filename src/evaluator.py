"""
Evaluation Module

Handles evaluation of the RAG system using custom metrics and frameworks.
"""

import json
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path
from .conversation_manager import ConversationManager
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates RAG system performance using custom metrics."""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        self.evaluation_results = []
        self.report_generator = ReportGenerator()
        
    def evaluate_relevance(self, query: str, response: str, context: str) -> float:
        """
        Evaluate relevance of response to query.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        # Check if response contains query keywords
        query_coverage = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        
        # Check if response uses context information
        context_usage = len(context_words.intersection(response_words)) / len(response_words) if response_words else 0
        
        # Combined relevance score
        relevance_score = (query_coverage * 0.6) + (context_usage * 0.4)
        
        return min(relevance_score, 1.0)
    
    def evaluate_accuracy(self, response: str, context: str) -> float:
        """
        Evaluate accuracy of response based on context.
        
        Args:
            response: Generated response
            context: Retrieved context
            
        Returns:
            Accuracy score (0-1)
        """
        # Simple accuracy based on information consistency
        response_sentences = response.split('.')
        context_sentences = context.split('.')
        
        accurate_sentences = 0
        total_sentences = len(response_sentences)
        
        for resp_sentence in response_sentences:
            if resp_sentence.strip():
                # Check if sentence has supporting evidence in context
                resp_words = set(resp_sentence.lower().split())
                for ctx_sentence in context_sentences:
                    ctx_words = set(ctx_sentence.lower().split())
                    # If there's significant word overlap, consider it supported
                    if len(resp_words.intersection(ctx_words)) > 2:
                        accurate_sentences += 1
                        break
        
        return accurate_sentences / total_sentences if total_sentences > 0 else 0
    
    def evaluate_contextual_awareness(self, conversation_history: List[Dict[str, str]]) -> float:
        """
        Evaluate contextual awareness based on conversation history.
        
        Args:
            conversation_history: Previous conversation turns
            
        Returns:
            Contextual awareness score (0-1)
        """
        if len(conversation_history) < 2:
            return 1.0  # No previous context to consider
        
        # Check if current response references previous topics
        current_response = conversation_history[-1]["response"]
        previous_queries = [turn["query"] for turn in conversation_history[:-1]]
        
        reference_score = 0
        for prev_query in previous_queries:
            prev_words = set(prev_query.lower().split())
            current_words = set(current_response.lower().split())
            
            # Check for word overlap indicating context awareness
            overlap = len(prev_words.intersection(current_words))
            if overlap > 0:
                reference_score += overlap / len(prev_words)
        
        return min(reference_score / len(previous_queries), 1.0)
    
    def evaluate_response_quality(self, response: str) -> float:
        """
        Evaluate quality of response.
        
        Args:
            response: Generated response
            
        Returns:
            Quality score (0-1)
        """
        # Check response length (not too short, not too long)
        word_count = len(response.split())
        length_score = 1.0 if 10 <= word_count <= 200 else 0.5
        
        # Check for proper sentence structure
        sentences = response.split('.')
        structure_score = 1.0 if len(sentences) > 1 else 0.7
        
        # Check for clarity indicators
        clarity_indicators = ['because', 'therefore', 'however', 'specifically', 'for example']
        clarity_score = 1.0 if any(indicator in response.lower() for indicator in clarity_indicators) else 0.8
        
        # Combined quality score
        quality_score = (length_score * 0.4) + (structure_score * 0.3) + (clarity_score * 0.3)
        
        return quality_score
    
    def evaluate_single_question(self, question: str) -> Dict[str, Any]:
        """
        Evaluate a single question.
        
        Args:
            question: Question to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating question: {question}")
        
        # Generate response
        result = self.conversation_manager.generate_response(question)
        
        # Calculate metrics
        relevance = self.evaluate_relevance(question, result["response"], result["context"])
        accuracy = self.evaluate_accuracy(result["response"], result["context"])
        contextual_awareness = self.evaluate_contextual_awareness(self.conversation_manager.get_conversation_history())
        response_quality = self.evaluate_response_quality(result["response"])
        
        # Overall score
        overall_score = (relevance * 0.3) + (accuracy * 0.3) + (contextual_awareness * 0.2) + (response_quality * 0.2)
        
        evaluation_result = {
            "question": question,
            "response": result["response"],
            "context": result["context"],
            "metrics": {
                "relevance": relevance,
                "accuracy": accuracy,
                "contextual_awareness": contextual_awareness,
                "response_quality": response_quality,
                "overall_score": overall_score
            },
            "conversation_turns": result["conversation_turns"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_results.append(evaluation_result)
        return evaluation_result
    
    def evaluate(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple questions.
        
        Args:
            questions: List of questions to evaluate
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Starting evaluation of {len(questions)} questions")
        
        # Clear conversation history for fresh start
        self.conversation_manager.clear_history()
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}")
            result = self.evaluate_single_question(question)
            results.append(result)
        
        logger.info("Evaluation completed")
        return results
    
    def calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall evaluation metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of overall metrics
        """
        if not results:
            return {}
        
        metrics = ["relevance", "accuracy", "contextual_awareness", "response_quality", "overall_score"]
        overall_metrics = {}
        
        for metric in metrics:
            scores = [result["metrics"][metric] for result in results]
            overall_metrics[f"avg_{metric}"] = sum(scores) / len(scores)
            overall_metrics[f"min_{metric}"] = min(scores)
            overall_metrics[f"max_{metric}"] = max(scores)
        
        return overall_metrics
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate evaluation report (both JSON and PDF).
        
        Args:
            results: List of evaluation results
            
        Returns:
            Path to generated PDF report
        """
        logger.info("Generating evaluation report...")
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(results)
        
        # Create report data
        report_data = {
            "evaluation_summary": {
                "total_questions": len(results),
                "evaluation_date": datetime.now().isoformat(),
                "overall_metrics": overall_metrics
            },
            "detailed_results": results
        }
        
        # Save JSON report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        json_report_path = reports_dir / f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to: {json_report_path}")
        
        # Generate PDF report
        pdf_report_path = self.report_generator.generate_report(report_data)
        
        logger.info(f"PDF report saved to: {pdf_report_path}")
        return pdf_report_path
