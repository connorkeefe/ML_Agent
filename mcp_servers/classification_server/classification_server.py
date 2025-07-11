"""
Problem Classification MCP Server
Handles ML problem type identification and capability validation.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from mcp.types import Tool, TextContent
from pydantic import BaseModel

from ..base_server import BaseMLServer, ServerConfig
from ...shared.schemas import (
    MLProblemType, 
    ProblemClassification,
    DataType
)


class ClassificationServer(BaseMLServer):
    """MCP Server for ML problem classification"""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.problem_keywords = {}
        self.capability_matrix = {}
    
    async def initialize(self) -> None:
        """Initialize the classification server"""
        self.logger.info("Initializing Classification Server")
        
        # Load problem type keywords and patterns
        self._load_classification_patterns()
    
    def _load_classification_patterns(self) -> None:
        """Load patterns for problem classification"""
        self.problem_keywords = {
            MLProblemType.SUPERVISED_CLASSIFICATION: {
                "keywords": ["classify", "classification", "predict category", "identify class", 
                           "binary classification", "multi-class", "sentiment analysis", 
                           "image recognition", "spam detection"],
                "indicators": ["categories", "classes", "labels", "discrete target"],
                "data_hints": ["categorical target", "discrete labels"]
            },
            MLProblemType.SUPERVISED_REGRESSION: {
                "keywords": ["predict", "regression", "forecast", "estimate value", 
                           "continuous prediction", "price prediction", "sales forecast"],
                "indicators": ["continuous", "numerical target", "price", "value", "amount"],
                "data_hints": ["numerical target", "continuous values"]
            },
            MLProblemType.UNSUPERVISED_CLUSTERING: {
                "keywords": ["cluster", "group", "segment", "find patterns", 
                           "customer segmentation", "market segmentation"],
                "indicators": ["groups", "clusters", "segments", "similar items"],
                "data_hints": ["no target variable", "unlabeled data"]
            },
            MLProblemType.UNSUPERVISED_ANOMALY_DETECTION: {
                "keywords": ["anomaly", "outlier", "fraud detection", "abnormal", 
                           "unusual patterns", "detect anomalies"],
                "indicators": ["outliers", "anomalies", "fraud", "unusual"],
                "data_hints": ["detect unusual patterns", "identify outliers"]
            },
            MLProblemType.TIME_SERIES_FORECASTING: {
                "keywords": ["time series", "forecast", "predict future", "temporal", 
                           "trend analysis", "seasonal patterns"],
                "indicators": ["time", "date", "temporal", "sequential", "trend"],
                "data_hints": ["datetime column", "temporal data", "time-ordered"]
            }
        }
    
    def get_tool_definitions(self) -> List[Tool]:
        """Return classification server tools"""
        return [
            Tool(
                name="classify_problem",
                description="Analyze user prompt and data to classify ML problem type",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_prompt": {
                            "type": "string",
                            "description": "User's description of their ML problem"
                        },
                        "data_preview": {
                            "type": "object",
                            "description": "Preview of the data structure",
                            "properties": {
                                "columns": {"type": "array", "items": {"type": "string"}},
                                "dtypes": {"type": "object"},
                                "sample_rows": {"type": "array"},
                                "shape": {"type": "array", "items": {"type": "integer"}}
                            }
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context from conversation",
                            "default": {}
                        }
                    },
                    "required": ["user_prompt"]
                }
            ),
            Tool(
                name="request_clarification",
                description="Generate follow-up questions for unclear problems",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "partial_classification": {
                            "type": "object",
                            "description": "Partial classification results"
                        },
                        "missing_info": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of missing information needed"
                        }
                    },
                    "required": ["missing_info"]
                }
            ),
            Tool(
                name="validate_capability",
                description="Check if problem type is supported by the system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_type": {
                            "type": "string",
                            "description": "Identified ML problem type"
                        },
                        "data_characteristics": {
                            "type": "object",
                            "description": "Characteristics of the dataset"
                        }
                    },
                    "required": ["problem_type"]
                }
            )
        ]
    
    async def handle_classify_problem(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle problem classification"""
        user_prompt = parameters["user_prompt"]
        data_preview = parameters.get("data_preview", {})
        context = parameters.get("context", {})
        
        self.logger.info("Classifying ML problem", prompt_length=len(user_prompt))
        
        # Analyze text patterns
        text_scores = self._analyze_text_patterns(user_prompt)
        
        # Analyze data patterns if available
        data_scores = {}
        if data_preview:
            data_scores = self._analyze_data_patterns(data_preview)
        
        # Combine scores
        combined_scores = self._combine_scores(text_scores, data_scores)
        
        # Determine best classification
        classification = self._determine_classification(combined_scores, user_prompt, data_preview)
        
        return {
            "classification": classification.dict(),
            "confidence_scores": combined_scores,
            "reasoning_details": {
                "text_analysis": text_scores,
                "data_analysis": data_scores
            }
        }
    
    async def handle_request_clarification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clarification questions"""
        missing_info = parameters["missing_info"]
        partial_classification = parameters.get("partial_classification", {})
        
        questions = []
        
        if "target_variable" in missing_info:
            questions.extend([
                "What are you trying to predict or find in your data?",
                "Do you have a specific outcome or target variable in mind?",
                "Are you looking to predict categories/classes or numerical values?"
            ])
        
        if "problem_goal" in missing_info:
            questions.extend([
                "What business problem are you trying to solve?",
                "What would success look like for this analysis?",
                "How will you use the results of this analysis?"
            ])
        
        if "data_description" in missing_info:
            questions.extend([
                "Can you describe what each column in your data represents?",
                "What type of data do you have (numerical, categorical, text, etc.)?",
                "Do you have labeled data or is this exploratory analysis?"
            ])
        
        if "time_component" in missing_info:
            questions.extend([
                "Does your data have a time component?",
                "Are you looking to predict future values?",
                "Is the order of your data points important?"
            ])
        
        return {
            "clarification_questions": questions[:3],  # Limit to top 3
            "suggested_info_needed": missing_info,
            "next_steps": "Please provide answers to help classify your problem accurately"
        }
    
    async def handle_validate_capability(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system capability for problem type"""
        problem_type = parameters["problem_type"]
        data_characteristics = parameters.get("data_characteristics", {})
        
        # Get capability matrix from resources
        capabilities = self.get_resource_content("capability_matrix")
        
        is_supported = problem_type in capabilities.get("supported_problems", [])
        limitations = capabilities.get("limitations", {}).get(problem_type, [])
        requirements = capabilities.get("requirements", {}).get(problem_type, [])
        
        # Check data requirements
        data_issues = []
        if data_characteristics:
            data_issues = self._check_data_requirements(problem_type, data_characteristics)
        
        return {
            "is_supported": is_supported,
            "problem_type": problem_type,
            "limitations": limitations,
            "requirements": requirements,
            "data_issues": data_issues,
            "recommendation": "proceed" if is_supported and not data_issues else "clarify_requirements"
        }
    
    def _analyze_text_patterns(self, text: str) -> Dict[str, float]:
        """Analyze text for problem type indicators"""
        text_lower = text.lower()
        scores = {}
        
        for problem_type, patterns in self.problem_keywords.items():
            score = 0.0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    score += 1.0
            
            # Check indicators
            for indicator in patterns["indicators"]:
                if indicator in text_lower:
                    score += 0.5
            
            # Normalize by total possible score
            max_score = len(patterns["keywords"]) + len(patterns["indicators"]) * 0.5
            scores[problem_type] = score / max_score if max_score > 0 else 0.0
        
        return scores
    
    def _analyze_data_patterns(self, data_preview: Dict[str, Any]) -> Dict[str, float]:
        """Analyze data structure for problem type hints"""
        scores = {}
        
        columns = data_preview.get("columns", [])
        dtypes = data_preview.get("dtypes", {})
        shape = data_preview.get("shape", [0, 0])
        
        # Look for target variable patterns
        target_indicators = ["target", "label", "class", "y", "output", "prediction"]
        has_likely_target = any(
            any(indicator in col.lower() for indicator in target_indicators)
            for col in columns
        )
        
        # Look for datetime columns
        datetime_cols = [col for col, dtype in dtypes.items() 
                        if "datetime" in str(dtype) or "date" in col.lower()]
        
        # Classification hints
        if has_likely_target:
            scores[MLProblemType.SUPERVISED_CLASSIFICATION] = 0.7
            scores[MLProblemType.SUPERVISED_REGRESSION] = 0.7
        
        # Time series hints
        if datetime_cols:
            scores[MLProblemType.TIME_SERIES_FORECASTING] = 0.8
        
        # Unsupervised hints (no clear target)
        if not has_likely_target:
            scores[MLProblemType.UNSUPERVISED_CLUSTERING] = 0.6
            scores[MLProblemType.UNSUPERVISED_ANOMALY_DETECTION] = 0.4
        
        return scores
    
    def _combine_scores(self, text_scores: Dict[str, float], data_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine text and data analysis scores"""
        all_problem_types = set(text_scores.keys()) | set(data_scores.keys())
        combined = {}
        
        for problem_type in all_problem_types:
            text_score = text_scores.get(problem_type, 0.0)
            data_score = data_scores.get(problem_type, 0.0)
            
            # Weight text analysis higher if no data preview
            if not data_scores:
                combined[problem_type] = text_score
            else:
                # Combine with weighted average
                combined[problem_type] = 0.6 * text_score + 0.4 * data_score
        
        return combined
    
    def _determine_classification(self, scores: Dict[str, float], prompt: str, data_preview: Dict) -> ProblemClassification:
        """Determine final classification from scores"""
        
        if not scores or max(scores.values()) == 0:
            return ProblemClassification(
                problem_type=MLProblemType.SUPERVISED_CLASSIFICATION,  # Default
                confidence=0.1,
                reasoning="Unable to determine problem type from provided information",
                needs_clarification=True,
                required_clarifications=["problem_goal", "target_variable", "data_description"]
            )
        
        # Get best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # Check if confidence is too low
        if confidence < 0.3:
            return ProblemClassification(
                problem_type=best_type,
                confidence=confidence,
                reasoning=f"Low confidence classification. Best match: {best_type}",
                needs_clarification=True,
                required_clarifications=self._identify_missing_info(prompt, data_preview)
            )
        
        # Generate reasoning
        reasoning = f"Classified as {best_type} based on "
        if "predict" in prompt.lower() or "forecast" in prompt.lower():
            reasoning += "prediction/forecasting keywords. "
        if data_preview and "target" in str(data_preview).lower():
            reasoning += "Apparent target variable in data. "
        
        return ProblemClassification(
            problem_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            needs_clarification=confidence < 0.7,
            required_clarifications=self._identify_missing_info(prompt, data_preview) if confidence < 0.7 else []
        )
    
    def _identify_missing_info(self, prompt: str, data_preview: Dict) -> List[str]:
        """Identify what information is missing for better classification"""
        missing = []
        
        # Check for target variable mention
        target_keywords = ["predict", "target", "outcome", "label", "class"]
        if not any(keyword in prompt.lower() for keyword in target_keywords):
            missing.append("target_variable")
        
        # Check for goal/purpose
        goal_keywords = ["goal", "want", "need", "trying", "business", "problem"]
        if not any(keyword in prompt.lower() for keyword in goal_keywords):
            missing.append("problem_goal")
        
        # Check for data description
        if not data_preview:
            missing.append("data_description")
        
        return missing
    
    def _check_data_requirements(self, problem_type: str, data_characteristics: Dict) -> List[str]:
        """Check if data meets requirements for problem type"""
        issues = []
        
        row_count = data_characteristics.get("row_count", 0)
        col_count = data_characteristics.get("col_count", 0)
        
        # Minimum data requirements
        if row_count < 100:
            issues.append("Dataset may be too small (< 100 rows)")
        
        if col_count < 2:
            issues.append("Dataset needs more features (< 2 columns)")
        
        # Problem-specific requirements
        if "supervised" in problem_type and not data_characteristics.get("has_target"):
            issues.append("Supervised learning requires a target variable")
        
        if "time_series" in problem_type and not data_characteristics.get("has_datetime"):
            issues.append("Time series analysis requires datetime information")
        
        return issues