"""
ML Decision MCP Server
Handles feature selection and model recommendations.
"""

from typing import Dict, List, Any, Optional
from mcp.types import Tool
from ..base_server import BaseMLServer, ServerConfig


class DecisionServer(BaseMLServer):
    """MCP Server for ML decision making"""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config)
    
    async def initialize(self) -> None:
        """Initialize the decision server"""
        self.logger.info("Initializing ML Decision Server")
    
    def get_tool_definitions(self) -> List[Tool]:
        """Return decision server tools"""
        return [
            Tool(
                name="recommend_features",
                description="Recommend features for model training",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_schema": {
                            "type": "object",
                            "description": "Analysis of the data schema"
                        },
                        "problem_type": {
                            "type": "string",
                            "description": "The ML problem type"
                        },
                        "quality_report": {
                            "type": "object",
                            "description": "Data quality assessment"
                        }
                    },
                    "required": ["data_schema", "problem_type"]
                }
            ),
            Tool(
                name="recommend_models",
                description="Recommend suitable models for the problem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_type": {
                            "type": "string",
                            "description": "The ML problem type"
                        },
                        "data_characteristics": {
                            "type": "object",
                            "description": "Characteristics of the dataset"
                        },
                        "target_info": {
                            "type": "object",
                            "description": "Information about the target variable"
                        }
                    },
                    "required": ["problem_type"]
                }
            ),
            Tool(
                name="suggest_feature_engineering",
                description="Suggest feature engineering transformations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_schema": {
                            "type": "object",
                            "description": "Analysis of the data schema"
                        },
                        "problem_type": {
                            "type": "string",
                            "description": "The ML problem type"
                        }
                    },
                    "required": ["data_schema", "problem_type"]
                }
            )
        ]
    
    async def handle_recommend_features(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend features for model training"""
        data_schema = parameters["data_schema"]
        problem_type = parameters["problem_type"]
        quality_report = parameters.get("quality_report", {})
        
        self.logger.info("Recommending features", problem_type=problem_type)
        
        column_analysis = data_schema.get("column_analysis", {})
        
        # Identify features to keep
        recommended_features = []
        features_to_drop = []
        
        for col_name, col_info in column_analysis.items():
            data_type = col_info.get("data_type", "unknown")
            potential_role = col_info.get("potential_role", "feature")
            
            # Skip ID columns
            if potential_role == "identifier":
                features_to_drop.append(col_name)
                continue
            
            # Skip potential target columns for features
            if potential_role == "potential_target":
                continue
            
            # Include useful feature types
            if data_type in ["numeric", "categorical", "boolean"]:
                recommended_features.append(col_name)
            elif data_type == "datetime":
                # Recommend feature engineering for datetime
                recommended_features.append(col_name)
            else:
                # Text and mixed types might need special handling
                if "text" in data_type and "classification" in problem_type:
                    recommended_features.append(col_name)
                else:
                    features_to_drop.append(col_name)
        
        # Generate feature engineering suggestions
        feature_engineering_suggestions = self._get_feature_engineering_suggestions(
            column_analysis, problem_type
        )
        
        reasoning = f"Selected {len(recommended_features)} features based on data types and problem requirements. "
        reasoning += f"Excluded {len(features_to_drop)} features (IDs, irrelevant types)."
        
        return {
            "feature_recommendations": {
                "recommended_features": recommended_features,
                "features_to_drop": features_to_drop,
                "feature_engineering_suggestions": feature_engineering_suggestions,
                "reasoning": reasoning
            }
        }
    
    async def handle_recommend_models(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend suitable models"""
        problem_type = parameters["problem_type"]
        data_characteristics = parameters.get("data_characteristics", {})
        target_info = parameters.get("target_info", {})
        
        self.logger.info("Recommending models", problem_type=problem_type)
        
        # Get dataset size info
        shape = data_characteristics.get("shape", [1000, 10])
        n_samples = shape[0]
        n_features = shape[1]
        
        # Get model recommendations based on problem type
        if "classification" in problem_type:
            models = self._get_classification_models(n_samples, n_features, target_info)
        elif "regression" in problem_type:
            models = self._get_regression_models(n_samples, n_features, target_info)
        elif "clustering" in problem_type:
            models = self._get_clustering_models(n_samples, n_features)
        else:
            models = self._get_default_models()
        
        # Sort by recommended priority
        models.sort(key=lambda x: x.get("priority", 5))
        
        return {
            "model_recommendations": {
                "models": models,
                "dataset_size": n_samples,
                "feature_count": n_features,
                "recommendations": self._get_model_selection_advice(n_samples, n_features, problem_type)
            }
        }
    
    async def handle_suggest_feature_engineering(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest feature engineering transformations"""
        data_schema = parameters["data_schema"]
        problem_type = parameters["problem_type"]
        
        self.logger.info("Suggesting feature engineering")
        
        column_analysis = data_schema.get("column_analysis", {})
        suggestions = self._get_feature_engineering_suggestions(column_analysis, problem_type)
        
        return {
            "feature_engineering_suggestions": suggestions
        }
    
    def _get_feature_engineering_suggestions(self, column_analysis: Dict, problem_type: str) -> List[str]:
        """Generate feature engineering suggestions"""
        suggestions = []
        
        for col_name, col_info in column_analysis.items():
            data_type = col_info.get("data_type", "unknown")
            
            if data_type == "datetime":
                suggestions.append(f"Extract date features from {col_name} (year, month, day, weekday)")
            elif data_type == "categorical":
                suggestions.append(f"Consider one-hot encoding for {col_name}")
            elif data_type == "text":
                if "classification" in problem_type:
                    suggestions.append(f"Apply text vectorization to {col_name} (TF-IDF, Count)")
            elif data_type == "numeric":
                suggestions.append(f"Consider scaling/normalization for {col_name}")
        
        # General suggestions
        suggestions.append("Create interaction features between important variables")
        suggestions.append("Consider polynomial features for non-linear relationships")
        
        return suggestions
    
    def _get_classification_models(self, n_samples: int, n_features: int, target_info: Dict) -> List[Dict]:
        """Get classification model recommendations"""
        models = []
        
        # Random Forest
        models.append({
            "model_name": "Random Forest",
            "model_type": "ensemble",
            "priority": 1,
            "pros": [
                "Handles mixed data types well",
                "Provides feature importance",
                "Robust to outliers",
                "Good default performance"
            ],
            "cons": [
                "Can overfit with small datasets",
                "Less interpretable than linear models"
            ],
            "complexity": "medium",
            "interpretability": "medium",
            "performance_expectation": "high",
            "training_time": "medium",
            "suitable_for_size": n_samples >= 100
        })
        
        # Logistic Regression
        models.append({
            "model_name": "Logistic Regression",
            "model_type": "linear",
            "priority": 2,
            "pros": [
                "Highly interpretable",
                "Fast training and prediction",
                "Good baseline model",
                "Probabilistic output"
            ],
            "cons": [
                "Assumes linear relationship",
                "Sensitive to outliers",
                "Requires feature scaling"
            ],
            "complexity": "low",
            "interpretability": "high",
            "performance_expectation": "medium",
            "training_time": "fast",
            "suitable_for_size": True
        })
        
        # Gradient Boosting
        if n_samples >= 500:
            models.append({
                "model_name": "Gradient Boosting",
                "model_type": "ensemble",
                "priority": 3,
                "pros": [
                    "Often achieves best performance",
                    "Handles missing values",
                    "Feature importance available"
                ],
                "cons": [
                    "Prone to overfitting",
                    "Requires hyperparameter tuning",
                    "Longer training time"
                ],
                "complexity": "high",
                "interpretability": "low",
                "performance_expectation": "high",
                "training_time": "slow",
                "suitable_for_size": n_samples >= 500
            })
        
        return [model for model in models if model.get("suitable_for_size", True)]
    
    def _get_regression_models(self, n_samples: int, n_features: int, target_info: Dict) -> List[Dict]:
        """Get regression model recommendations"""
        models = []
        
        # Random Forest Regressor
        models.append({
            "model_name": "Random Forest Regressor",
            "model_type": "ensemble",
            "priority": 1,
            "pros": [
                "Handles non-linear relationships",
                "Robust to outliers",
                "Feature importance",
                "Good default performance"
            ],
            "cons": [
                "Can overfit",
                "Memory intensive for large datasets"
            ],
            "complexity": "medium",
            "interpretability": "medium",
            "performance_expectation": "high",
            "training_time": "medium",
            "suitable_for_size": n_samples >= 100
        })
        
        # Linear Regression
        models.append({
            "model_name": "Linear Regression",
            "model_type": "linear",
            "priority": 2,
            "pros": [
                "Highly interpretable",
                "Fast training and prediction",
                "Good baseline",
                "Works well with linear relationships"
            ],
            "cons": [
                "Assumes linear relationship",
                "Sensitive to outliers",
                "Requires feature scaling"
            ],
            "complexity": "low",
            "interpretability": "high",
            "performance_expectation": "medium",
            "training_time": "fast",
            "suitable_for_size": True
        })
        
        return models
    
    def _get_clustering_models(self, n_samples: int, n_features: int) -> List[Dict]:
        """Get clustering model recommendations"""
        models = []
        
        # K-Means
        models.append({
            "model_name": "K-Means",
            "model_type": "centroid",
            "priority": 1,
            "pros": [
                "Simple and fast",
                "Works well with spherical clusters",
                "Scalable to large datasets"
            ],
            "cons": [
                "Requires specifying number of clusters",
                "Sensitive to initialization",
                "Assumes spherical clusters"
            ],
            "complexity": "low",
            "interpretability": "high",
            "performance_expectation": "medium",
            "training_time": "fast",
            "suitable_for_size": True
        })
        
        return models
    
    def _get_default_models(self) -> List[Dict]:
        """Get default model recommendations"""
        return [{
            "model_name": "Random Forest",
            "model_type": "ensemble",
            "priority": 1,
            "pros": ["Versatile", "Good default choice"],
            "cons": ["May not be optimal for specific problems"],
            "complexity": "medium",
            "interpretability": "medium",
            "performance_expectation": "medium",
            "training_time": "medium",
            "suitable_for_size": True
        }]
    
    def _get_model_selection_advice(self, n_samples: int, n_features: int, problem_type: str) -> List[str]:
        """Get advice for model selection"""
        advice = []
        
        if n_samples < 1000:
            advice.append("Small dataset: prefer simpler models to avoid overfitting")
        elif n_samples > 100000:
            advice.append("Large dataset: can use more complex models")
        
        if n_features > n_samples:
            advice.append("High-dimensional data: consider regularization or feature selection")
        
        if "classification" in problem_type:
            advice.append("Start with logistic regression as baseline, then try ensemble methods")
        elif "regression" in problem_type:
            advice.append("Start with linear regression as baseline, then try tree-based methods")
        
        return advice