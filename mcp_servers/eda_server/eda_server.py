"""
Exploratory Data Analysis MCP Server
Handles data analysis, quality assessment, and visualization recommendations.
"""

from typing import Dict, List, Any, Optional
from mcp.types import Tool
from ..base_server import BaseMLServer, ServerConfig
from ...shared.schemas import DataType, EDAResult


class EDAServer(BaseMLServer):
    """MCP Server for exploratory data analysis"""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config)
    
    async def initialize(self) -> None:
        """Initialize the EDA server"""
        self.logger.info("Initializing EDA Server")
    
    def get_tool_definitions(self) -> List[Tool]:
        """Return EDA server tools"""
        return [
            Tool(
                name="analyze_data_schema",
                description="Analyze the structure and characteristics of the dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of the standardized data"
                        },
                        "problem_type": {
                            "type": "string",
                            "description": "The identified ML problem type"
                        }
                    },
                    "required": ["data_summary"]
                }
            ),
            Tool(
                name="generate_data_quality_report",
                description="Generate comprehensive data quality assessment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of the dataset"
                        }
                    },
                    "required": ["data_summary"]
                }
            ),
            Tool(
                name="identify_target_variable",
                description="Identify and analyze the target variable for supervised learning",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of the dataset"
                        },
                        "problem_type": {
                            "type": "string",
                            "description": "The ML problem type"
                        },
                        "user_hints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "User hints about target variable"
                        }
                    },
                    "required": ["data_summary", "problem_type"]
                }
            ),
            Tool(
                name="detect_time_series_patterns",
                description="Detect time series patterns and characteristics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of the dataset"
                        }
                    },
                    "required": ["data_summary"]
                }
            ),
            Tool(
                name="generate_correlation_analysis",
                description="Generate correlation analysis between features",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of the dataset"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Target variable column name"
                        }
                    },
                    "required": ["data_summary"]
                }
            )
        ]
    
    async def handle_analyze_data_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data schema and structure"""
        data_summary = parameters["data_summary"]
        problem_type = parameters.get("problem_type")
        
        self.logger.info("Analyzing data schema", problem_type=problem_type)
        
        # Extract basic information
        columns = data_summary.get("columns", [])
        dtypes = data_summary.get("dtypes", {})
        shape = data_summary.get("shape", [0, 0])
        
        # Analyze column characteristics
        column_analysis = {}
        for col in columns:
            dtype = dtypes.get(col, "unknown")
            column_analysis[col] = {
                "dtype": dtype,
                "data_type": self._classify_data_type(dtype, col),
                "potential_role": self._determine_column_role(col, dtype, problem_type)
            }
        
        # Generate schema insights
        insights = self._generate_schema_insights(column_analysis, shape, problem_type)
        
        return {
            "schema_analysis": {
                "column_analysis": column_analysis,
                "shape": shape,
                "insights": insights,
                "recommendations": self._generate_schema_recommendations(column_analysis, problem_type)
            }
        }
    
    async def handle_generate_data_quality_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data quality assessment"""
        data_summary = parameters["data_summary"]
        
        self.logger.info("Generating data quality report")
        
        # Extract quality metrics
        basic_info = data_summary.get("basic_info", {})
        missing_data = data_summary.get("missing_data", {})
        
        shape = basic_info.get("shape", [0, 0])
        total_cells = shape[0] * shape[1] if len(shape) >= 2 else 0
        
        quality_report = {
            "total_missing_values": missing_data.get("total_missing", 0),
            "missing_percentage": missing_data.get("missing_percentage", 0),
            "duplicate_rows": 0,  # Would calculate from actual data
            "duplicate_percentage": 0.0,
            "outliers_detected": {},
            "data_types_consistent": True,
            "recommendations": []
        }
        
        # Generate quality recommendations
        if quality_report["missing_percentage"] > 30:
            quality_report["recommendations"].append(
                "High percentage of missing values detected. Consider imputation strategies."
            )
        
        if shape[0] < 1000:
            quality_report["recommendations"].append(
                "Small dataset size. Consider gathering more data for better model performance."
            )
        
        return {"quality_report": quality_report}
    
    async def handle_identify_target_variable(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Identify target variable for supervised learning"""
        data_summary = parameters["data_summary"]
        problem_type = parameters["problem_type"]
        user_hints = parameters.get("user_hints", [])
        
        self.logger.info("Identifying target variable", problem_type=problem_type)
        
        columns = data_summary.get("basic_info", {}).get("columns", [])
        dtypes = data_summary.get("basic_info", {}).get("dtypes", {})
        
        # Target variable indicators
        target_indicators = [
            "target", "label", "class", "y", "output", "prediction",
            "outcome", "result", "response", "dependent"
        ]
        
        # Score columns for likelihood of being target
        column_scores = {}
        for col in columns:
            score = 0
            col_lower = col.lower()
            
            # Check for target indicators in column name
            for indicator in target_indicators:
                if indicator in col_lower:
                    score += 10
            
            # Check user hints
            for hint in user_hints:
                if hint.lower() in col_lower:
                    score += 15
            
            # Check data type appropriateness
            dtype = dtypes.get(col, "")
            if "supervised_classification" in problem_type:
                if "object" in dtype or "category" in dtype:
                    score += 5
            elif "supervised_regression" in problem_type:
                if "int" in dtype or "float" in dtype:
                    score += 5
            
            column_scores[col] = score
        
        # Find best candidate
        if column_scores:
            best_column = max(column_scores.keys(), key=lambda k: column_scores[k])
            best_score = column_scores[best_column]
            
            if best_score > 0:
                target_type = "classification" if "classification" in problem_type else "regression"
                
                return {
                    "target_analysis": {
                        "column_name": best_column,
                        "target_type": target_type,
                        "confidence": min(best_score / 20.0, 1.0),
                        "alternatives": [col for col, score in column_scores.items() 
                                       if score > 5 and col != best_column]
                    }
                }
        
        return {
            "target_analysis": None,
            "message": "Could not identify target variable automatically",
            "suggestions": "Please specify the target column name"
        }
    
    async def handle_detect_time_series_patterns(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect time series patterns"""
        data_summary = parameters["data_summary"]
        
        self.logger.info("Detecting time series patterns")
        
        data_types = data_summary.get("data_types", {})
        datetime_columns = data_types.get("datetime_columns", [])
        columns = data_summary.get("basic_info", {}).get("columns", [])
        
        # Look for datetime columns
        time_indicators = ["date", "time", "timestamp", "day", "month", "year"]
        potential_time_cols = []
        
        for col in columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in time_indicators):
                potential_time_cols.append(col)
        
        is_time_series = len(datetime_columns) > 0 or len(potential_time_cols) > 0
        
        return {
            "time_series_features": {
                "is_time_series": is_time_series,
                "datetime_columns": datetime_columns,
                "potential_time_columns": potential_time_cols,
                "recommendations": [
                    "Convert time columns to datetime format",
                    "Check for regular time intervals",
                    "Consider seasonal patterns"
                ] if is_time_series else []
            }
        }
    
    async def handle_generate_correlation_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate correlation analysis"""
        data_summary = parameters["data_summary"]
        target_column = parameters.get("target_column")
        
        self.logger.info("Generating correlation analysis", target_column=target_column)
        
        data_types = data_summary.get("data_types", {})
        numerical_columns = data_types.get("numerical_columns", [])
        
        # Mock correlation analysis (would use actual data in real implementation)
        correlations = {}
        if target_column and target_column in numerical_columns:
            for col in numerical_columns:
                if col != target_column:
                    # Mock correlation values
                    correlations[col] = 0.5 if "important" in col.lower() else 0.2
        
        return {
            "correlation_analysis": {
                "target_correlations": correlations,
                "high_correlation_features": [col for col, corr in correlations.items() if abs(corr) > 0.7],
                "recommendations": [
                    "Remove highly correlated features to reduce multicollinearity",
                    "Focus on features with strong target correlation"
                ]
            }
        }
    
    def _classify_data_type(self, dtype: str, column_name: str) -> str:
        """Classify the data type of a column"""
        dtype_lower = str(dtype).lower()
        col_lower = column_name.lower()
        
        if "int" in dtype_lower or "float" in dtype_lower:
            return "numeric"
        elif "datetime" in dtype_lower or "date" in col_lower:
            return "datetime"
        elif "bool" in dtype_lower:
            return "boolean"
        elif "object" in dtype_lower:
            # Try to determine if categorical or text
            if any(word in col_lower for word in ["category", "type", "class", "group"]):
                return "categorical"
            else:
                return "text"
        else:
            return "mixed"
    
    def _determine_column_role(self, column_name: str, dtype: str, problem_type: str) -> str:
        """Determine the likely role of a column"""
        col_lower = column_name.lower()
        
        # Target indicators
        target_indicators = ["target", "label", "class", "y", "output"]
        if any(indicator in col_lower for indicator in target_indicators):
            return "potential_target"
        
        # ID indicators
        id_indicators = ["id", "index", "key", "identifier"]
        if any(indicator in col_lower for indicator in id_indicators):
            return "identifier"
        
        # Datetime indicators
        if "datetime" in str(dtype).lower() or any(word in col_lower for word in ["date", "time"]):
            return "datetime"
        
        # Feature
        return "feature"
    
    def _generate_schema_insights(self, column_analysis: Dict, shape: List[int], problem_type: str) -> List[str]:
        """Generate insights about the data schema"""
        insights = []
        
        total_columns = len(column_analysis)
        numeric_columns = sum(1 for col in column_analysis.values() if col["data_type"] == "numeric")
        categorical_columns = sum(1 for col in column_analysis.values() if col["data_type"] == "categorical")
        
        insights.append(f"Dataset has {shape[0]} rows and {total_columns} columns")
        insights.append(f"Found {numeric_columns} numeric and {categorical_columns} categorical features")
        
        # Problem-specific insights
        if "supervised" in problem_type:
            potential_targets = [name for name, info in column_analysis.items() 
                               if info["potential_role"] == "potential_target"]
            if potential_targets:
                insights.append(f"Potential target variables: {', '.join(potential_targets)}")
            else:
                insights.append("No obvious target variable found - may need user specification")
        
        return insights
    
    def _generate_schema_recommendations(self, column_analysis: Dict, problem_type: str) -> List[str]:
        """Generate recommendations based on schema analysis"""
        recommendations = []
        
        # Check for ID columns
        id_columns = [name for name, info in column_analysis.items() 
                     if info["potential_role"] == "identifier"]
        if id_columns:
            recommendations.append(f"Consider removing ID columns: {', '.join(id_columns)}")
        
        # Check for datetime columns
        datetime_columns = [name for name, info in column_analysis.items() 
                          if info["data_type"] == "datetime"]
        if datetime_columns:
            recommendations.append("Extract useful features from datetime columns (year, month, day, etc.)")
        
        # Problem-specific recommendations
        if "classification" in problem_type:
            recommendations.append("Ensure target variable is properly encoded for classification")
        elif "regression" in problem_type:
            recommendations.append("Check target variable for outliers and distribution")
        
        return recommendations