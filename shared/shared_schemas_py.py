"""
Shared schemas and data models for the ML Processor MCP system.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class MLProblemType(str, Enum):
    """Types of ML problems"""
    SUPERVISED_CLASSIFICATION = "supervised_classification"
    SUPERVISED_REGRESSION = "supervised_regression"
    SEMI_SUPERVISED_CLASSIFICATION = "semi_supervised_classification"
    SEMI_SUPERVISED_REGRESSION = "semi_supervised_regression"
    UNSUPERVISED_CLUSTERING = "unsupervised_clustering"
    UNSUPERVISED_DIMENSIONALITY_REDUCTION = "unsupervised_dimensionality_reduction"
    UNSUPERVISED_ANOMALY_DETECTION = "unsupervised_anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"


class DataType(str, Enum):
    """Data types for columns"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"


class WorkflowState(str, Enum):
    """Workflow states"""
    INIT = "init"
    PROBLEM_CLASSIFICATION = "problem_classification"
    DATA_INGESTION = "data_ingestion"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    DECISION_MAKING = "decision_making"
    USER_CONFIRMATION = "user_confirmation"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATION = "evaluation"
    EXPORT = "export"
    COMPLETED = "completed"
    ERROR = "error"


class ColumnInfo(BaseModel):
    """Information about a data column"""
    name: str
    dtype: str
    data_type: DataType
    missing_percentage: float = Field(ge=0, le=100)
    unique_values: int = Field(ge=0)
    sample_values: List[Any] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None
    
    @validator('missing_percentage')
    def validate_missing_percentage(cls, v):
        return round(v, 2)


class DataSchema(BaseModel):
    """Schema information for a dataset"""
    columns: Dict[str, ColumnInfo]
    shape: tuple[int, int]
    memory_usage: str
    file_size: Optional[str] = None
    has_header: bool = True
    delimiter: Optional[str] = None
    encoding: Optional[str] = None


class StandardizedData(BaseModel):
    """Standardized data format used throughout the system"""
    data: Any  # pandas DataFrame (can't serialize directly)
    metadata: Dict[str, Any]
    schema: DataSchema
    
    class Config:
        arbitrary_types_allowed = True


class ProblemClassification(BaseModel):
    """Result of problem classification"""
    problem_type: MLProblemType
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    required_clarifications: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    unsupported_problem: bool = False
    suggested_goals: List[str] = Field(default_factory=list)


class TargetAnalysis(BaseModel):
    """Analysis of target variable"""
    column_name: str
    target_type: Literal["classification", "regression"]
    missing_percentage: float = Field(ge=0, le=100)
    class_distribution: Optional[Dict[str, int]] = None
    statistics: Optional[Dict[str, float]] = None
    is_balanced: Optional[bool] = None
    recommended_metrics: List[str] = Field(default_factory=list)


class TimeSeriesFeatures(BaseModel):
    """Time series characteristics"""
    is_time_series: bool = False
    datetime_columns: List[str] = Field(default_factory=list)
    frequency: Optional[str] = None
    seasonality: Optional[Dict[str, Any]] = None
    trend: Optional[str] = None
    stationarity: Optional[bool] = None


class DataQualityReport(BaseModel):
    """Data quality assessment"""
    total_missing_values: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    outliers_detected: Dict[str, int] = Field(default_factory=dict)
    data_types_consistent: bool = True
    recommendations: List[str] = Field(default_factory=list)


class EDAResult(BaseModel):
    """Result of exploratory data analysis"""
    data_schema: DataSchema
    target_analysis: Optional[TargetAnalysis] = None
    time_series_features: TimeSeriesFeatures
    data_quality_report: DataQualityReport
    correlations: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    visualizations: List[str] = Field(default_factory=list)


class ModelRecommendation(BaseModel):
    """Recommendation for a specific model"""
    model_name: str
    model_type: str
    pros: List[str]
    cons: List[str]
    complexity: Literal["low", "medium", "high"]
    interpretability: Literal["low", "medium", "high"]
    performance_expectation: Literal["low", "medium", "high"]
    training_time: Literal["fast", "medium", "slow"]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    data_requirements: Dict[str, Any] = Field(default_factory=dict)


class FeatureRecommendation(BaseModel):
    """Recommendation for feature selection"""
    recommended_features: List[str]
    features_to_drop: List[str] = Field(default_factory=list)
    feature_engineering_suggestions: List[str] = Field(default_factory=list)
    reasoning: str


class PreprocessingStep(BaseModel):
    """A single preprocessing step"""
    name: str
    method: str
    columns: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    order: int = 0


class PreprocessingPlan(BaseModel):
    """Complete preprocessing plan"""
    steps: List[PreprocessingStep]
    train_test_split: Optional[Dict[str, Any]] = None
    validation_strategy: Optional[str] = None
    estimated_time: Optional[str] = None


class TrainingSpecification(BaseModel):
    """Specification for model training"""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    cross_validation: Optional[Dict[str, Any]] = None
    early_stopping: Optional[Dict[str, Any]] = None


class ModelMetrics(BaseModel):
    """Model evaluation metrics"""
    primary_metric: str
    primary_score: float
    all_metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None


class TrainingResult(BaseModel):
    """Result of model training"""
    model_type: str
    training_time: float
    metrics: ModelMetrics
    model_path: Optional[str] = None
    best_parameters: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None


class ExportPackage(BaseModel):
    """Complete export package"""
    model_files: List[str]
    code_files: List[str]
    documentation_files: List[str]
    requirements_file: str
    export_path: str
    package_size: str
    created_at: datetime = Field(default_factory=datetime.now)


class WorkflowContext(BaseModel):
    """Complete workflow context"""
    workflow_id: str
    current_state: WorkflowState
    problem_classification: Optional[ProblemClassification] = None
    standardized_data: Optional[StandardizedData] = None
    eda_result: Optional[EDAResult] = None
    feature_recommendations: Optional[FeatureRecommendation] = None
    model_recommendations: List[ModelRecommendation] = Field(default_factory=list)
    selected_model: Optional[str] = None
    preprocessing_plan: Optional[PreprocessingPlan] = None
    training_specification: Optional[TrainingSpecification] = None
    training_result: Optional[TrainingResult] = None
    export_package: Optional[ExportPackage] = None
    user_inputs: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, datetime] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class MCPToolCall(BaseModel):
    """MCP tool call representation"""
    server_name: str
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class MCPToolResult(BaseModel):
    """MCP tool result representation"""
    tool_call: MCPToolCall
    result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ServerHealth(BaseModel):
    """Server health check result"""
    server_name: str
    status: Literal["healthy", "unhealthy", "unknown"]
    uptime: Optional[str] = None
    resources_loaded: int = 0
    tools_registered: int = 0
    last_activity: Optional[datetime] = None
    error_message: Optional[str] = None