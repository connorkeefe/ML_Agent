"""
Data Ingestion MCP Server
Handles loading and standardizing data from various formats.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import chardet

from mcp.types import Tool, TextContent
from ..base_server import BaseMLServer, ServerConfig
from ...shared.schemas import (
    StandardizedData, 
    DataSchema, 
    ColumnInfo, 
    DataType
)


class IngestionServer(BaseMLServer):
    """MCP Server for data ingestion and standardization"""
    
    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.supported_formats = {}
        self.encoding_cache = {}
    
    async def initialize(self) -> None:
        """Initialize the ingestion server"""
        self.logger.info("Initializing Data Ingestion Server")
        
        # Load supported formats configuration
        self._load_format_handlers()
    
    def _load_format_handlers(self) -> None:
        """Load configuration for supported file formats"""
        self.supported_formats = {
            '.csv': self._load_csv,
            '.tsv': self._load_csv,
            '.txt': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.feather': self._load_feather
        }
    
    def get_tool_definitions(self) -> List[Tool]:
        """Return ingestion server tools"""
        return [
            Tool(
                name="load_data",
                description="Load data from a file path and detect format automatically",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the data file"
                        },
                        "format_hint": {
                            "type": "string",
                            "description": "Optional format hint (csv, excel, json, etc.)",
                            "enum": ["csv", "excel", "json", "parquet", "auto"]
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (auto-detected if not specified)"
                        },
                        "options": {
                            "type": "object",
                            "description": "Format-specific loading options",
                            "properties": {
                                "separator": {"type": "string"},
                                "header_row": {"type": "integer"},
                                "sheet_name": {"type": "string"},
                                "nrows": {"type": "integer"}
                            }
                        }
                    },
                    "required": ["filepath"]
                }
            ),
            Tool(
                name="validate_data",
                description="Validate loaded data for integrity and completeness",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data_summary": {
                            "type": "object",
                            "description": "Summary of loaded data for validation"
                        }
                    },
                    "required": ["data_summary"]
                }
            ),
            Tool(
                name="standardize_format",
                description="Convert raw data to standardized format",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "raw_data_info": {
                            "type": "object",
                            "description": "Information about raw data to standardize"
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Original file path for reference"
                        }
                    },
                    "required": ["raw_data_info", "filepath"]
                }
            ),
            Tool(
                name="get_data_preview",
                description="Get a preview of the loaded data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the data file"
                        },
                        "n_rows": {
                            "type": "integer",
                            "description": "Number of rows to preview",
                            "default": 5
                        },
                        "format_hint": {
                            "type": "string",
                            "description": "Optional format hint"
                        }
                    },
                    "required": ["filepath"]
                }
            )
        ]
    
    async def handle_load_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data loading from file"""
        filepath = parameters["filepath"]
        format_hint = parameters.get("format_hint", "auto")
        encoding = parameters.get("encoding")
        options = parameters.get("options", {})
        
        self.logger.info("Loading data", filepath=filepath, format_hint=format_hint)
        
        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Detect format if auto
        if format_hint == "auto":
            format_hint = self._detect_format(filepath)
        
        # Detect encoding if not provided
        if not encoding:
            encoding = self._detect_encoding(filepath)
        
        # Load data based on format
        try:
            raw_data, load_info = await self._load_file(filepath, format_hint, encoding, options)
            
            # Generate data summary
            data_summary = self._generate_data_summary(raw_data, filepath, load_info)
            
            return {
                "success": True,
                "data_summary": data_summary,
                "load_info": load_info,
                "filepath": filepath,
                "format": format_hint,
                "encoding": encoding,
                "shape": raw_data.shape,
                "columns": raw_data.columns.tolist(),
                "dtypes": raw_data.dtypes.astype(str).to_dict()
            }
            
        except Exception as e:
            self.logger.error("Failed to load data", filepath=filepath, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath,
                "suggestions": self._get_loading_suggestions(filepath, str(e))
            }
    
    async def handle_validate_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation"""
        data_summary = parameters["data_summary"]
        
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check for basic issues
        if data_summary.get("shape", [0, 0])[0] == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset is empty (no rows)")
        
        if data_summary.get("shape", [0, 0])[1] == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset has no columns")
        
        # Check for warnings
        if data_summary.get("shape", [0, 0])[0] < 100:
            validation_results["warnings"].append("Dataset is very small (< 100 rows)")
        
        if data_summary.get("missing_percentage", 0) > 50:
            validation_results["warnings"].append("High percentage of missing values (> 50%)")
        
        # Generate recommendations
        if data_summary.get("shape", [0, 0])[0] < 1000:
            validation_results["recommendations"].append("Consider gathering more data for better model performance")
        
        if len(data_summary.get("potential_issues", [])) > 0:
            validation_results["recommendations"].extend(data_summary["potential_issues"])
        
        return validation_results
    
    async def handle_standardize_format(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data standardization"""
        raw_data_info = parameters["raw_data_info"]
        filepath = parameters["filepath"]
        
        self.logger.info("Standardizing data format", filepath=filepath)
        
        try:
            # Re-load the data for standardization
            format_hint = raw_data_info.get("format", "auto")
            encoding = raw_data_info.get("encoding")
            
            raw_data, load_info = await self._load_file(filepath, format_hint, encoding, {})
            
            # Create standardized data schema
            schema = self._create_data_schema(raw_data)
            
            # Clean and standardize the data
            standardized_df = self._standardize_dataframe(raw_data)
            
            # Create metadata
            metadata = {
                "source_file": filepath,
                "load_timestamp": datetime.now().isoformat(),
                "original_format": format_hint,
                "row_count": len(standardized_df),
                "column_count": len(standardized_df.columns),
                "dtypes": standardized_df.dtypes.astype(str).to_dict(),
                "memory_usage": f"{standardized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                "load_info": load_info
            }
            
            # Note: In a real implementation, you'd store the actual DataFrame
            # For this MCP response, we return a summary
            return {
                "success": True,
                "standardized_data_summary": {
                    "shape": standardized_df.shape,
                    "columns": standardized_df.columns.tolist(),
                    "dtypes": standardized_df.dtypes.astype(str).to_dict(),
                    "sample_data": standardized_df.head().to_dict(),
                    "null_counts": standardized_df.isnull().sum().to_dict()
                },
                "schema": schema.dict(),
                "metadata": metadata,
                "data_id": f"data_{hash(filepath)}_{int(datetime.now().timestamp())}"
            }
            
        except Exception as e:
            self.logger.error("Failed to standardize data", filepath=filepath, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }
    
    async def handle_get_data_preview(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data preview generation"""
        filepath = parameters["filepath"]
        n_rows = parameters.get("n_rows", 5)
        format_hint = parameters.get("format_hint", "auto")
        
        try:
            # Detect format if needed
            if format_hint == "auto":
                format_hint = self._detect_format(filepath)
            
            # Load small sample for preview
            if format_hint == "csv":
                preview_df = pd.read_csv(filepath, nrows=n_rows)
            elif format_hint == "excel":
                preview_df = pd.read_excel(filepath, nrows=n_rows)
            elif format_hint == "json":
                preview_df = pd.read_json(filepath).head(n_rows)
            else:
                raise ValueError(f"Preview not supported for format: {format_hint}")
            
            return {
                "success": True,
                "preview": {
                    "columns": preview_df.columns.tolist(),
                    "dtypes": preview_df.dtypes.astype(str).to_dict(),
                    "sample_rows": preview_df.to_dict(orient="records"),
                    "shape": preview_df.shape
                },
                "filepath": filepath,
                "format": format_hint
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }
    
    def _detect_format(self, filepath: str) -> str:
        """Detect file format from extension"""
        extension = Path(filepath).suffix.lower()
        
        format_mapping = {
            '.csv': 'csv',
            '.tsv': 'csv',
            '.txt': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.feather': 'feather'
        }
        
        return format_mapping.get(extension, 'csv')  # Default to CSV
    
    def _detect_encoding(self, filepath: str) -> str:
        """Detect file encoding"""
        if filepath in self.encoding_cache:
            return self.encoding_cache[filepath]
        
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(10000)  # Read first 10KB
                result = chardet.detect(sample)
                encoding = result.get('encoding', 'utf-8')
                
                # Cache the result
                self.encoding_cache[filepath] = encoding
                return encoding
        except Exception:
            return 'utf-8'  # Default encoding
    
    async def _load_file(self, filepath: str, format_hint: str, encoding: str, options: Dict) -> tuple:
        """Load file based on format"""
        extension = Path(filepath).suffix.lower()
        
        if extension in ['.csv', '.tsv', '.txt'] or format_hint == 'csv':
            return await self._load_csv(filepath, encoding, options)
        elif extension in ['.xlsx', '.xls'] or format_hint == 'excel':
            return await self._load_excel(filepath, options)
        elif extension == '.json' or format_hint == 'json':
            return await self._load_json(filepath, encoding, options)
        elif extension == '.parquet' or format_hint == 'parquet':
            return await self._load_parquet(filepath, options)
        elif extension == '.feather' or format_hint == 'feather':
            return await self._load_feather(filepath, options)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    async def _load_csv(self, filepath: str, encoding: str, options: Dict) -> tuple:
        """Load CSV file"""
        load_options = {
            'encoding': encoding,
            'sep': options.get('separator', ','),
            'header': options.get('header_row', 0),
        }
        
        if 'nrows' in options:
            load_options['nrows'] = options['nrows']
        
        # Try different separators if the specified one fails
        separators = [load_options['sep'], ',', ';', '\t', '|']
        
        for sep in separators:
            try:
                load_options['sep'] = sep
                df = pd.read_csv(filepath, **load_options)
                
                # Check if loading was successful
                if len(df.columns) > 1 or len(df) > 0:
                    load_info = {
                        'separator': sep,
                        'encoding': encoding,
                        'header_row': load_options['header'],
                        'rows_loaded': len(df)
                    }
                    return df, load_info
            except Exception as e:
                if sep == separators[-1]:  # Last separator failed
                    raise e
                continue
        
        raise ValueError("Could not load CSV with any common separator")
    
    async def _load_excel(self, filepath: str, options: Dict) -> tuple:
        """Load Excel file"""
        load_options = {}
        
        if 'sheet_name' in options:
            load_options['sheet_name'] = options['sheet_name']
        if 'header_row' in options:
            load_options['header'] = options['header_row']
        if 'nrows' in options:
            load_options['nrows'] = options['nrows']
        
        df = pd.read_excel(filepath, **load_options)
        
        load_info = {
            'sheet_name': load_options.get('sheet_name', 0),
            'header_row': load_options.get('header', 0),
            'rows_loaded': len(df)
        }
        
        return df, load_info
    
    async def _load_json(self, filepath: str, encoding: str, options: Dict) -> tuple:
        """Load JSON file"""
        load_options = {'encoding': encoding}
        
        if 'nrows' in options:
            # For JSON, we'll load all and then slice
            df = pd.read_json(filepath, **load_options)
            df = df.head(options['nrows'])
        else:
            df = pd.read_json(filepath, **load_options)
        
        load_info = {
            'encoding': encoding,
            'rows_loaded': len(df)
        }
        
        return df, load_info
    
    async def _load_parquet(self, filepath: str, options: Dict) -> tuple:
        """Load Parquet file"""
        df = pd.read_parquet(filepath)
        
        if 'nrows' in options:
            df = df.head(options['nrows'])
        
        load_info = {
            'rows_loaded': len(df),
            'compression': 'auto'
        }
        
        return df, load_info
    
    async def _load_feather(self, filepath: str, options: Dict) -> tuple:
        """Load Feather file"""
        df = pd.read_feather(filepath)
        
        if 'nrows' in options:
            df = df.head(options['nrows'])
        
        load_info = {
            'rows_loaded': len(df)
        }
        
        return df, load_info
    
    def _generate_data_summary(self, df: pd.DataFrame, filepath: str, load_info: Dict) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        summary = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                "columns_with_missing": df.columns[df.isnull().any()].tolist(),
                "missing_by_column": df.isnull().sum().to_dict()
            },
            "data_types": {
                "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
            },
            "sample_data": df.head(3).to_dict(orient="records"),
            "potential_issues": [],
            "load_info": load_info
        }
        
        # Identify potential issues
        if summary["missing_data"]["missing_percentage"] > 30:
            summary["potential_issues"].append("High percentage of missing values")
        
        if df.shape[1] > 1000:
            summary["potential_issues"].append("Very high dimensional data (many columns)")
        
        if df.shape[0] < 100:
            summary["potential_issues"].append("Small dataset size")
        
        # Check for potential duplicate rows
        if df.duplicated().sum() > 0:
            summary["potential_issues"].append(f"{df.duplicated().sum()} duplicate rows found")
        
        return summary
    
    def _create_data_schema(self, df: pd.DataFrame) -> DataSchema:
        """Create standardized data schema"""
        columns = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = DataType.NUMERIC
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = DataType.DATETIME
            elif pd.api.types.is_bool_dtype(col_data):
                data_type = DataType.BOOLEAN
            elif col_data.dtype == 'object':
                # Try to determine if it's categorical or text
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    data_type = DataType.CATEGORICAL
                else:
                    data_type = DataType.TEXT
            else:
                data_type = DataType.MIXED
            
            # Calculate statistics
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
            unique_values = col_data.nunique()
            sample_values = col_data.dropna().head(5).tolist()
            
            # Generate statistics based on data type
            statistics = {}
            if data_type == DataType.NUMERIC:
                statistics = {
                    "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                    "std": float(col_data.std()) if not col_data.isnull().all() else None,
                    "min": float(col_data.min()) if not col_data.isnull().all() else None,
                    "max": float(col_data.max()) if not col_data.isnull().all() else None,
                    "median": float(col_data.median()) if not col_data.isnull().all() else None
                }
            
            columns[col] = ColumnInfo(
                name=col,
                dtype=str(col_data.dtype),
                data_type=data_type,
                missing_percentage=missing_pct,
                unique_values=unique_values,
                sample_values=sample_values,
                statistics=statistics
            )
        
        return DataSchema(
            columns=columns,
            shape=df.shape,
            memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        # Create a copy to avoid modifying original
        standardized_df = df.copy()
        
        # Clean column names
        standardized_df.columns = [
            col.strip().lower().replace(' ', '_').replace('-', '_')
            for col in standardized_df.columns
        ]
        
        # Convert obvious datetime columns
        for col in standardized_df.columns:
            if standardized_df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    pd.to_datetime(standardized_df[col].head(100), errors='raise')
                    standardized_df[col] = pd.to_datetime(standardized_df[col], errors='coerce')
                except:
                    pass
        
        return standardized_df
    
    def _get_loading_suggestions(self, filepath: str, error_msg: str) -> List[str]:
        """Get suggestions for fixing loading errors"""
        suggestions = []
        
        if "encoding" in error_msg.lower():
            suggestions.append("Try specifying a different encoding (utf-8, latin-1, cp1252)")
        
        if "separator" in error_msg.lower() or "delimiter" in error_msg.lower():
            suggestions.append("Try different separators: comma, semicolon, tab, or pipe")
        
        if "sheet" in error_msg.lower():
            suggestions.append("Specify the correct sheet name for Excel files")
        
        if "permission" in error_msg.lower():
            suggestions.append("Check file permissions and ensure file is not open in another application")
        
        if not suggestions:
            suggestions.append("Verify file format and ensure file is not corrupted")
        
        return suggestions