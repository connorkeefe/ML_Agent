"""
Parameter Builder for MCP Tool Calls
Dynamically builds parameters for tool calls based on context and schemas.
"""

from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class ParameterBuilder:
    """Builds parameters for MCP tool calls"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.context_store = {}
        
        # Parameter mapping rules
        self.mapping_rules = {
            "ml-classifier-server": {
                "classify_problem": {
                    "user_prompt": "user_input.prompt",
                    "data_preview": "derived.data_preview",
                    "context": "workflow.context"
                }
            },
            "data-ingestion-server": {
                "load_data": {
                    "filepath": "user_input.filepath",
                    "format_hint": "derived.file_extension"
                },
                "get_data_preview": {
                    "filepath": "user_input.filepath",
                    "n_rows": "constant.5"
                }
            },
            "eda-server": {
                "analyze_data_schema": {
                    "data_summary": "ingestion_result.standardized_data_summary",
                    "problem_type": "classification_result.problem_type"
                }
            },
            "ml-decision-server": {
                "recommend_features": {
                    "data_schema": "eda_result.schema_analysis",
                    "problem_type": "classification_result.problem_type",
                    "quality_report": "eda_result.quality_report"
                }
            }
        }
    
    async def build_parameters(self, tool_name: str, server_name: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build parameters for a tool call"""
        logger.info("Building parameters", tool_name=tool_name, server_name=server_name)
        
        # Get tool schema
        try:
            tool_schema = await self._get_tool_schema(server_name, tool_name)
        except Exception as e:
            logger.warning(f"Could not get tool schema: {e}")
            # Fallback to context data
            return context_data
        
        # Build parameters based on schema and mapping rules
        parameters = {}
        
        # Get mapping rules for this tool
        server_rules = self.mapping_rules.get(server_name, {})
        tool_rules = server_rules.get(tool_name, {})
        
        # Build each parameter
        for param_name, param_config in tool_schema.get('properties', {}).items():
            if param_name in tool_rules:
                # Use mapping rule
                value = self._resolve_mapping(tool_rules[param_name], context_data)
                if value is not None:
                    parameters[param_name] = value
            elif param_name in context_data:
                # Direct mapping from context
                parameters[param_name] = context_data[param_name]
            elif param_config.get('default') is not None:
                # Use default value
                parameters[param_name] = param_config['default']
            elif param_name in tool_schema.get('required', []):
                # Required parameter missing - try to derive
                derived_value = self._derive_parameter(param_name, param_config, context_data)
                if derived_value is not None:
                    parameters[param_name] = derived_value
                else:
                    logger.warning(f"Missing required parameter: {param_name}")
        
        logger.info("Built parameters", parameter_count=len(parameters))
        return parameters
    
    async def _get_tool_schema(self, server_name: str, tool_name: str) -> Dict[str, Any]:
        """Get schema for a specific tool"""
        try:
            tools = await self.mcp_client.list_tools(server_name)
            for tool in tools:
                if tool.name == tool_name:
                    return tool.inputSchema
            raise ValueError(f"Tool {tool_name} not found on server {server_name}")
        except Exception as e:
            logger.error(f"Failed to get tool schema: {e}")
            return {}
    
    def _resolve_mapping(self, mapping_path: str, context_data: Dict[str, Any]) -> Any:
        """Resolve a mapping path to a value"""
        if mapping_path.startswith("constant."):
            # Return constant value
            return mapping_path.split(".", 1)[1]
        
        if mapping_path.startswith("derived."):
            # Handle derived values
            derived_key = mapping_path.split(".", 1)[1]
            return self._get_derived_value(derived_key, context_data)
        
        # Navigate through nested dictionary
        keys = mapping_path.split(".")
        value = context_data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _get_derived_value(self, key: str, context_data: Dict[str, Any]) -> Any:
        """Get derived values that need calculation"""
        if key == "file_extension":
            filepath = context_data.get("user_input", {}).get("filepath", "")
            if filepath:
                return filepath.split(".")[-1].lower()
        
        elif key == "data_preview":
            # This would typically come from a previous tool call
            return context_data.get("data_preview", {})
        
        return None
    
    def _derive_parameter(self, param_name: str, param_config: Dict, context_data: Dict[str, Any]) -> Any:
        """Derive missing required parameters"""
        # Common parameter derivations
        if param_name == "filepath":
            return context_data.get("user_input", {}).get("filepath")
        
        elif param_name == "user_prompt":
            return context_data.get("user_input", {}).get("prompt")
        
        elif param_name == "problem_type":
            return context_data.get("classification_result", {}).get("problem_type")
        
        elif param_name == "data_summary":
            return context_data.get("ingestion_result", {}).get("data_summary")
        
        # Type-based defaults
        if param_config.get("type") == "object":
            return {}
        elif param_config.get("type") == "array":
            return []
        elif param_config.get("type") == "integer":
            return 0
        elif param_config.get("type") == "string":
            return ""
        
        return None