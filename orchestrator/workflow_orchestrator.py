"""
Workflow Orchestrator for ML Processor MCP System
Manages the flow between different MCP servers and coordinates the ML pipeline.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog

from mcp.client import Client as MCPClient
from mcp.client.websocket import WebSocketTransport

from ..shared.schemas import (
    WorkflowState, 
    WorkflowContext, 
    MCPToolCall, 
    MCPToolResult,
    ProblemClassification,
    StandardizedData
)
from .parameter_builder import ParameterBuilder
from .resource_manager import ResourceManager
from .context_manager import ContextManager
from .error_handler import ErrorHandler


class WorkflowOrchestrator:
    """Main orchestrator for the ML processing workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.mcp_client = None
        self.parameter_builder = None
        self.resource_manager = None
        self.context_manager = ContextManager()
        self.error_handler = ErrorHandler(self)
        
        # Workflow state
        self.current_context: Optional[WorkflowContext] = None
        self.server_registry = {}
        self.execution_history = []
        
        # State transition rules
        self.state_transitions = self._setup_state_transitions()
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and connect to MCP servers"""
        self.logger.info("Initializing Workflow Orchestrator")
        
        # Initialize MCP client
        await self._initialize_mcp_client()
        
        # Discover and register servers
        await self._discover_servers()
        
        # Initialize supporting components
        self.parameter_builder = ParameterBuilder(self.mcp_client)
        self.resource_manager = ResourceManager(self.mcp_client)
        
        # Initialize resources
        await self.resource_manager.initialize_resources()
        
        self.logger.info("Orchestrator initialization complete")
    
    async def _initialize_mcp_client(self) -> None:
        """Initialize MCP client connections"""
        self.mcp_client = MCPClient()
        
        # Connect to servers defined in config
        servers_config = self.config.get("servers", {})
        
        for server_name, server_config in servers_config.items():
            try:
                transport = WebSocketTransport(
                    f"ws://{server_config['host']}:{server_config['port']}"
                )
                
                await self.mcp_client.connect(server_name, transport)
                self.logger.info(f"Connected to server: {server_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_name}: {e}")
    
    async def _discover_servers(self) -> None:
        """Discover available servers and their capabilities"""
        connected_servers = await self.mcp_client.list_servers()
        
        for server in connected_servers:
            try:
                # Get server capabilities
                tools = await self.mcp_client.list_tools(server.name)
                resources = await self.mcp_client.list_resources(server.name)
                
                self.server_registry[server.name] = {
                    'name': server.name,
                    'tools': {tool.name: tool for tool in tools},
                    'resources': {res.name: res for res in resources},
                    'status': 'connected',
                    'last_activity': datetime.now()
                }
                
                self.logger.info(
                    "Registered server",
                    server=server.name,
                    tools_count=len(tools),
                    resources_count=len(resources)
                )
                
            except Exception as e:
                self.logger.error(f"Failed to discover server {server.name}: {e}")
    
    def _setup_state_transitions(self) -> Dict[WorkflowState, callable]:
        """Setup state transition logic"""
        return {
            WorkflowState.INIT: self._from_init_state,
            WorkflowState.PROBLEM_CLASSIFICATION: self._from_classification_state,
            WorkflowState.DATA_INGESTION: self._from_ingestion_state,
            WorkflowState.EXPLORATORY_ANALYSIS: self._from_eda_state,
            WorkflowState.DECISION_MAKING: self._from_decision_state,
            WorkflowState.USER_CONFIRMATION: self._from_user_confirmation_state,
            WorkflowState.PREPROCESSING: self._from_preprocessing_state,
            WorkflowState.TRAINING: self._from_training_state,
            WorkflowState.EVALUATION: self._from_evaluation_state,
            WorkflowState.EXPORT: self._from_export_state
        }
    
    async def start_workflow(self, user_input: Dict[str, Any]) -> WorkflowContext:
        """Start a new ML processing workflow"""
        self.logger.info("Starting new workflow", user_input_keys=list(user_input.keys()))
        
        # Create new workflow context
        workflow_id = f"workflow_{int(datetime.now().timestamp())}"
        self.current_context = WorkflowContext(
            workflow_id=workflow_id,
            current_state=WorkflowState.INIT,
            user_inputs=user_input,
            timestamps={'start': datetime.now()}
        )
        
        # Begin workflow execution
        return await self.execute_workflow()
    
    async def execute_workflow(self) -> WorkflowContext:
        """Execute the main workflow loop"""
        while self.current_context.current_state not in [WorkflowState.COMPLETED, WorkflowState.ERROR]:
            try:
                self.logger.info(
                    "Executing workflow state",
                    state=self.current_context.current_state,
                    workflow_id=self.current_context.workflow_id
                )
                
                # Execute current state
                result = await self._execute_current_state()
                
                # Update context
                self._update_context_from_result(result)
                
                # Determine next state
                next_state = await self._determine_next_state(result)
                
                # Transition to next state
                await self._transition_to_state(next_state)
                
                # Record execution
                self.execution_history.append({
                    'state': self.current_context.current_state,
                    'timestamp': datetime.now(),
                    'result_summary': self._summarize_result(result)
                })
                
            except Exception as e:
                await self.error_handler.handle_error(e, self.current_context.dict())
                break
        
        # Mark completion timestamp
        self.current_context.timestamps['end'] = datetime.now()
        return self.current_context
    
    async def _execute_current_state(self) -> Dict[str, Any]:
        """Execute the current workflow state"""
        state = self.current_context.current_state
        
        state_handlers = {
            WorkflowState.INIT: self._handle_init_state,
            WorkflowState.PROBLEM_CLASSIFICATION: self._handle_classification_state,
            WorkflowState.DATA_INGESTION: self._handle_ingestion_state,
            WorkflowState.EXPLORATORY_ANALYSIS: self._handle_eda_state,
            WorkflowState.DECISION_MAKING: self._handle_decision_state,
            WorkflowState.USER_CONFIRMATION: self._handle_user_confirmation_state,
            WorkflowState.PREPROCESSING: self._handle_preprocessing_state,
            WorkflowState.TRAINING: self._handle_training_state,
            WorkflowState.EVALUATION: self._handle_evaluation_state,
            WorkflowState.EXPORT: self._handle_export_state
        }
        
        handler = state_handlers.get(state)
        if handler:
            return await handler()
        else:
            raise ValueError(f"No handler for state: {state}")
    
    async def _handle_init_state(self) -> Dict[str, Any]:
        """Handle initialization state"""
        user_inputs = self.current_context.user_inputs
        
        # Check if we have required inputs
        required_inputs = ['prompt']
        missing_inputs = [inp for inp in required_inputs if inp not in user_inputs]
        
        if missing_inputs:
            return {
                'needs_user_input': True,
                'missing_inputs': missing_inputs,
                'message': f"Please provide: {', '.join(missing_inputs)}"
            }
        
        # Check if filepath is provided
        if 'filepath' not in user_inputs:
            return {
                'needs_user_input': True,
                'missing_inputs': ['filepath'],
                'message': "Please provide the path to your data file"
            }
        
        return {
            'ready_for_classification': True,
            'user_prompt': user_inputs['prompt'],
            'filepath': user_inputs['filepath']
        }
    
    async def _handle_classification_state(self) -> Dict[str, Any]:
        """Handle problem classification state"""
        # Get data preview if filepath is available
        data_preview = {}
        if 'filepath' in self.current_context.user_inputs:
            try:
                preview_result = await self._call_tool(
                    'data-ingestion-server',
                    'get_data_preview',
                    {
                        'filepath': self.current_context.user_inputs['filepath'],
                        'n_rows': 5
                    }
                )
                if preview_result.success:
                    data_preview = preview_result.result.get('preview', {})
            except Exception as e:
                self.logger.warning(f"Could not get data preview: {e}")
        
        # Build parameters for classification
        parameters = await self.parameter_builder.build_parameters(
            tool_name="classify_problem",
            server_name="ml-classifier-server",
            context_data={
                "user_prompt": self.current_context.user_inputs.get('prompt', ''),
                "data_preview": data_preview,
                "context": self.current_context.dict()
            }
        )
        
        # Call classification tool
        result = await self._call_tool(
            'ml-classifier-server',
            'classify_problem',
            parameters
        )
        
        if result.success:
            classification = result.result.get('classification', {})
            return {
                'classification_result': classification,
                'confidence': classification.get('confidence', 0),
                'needs_clarification': classification.get('needs_clarification', False),
                'required_clarifications': classification.get('required_clarifications', [])
            }
        else:
            raise Exception(f"Classification failed: {result.error_message}")
    
    async def _handle_ingestion_state(self) -> Dict[str, Any]:
        """Handle data ingestion state"""
        filepath = self.current_context.user_inputs['filepath']
        
        # Load data
        load_result = await self._call_tool(
            'data-ingestion-server',
            'load_data',
            {'filepath': filepath}
        )
        
        if not load_result.success:
            raise Exception(f"Data loading failed: {load_result.error_message}")
        
        # Validate data
        validate_result = await self._call_tool(
            'data-ingestion-server',
            'validate_data',
            {'data_summary': load_result.result['data_summary']}
        )
        
        if not validate_result.success:
            raise Exception(f"Data validation failed: {validate_result.error_message}")
        
        # Standardize format
        standardize_result = await self._call_tool(
            'data-ingestion-server',
            'standardize_format',
            {
                'raw_data_info': load_result.result,
                'filepath': filepath
            }
        )
        
        if not standardize_result.success:
            raise Exception(f"Data standardization failed: {standardize_result.error_message}")
        
        return {
            'ingestion_success': True,
            'data_summary': load_result.result['data_summary'],
            'validation_result': validate_result.result,
            'standardized_data': standardize_result.result,
            'data_id': standardize_result.result['data_id']
        }
    
    async def _handle_eda_state(self) -> Dict[str, Any]:
        """Handle exploratory data analysis state"""
        # Get standardized data info from previous step
        standardized_data = self.current_context.dict().get('ingestion_result', {}).get('standardized_data', {})
        problem_type = self.current_context.problem_classification.problem_type if self.current_context.problem_classification else None
        
        # Analyze data schema
        schema_result = await self._call_tool(
            'eda-server',
            'analyze_data_schema',
            {
                'data_summary': standardized_data.get('standardized_data_summary', {}),
                'problem_type': problem_type
            }
        )
        
        # Generate data quality report
        quality_result = await self._call_tool(
            'eda-server',
            'generate_data_quality_report',
            {
                'data_summary': standardized_data.get('standardized_data_summary', {})
            }
        )
        
        # Analyze target variable if supervised learning
        target_result = None
        if problem_type and 'supervised' in problem_type:
            # Try to identify target variable
            target_result = await self._call_tool(
                'eda-server',
                'identify_target_variable',
                {
                    'data_summary': standardized_data.get('standardized_data_summary', {}),
                    'problem_type': problem_type,
                    'user_hints': self.current_context.user_inputs.get('target_hints', [])
                }
            )
        
        return {
            'schema_analysis': schema_result.result if schema_result.success else {},
            'quality_report': quality_result.result if quality_result.success else {},
            'target_analysis': target_result.result if target_result and target_result.success else None,
            'eda_complete': True
        }
    
    async def _handle_decision_state(self) -> Dict[str, Any]:
        """Handle ML decision making state"""
        # Get EDA results
        eda_result = self.current_context.dict().get('eda_result', {})
        problem_type = self.current_context.problem_classification.problem_type if self.current_context.problem_classification else None
        
        # Get feature recommendations
        feature_result = await self._call_tool(
            'ml-decision-server',
            'recommend_features',
            {
                'data_schema': eda_result.get('schema_analysis', {}),
                'problem_type': problem_type,
                'quality_report': eda_result.get('quality_report', {})
            }
        )
        
        # Get model recommendations
        model_result = await self._call_tool(
            'ml-decision-server',
            'recommend_models',
            {
                'problem_type': problem_type,
                'data_characteristics': eda_result.get('schema_analysis', {}),
                'target_info': eda_result.get('target_analysis', {})
            }
        )
        
        return {
            'feature_recommendations': feature_result.result if feature_result.success else {},
            'model_recommendations': model_result.result if model_result.success else {},
            'needs_user_selection': True
        }
    
    async def _handle_user_confirmation_state(self) -> Dict[str, Any]:
        """Handle user confirmation/input state"""
        # This would typically wait for user input
        # For now, we'll simulate automatic selection
        decision_result = self.current_context.dict().get('decision_result', {})
        
        # Auto-select first recommended model for demo
        model_recommendations = decision_result.get('model_recommendations', {}).get('models', [])
        selected_model = model_recommendations[0] if model_recommendations else None
        
        # Auto-accept feature recommendations
        feature_recommendations = decision_result.get('feature_recommendations', {})
        
        return {
            'user_selections': {
                'selected_model': selected_model,
                'approved_features': feature_recommendations.get('recommended_features', []),
                'preprocessing_preferences': {}
            },
            'confirmation_complete': True
        }
    
    async def _handle_preprocessing_state(self) -> Dict[str, Any]:
        """Handle data preprocessing state"""
        user_selections = self.current_context.dict().get('user_confirmation_result', {}).get('user_selections', {})
        eda_result = self.current_context.dict().get('eda_result', {})
        
        # Generate preprocessing plan
        plan_result = await self._call_tool(
            'preprocessing-server',
            'generate_preprocessing_plan',
            {
                'data_schema': eda_result.get('schema_analysis', {}),
                'selected_features': user_selections.get('approved_features', []),
                'problem_type': self.current_context.problem_classification.problem_type,
                'quality_report': eda_result.get('quality_report', {})
            }
        )
        
        # Execute preprocessing
        if plan_result.success:
            execute_result = await self._call_tool(
                'preprocessing-server',
                'execute_preprocessing',
                {
                    'preprocessing_plan': plan_result.result,
                    'data_id': self.current_context.dict().get('ingestion_result', {}).get('data_id'),
                    'selected_features': user_selections.get('approved_features', [])
                }
            )
            
            return {
                'preprocessing_plan': plan_result.result,
                'preprocessing_result': execute_result.result if execute_result.success else {},
                'preprocessing_complete': True
            }
        else:
            raise Exception(f"Preprocessing plan generation failed: {plan_result.error_message}")
    
    async def _handle_training_state(self) -> Dict[str, Any]:
        """Handle model training state"""
        user_selections = self.current_context.dict().get('user_confirmation_result', {}).get('user_selections', {})
        preprocessing_result = self.current_context.dict().get('preprocessing_result', {})
        
        # Generate training specification
        spec_result = await self._call_tool(
            'training-server',
            'create_training_specification',
            {
                'selected_model': user_selections.get('selected_model'),
                'problem_type': self.current_context.problem_classification.problem_type,
                'data_characteristics': preprocessing_result.get('processed_data_summary', {})
            }
        )
        
        if not spec_result.success:
            raise Exception(f"Training specification failed: {spec_result.error_message}")
        
        # Train model
        training_result = await self._call_tool(
            'training-server',
            'train_model',
            {
                'training_specification': spec_result.result,
                'preprocessed_data_id': preprocessing_result.get('processed_data_id'),
                'validation_strategy': 'cross_validation'
            }
        )
        
        if not training_result.success:
            raise Exception(f"Model training failed: {training_result.error_message}")
        
        return {
            'training_specification': spec_result.result,
            'training_result': training_result.result,
            'model_id': training_result.result.get('model_id'),
            'training_complete': True
        }
    
    async def _handle_evaluation_state(self) -> Dict[str, Any]:
        """Handle model evaluation state"""
        training_result = self.current_context.dict().get('training_result', {})
        
        # Evaluate model
        eval_result = await self._call_tool(
            'training-server',
            'evaluate_model',
            {
                'model_id': training_result.get('model_id'),
                'evaluation_metrics': training_result.get('training_specification', {}).get('evaluation_metrics', []),
                'problem_type': self.current_context.problem_classification.problem_type
            }
        )
        
        if not eval_result.success:
            raise Exception(f"Model evaluation failed: {eval_result.error_message}")
        
        # Generate model report
        report_result = await self._call_tool(
            'training-server',
            'generate_model_report',
            {
                'model_id': training_result.get('model_id'),
                'evaluation_results': eval_result.result,
                'training_info': training_result.get('training_result', {})
            }
        )
        
        return {
            'evaluation_results': eval_result.result,
            'model_report': report_result.result if report_result.success else {},
            'evaluation_complete': True
        }
    
    async def _handle_export_state(self) -> Dict[str, Any]:
        """Handle model export state"""
        # Collect all pipeline artifacts
        pipeline_artifacts = {
            'model_id': self.current_context.dict().get('training_result', {}).get('model_id'),
            'preprocessing_plan': self.current_context.dict().get('preprocessing_result', {}).get('preprocessing_plan'),
            'training_specification': self.current_context.dict().get('training_result', {}).get('training_specification'),
            'evaluation_results': self.current_context.dict().get('evaluation_result', {}),
            'workflow_context': self.current_context.dict()
        }
        
        # Generate complete pipeline code
        code_result = await self._call_tool(
            'export-server',
            'compile_pipeline_code',
            {
                'pipeline_artifacts': pipeline_artifacts,
                'export_format': 'python_script'
            }
        )
        
        # Export model artifacts
        export_result = await self._call_tool(
            'export-server',
            'export_model_package',
            {
                'pipeline_artifacts': pipeline_artifacts,
                'code_package': code_result.result if code_result.success else {},
                'documentation_level': 'comprehensive'
            }
        )
        
        if not export_result.success:
            raise Exception(f"Model export failed: {export_result.error_message}")
        
        return {
            'export_package': export_result.result,
            'pipeline_code': code_result.result if code_result.success else {},
            'export_complete': True
        }
    
    async def _call_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        """Call an MCP tool and return structured result"""
        tool_call = MCPToolCall(
            server_name=server_name,
            tool_name=tool_name,
            parameters=parameters
        )
        
        try:
            start_time = datetime.now()
            
            # Call the tool
            result = await self.mcp_client.call_tool(server_name, tool_name, parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse result content
            parsed_result = {}
            if result.content:
                try:
                    # Assuming text content with JSON
                    content_text = result.content[0].text if result.content else "{}"
                    parsed_result = json.loads(content_text)
                except (json.JSONDecodeError, AttributeError):
                    parsed_result = {"raw_content": str(result.content)}
            
            return MCPToolResult(
                tool_call=tool_call,
                result=parsed_result,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return MCPToolResult(
                tool_call=tool_call,
                result={},
                success=False,
                error_message=str(e),
                execution_time=0
            )
    
    def _update_context_from_result(self, result: Dict[str, Any]) -> None:
        """Update workflow context based on state execution result"""
        current_state = self.current_context.current_state
        
        # Update context based on current state
        if current_state == WorkflowState.PROBLEM_CLASSIFICATION:
            if 'classification_result' in result:
                self.current_context.problem_classification = ProblemClassification(**result['classification_result'])
        
        elif current_state == WorkflowState.DATA_INGESTION:
            self.current_context.user_inputs['ingestion_result'] = result
        
        elif current_state == WorkflowState.EXPLORATORY_ANALYSIS:
            self.current_context.user_inputs['eda_result'] = result
        
        elif current_state == WorkflowState.DECISION_MAKING:
            self.current_context.user_inputs['decision_result'] = result
        
        elif current_state == WorkflowState.USER_CONFIRMATION:
            self.current_context.user_inputs['user_confirmation_result'] = result
        
        elif current_state == WorkflowState.PREPROCESSING:
            self.current_context.user_inputs['preprocessing_result'] = result
        
        elif current_state == WorkflowState.TRAINING:
            self.current_context.user_inputs['training_result'] = result
        
        elif current_state == WorkflowState.EVALUATION:
            self.current_context.user_inputs['evaluation_result'] = result
        
        elif current_state == WorkflowState.EXPORT:
            self.current_context.user_inputs['export_result'] = result
        
        # Update timestamp
        self.current_context.timestamps[f'{current_state}_completed'] = datetime.now()
    
    async def _determine_next_state(self, current_result: Dict[str, Any]) -> WorkflowState:
        """Determine the next workflow state based on current result"""
        current_state = self.current_context.current_state
        
        transition_func = self.state_transitions.get(current_state)
        if transition_func:
            return await transition_func(current_result)
        else:
            return WorkflowState.ERROR
    
    async def _transition_to_state(self, next_state: WorkflowState) -> None:
        """Transition to the next workflow state"""
        previous_state = self.current_context.current_state
        self.current_context.current_state = next_state
        
        self.logger.info(
            "State transition",
            from_state=previous_state,
            to_state=next_state,
            workflow_id=self.current_context.workflow_id
        )
    
    # State transition logic functions
    async def _from_init_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from init state"""
        if result.get('needs_user_input'):
            return WorkflowState.USER_CONFIRMATION
        elif result.get('ready_for_classification'):
            return WorkflowState.PROBLEM_CLASSIFICATION
        else:
            return WorkflowState.ERROR
    
    async def _from_classification_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from classification state"""
        if result.get('needs_clarification'):
            return WorkflowState.USER_CONFIRMATION
        elif result.get('confidence', 0) > 0.3:
            return WorkflowState.DATA_INGESTION
        else:
            return WorkflowState.USER_CONFIRMATION
    
    async def _from_ingestion_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from ingestion state"""
        if result.get('ingestion_success'):
            return WorkflowState.EXPLORATORY_ANALYSIS
        else:
            return WorkflowState.ERROR
    
    async def _from_eda_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from EDA state"""
        if result.get('eda_complete'):
            return WorkflowState.DECISION_MAKING
        else:
            return WorkflowState.ERROR
    
    async def _from_decision_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from decision state"""
        if result.get('needs_user_selection'):
            return WorkflowState.USER_CONFIRMATION
        else:
            return WorkflowState.PREPROCESSING
    
    async def _from_user_confirmation_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from user confirmation state"""
        if result.get('confirmation_complete'):
            # Determine where to go based on what was confirmed
            if 'user_selections' in result:
                return WorkflowState.PREPROCESSING
            elif 'classification_clarified' in result:
                return WorkflowState.DATA_INGESTION
            else:
                return WorkflowState.PROBLEM_CLASSIFICATION
        else:
            return WorkflowState.USER_CONFIRMATION
    
    async def _from_preprocessing_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from preprocessing state"""
        if result.get('preprocessing_complete'):
            return WorkflowState.TRAINING
        else:
            return WorkflowState.ERROR
    
    async def _from_training_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from training state"""
        if result.get('training_complete'):
            return WorkflowState.EVALUATION
        else:
            return WorkflowState.ERROR
    
    async def _from_evaluation_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from evaluation state"""
        if result.get('evaluation_complete'):
            return WorkflowState.EXPORT
        else:
            return WorkflowState.ERROR
    
    async def _from_export_state(self, result: Dict[str, Any]) -> WorkflowState:
        """Transition logic from export state"""
        if result.get('export_complete'):
            return WorkflowState.COMPLETED
        else:
            return WorkflowState.ERROR
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of execution result for history"""
        return {
            'success': result.get('success', True),
            'key_outputs': list(result.keys())[:5],  # First 5 keys
            'has_errors': 'error' in result or 'errors' in result
        }
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        if not self.current_context:
            return {"status": "no_active_workflow"}
        
        return {
            "workflow_id": self.current_context.workflow_id,
            "current_state": self.current_context.current_state,
            "progress_percentage": self._calculate_progress(),
            "elapsed_time": self._calculate_elapsed_time(),
            "last_activity": self.execution_history[-1] if self.execution_history else None,
            "errors": self.current_context.errors
        }
    
    def _calculate_progress(self) -> float:
        """Calculate workflow progress percentage"""
        state_order = [
            WorkflowState.INIT,
            WorkflowState.PROBLEM_CLASSIFICATION,
            WorkflowState.DATA_INGESTION,
            WorkflowState.EXPLORATORY_ANALYSIS,
            WorkflowState.DECISION_MAKING,
            WorkflowState.USER_CONFIRMATION,
            WorkflowState.PREPROCESSING,
            WorkflowState.TRAINING,
            WorkflowState.EVALUATION,
            WorkflowState.EXPORT,
            WorkflowState.COMPLETED
        ]
        
        try:
            current_index = state_order.index(self.current_context.current_state)
            return (current_index / (len(state_order) - 1)) * 100
        except ValueError:
            return 0.0
    
    def _calculate_elapsed_time(self) -> str:
        """Calculate elapsed time since workflow start"""
        if not self.current_context or 'start' not in self.current_context.timestamps:
            return "0s"
        
        start_time = self.current_context.timestamps['start']
        elapsed = datetime.now() - start_time
        
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and disconnect from servers"""
        self.logger.info("Shutting down orchestrator")
        
        if self.mcp_client:
            await self.mcp_client.disconnect_all()
        
        self.logger.info("Orchestrator shutdown complete")"""
Workflow Orchestrator for ML Processor MCP System
Manages the flow between different MCP servers and coordinates the ML pipeline.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog

from mcp.client import Client as MCPClient
from mcp.client.websocket import WebSocketTransport

from ..shared.schemas import (
    WorkflowState, 
    WorkflowContext, 
    MCPToolCall, 
    MCPToolResult,
    ProblemClassification,
    StandardizedData
)
from .parameter_builder import ParameterBuilder
from .resource_manager import ResourceManager
from .context_manager import ContextManager
from .error_handler import ErrorHandler


class WorkflowOrchestrator:
    """Main orchestrator for the ML processing workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.mcp_client = None
        self.parameter_builder = None
        self.resource_manager = None
        self.context_manager = ContextManager()
        self.error_handler = ErrorHandler(self)
        
        # Workflow state
        self.current_context: Optional[WorkflowContext] = None
        self.server_registry = {}
        self.execution_history = []
        
        # State transition rules
        self.state_transitions = self._setup_state_transitions()
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and connect to MCP servers"""
        self.logger.info("Initializing Workflow Orchestrator")
        
        # Initialize MCP client
        await