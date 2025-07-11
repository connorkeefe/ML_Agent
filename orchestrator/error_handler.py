"""
Error Handler for ML Processor Workflow
Handles errors with appropriate recovery strategies.
"""

from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class ParameterValidationError(Exception):
    """Error in parameter validation"""
    pass


class ResourceNotFoundError(Exception):
    """Required resource not found"""
    pass


class AgentUnavailableError(Exception):
    """Agent/server is unavailable"""
    pass


class ErrorHandler:
    """Handles errors with recovery strategies"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 3
        
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors with appropriate recovery strategies"""
        error_type = type(error).__name__
        workflow_id = context.get('workflow_id', 'unknown')
        
        logger.error("Handling error", 
                    error_type=error_type, 
                    error_message=str(error),
                    workflow_id=workflow_id)
        
        # Track error counts
        error_key = f"{workflow_id}:{error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Check if we've exceeded retry limit
        if self.error_counts[error_key] > self.max_retries:
            return await self._handle_max_retries_exceeded(error, context)
        
        # Route to specific error handlers
        if isinstance(error, ParameterValidationError):
            return await self._recover_parameters(error, context)
        elif isinstance(error, ResourceNotFoundError):
            return await self._find_alternative_resources(error, context)
        elif isinstance(error, AgentUnavailableError):
            return await self._handle_agent_failure(error, context)
        else:
            return await self._general_error_recovery(error, context)
    
    async def _recover_parameters(self, error: ParameterValidationError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover missing parameters"""
        logger.info("Attempting parameter recovery")
        
        try:
            # Try to rebuild parameters with more context
            if hasattr(self.orchestrator, 'parameter_builder'):
                # This would need more sophisticated logic
                pass
            
            return {
                'recovery_attempted': True,
                'recovery_successful': False,
                'error_type': 'parameter_validation',
                'suggestion': 'Please provide missing required parameters'
            }
        except Exception as e:
            logger.error(f"Parameter recovery failed: {e}")
            return {'recovery_attempted': True, 'recovery_successful': False}
    
    async def _find_alternative_resources(self, error: ResourceNotFoundError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to find alternative resources"""
        logger.info("Searching for alternative resources")
        
        try:
            # Search for similar resources
            if hasattr(self.orchestrator, 'resource_manager'):
                # This would search for alternative resources
                pass
            
            return {
                'recovery_attempted': True,
                'recovery_successful': False,
                'error_type': 'resource_not_found',
                'suggestion': 'Required resource unavailable, using fallback behavior'
            }
        except Exception as e:
            logger.error(f"Resource recovery failed: {e}")
            return {'recovery_attempted': True, 'recovery_successful': False}
    
    async def _handle_agent_failure(self, error: AgentUnavailableError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to restart agent or use backup"""
        logger.info("Handling agent failure")
        
        try:
            # Attempt to reconnect to agent
            # This would involve checking server status and attempting reconnection
            
            return {
                'recovery_attempted': True,
                'recovery_successful': False,
                'error_type': 'agent_unavailable',
                'suggestion': 'Please check that all MCP servers are running'
            }
        except Exception as e:
            logger.error(f"Agent recovery failed: {e}")
            return {'recovery_attempted': True, 'recovery_successful': False}
    
    async def _general_error_recovery(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """General error handling"""
        logger.info("Applying general error recovery")
        
        # Determine if error is recoverable
        recoverable_errors = [
            'ConnectionError',
            'TimeoutError',
            'TemporaryError'
        ]
        
        error_type = type(error).__name__
        is_recoverable = error_type in recoverable_errors
        
        if is_recoverable:
            # Wait and retry
            import asyncio
            await asyncio.sleep(2)  # Brief delay before retry
            
            return {
                'recovery_attempted': True,
                'recovery_successful': True,
                'action': 'retry_after_delay',
                'error_type': error_type
            }
        else:
            return {
                'recovery_attempted': True,
                'recovery_successful': False,
                'error_type': error_type,
                'fatal': True,
                'message': f"Unrecoverable error: {str(error)}"
            }
    
    async def _handle_max_retries_exceeded(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case where max retries have been exceeded"""
        logger.error("Maximum retries exceeded")
        
        return {
            'recovery_attempted': True,
            'recovery_successful': False,
            'max_retries_exceeded': True,
            'error_type': type(error).__name__,
            'fatal': True,
            'message': f"Failed after {self.max_retries} retries: {str(error)}"
        }
    
    def reset_error_counts(self, workflow_id: str) -> None:
        """Reset error counts for a workflow"""
        keys_to_remove = [k for k in self.error_counts.keys() if k.startswith(f"{workflow_id}:")]
        for key in keys_to_remove:
            del self.error_counts[key]
        
        logger.info("Error counts reset", workflow_id=workflow_id)