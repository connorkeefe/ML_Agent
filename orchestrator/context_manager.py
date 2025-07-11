"""
Context Manager for ML Processor Workflow
Manages workflow context and state persistence.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class ContextManager:
    """Manages workflow context and state"""
    
    def __init__(self):
        self.context_store: Dict[str, Dict[str, Any]] = {}
        self.state_history: List[Dict[str, Any]] = []
        
    def update_context(self, workflow_id: str, key: str, value: Any, metadata: Dict = None) -> None:
        """Update context with new information"""
        if workflow_id not in self.context_store:
            self.context_store[workflow_id] = {}
        
        self.context_store[workflow_id][key] = {
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'workflow_id': workflow_id
        }
        
        logger.info("Context updated", workflow_id=workflow_id, key=key)
    
    def get_context(self, workflow_id: str, key: str = None) -> Any:
        """Get context value(s)"""
        if workflow_id not in self.context_store:
            return None
        
        if key is None:
            # Return all context for workflow
            return {k: v['value'] for k, v in self.context_store[workflow_id].items()}
        
        context_item = self.context_store[workflow_id].get(key)
        return context_item['value'] if context_item else None
    
    def get_context_for_agent(self, workflow_id: str, agent_name: str) -> Dict[str, Any]:
        """Get relevant context for specific agent"""
        
        # Define what context each agent needs
        agent_context_rules = {
            'ml-classifier-server': ['user_input', 'data_preview'],
            'data-ingestion-server': ['user_input', 'file_info'],
            'eda-server': ['ingestion_result', 'problem_type', 'user_preferences'],
            'ml-decision-server': ['eda_result', 'problem_type', 'user_preferences'],
            'preprocessing-server': ['decision_result', 'eda_result', 'data_info'],
            'training-server': ['preprocessing_result', 'model_config', 'problem_type'],
            'export-server': ['training_result', 'preprocessing_pipeline', 'workflow_summary']
        }
        
        if workflow_id not in self.context_store:
            return {}
        
        relevant_keys = agent_context_rules.get(agent_name, [])
        context = {}
        
        for key in relevant_keys:
            value = self.get_context(workflow_id, key)
            if value is not None:
                context[key] = value
        
        # Always include basic workflow info
        context['workflow_id'] = workflow_id
        context['timestamp'] = datetime.now()
        
        return context
    
    def record_state_transition(self, workflow_id: str, from_state: str, to_state: str, 
                              result: Dict[str, Any]) -> None:
        """Record a state transition in history"""
        transition_record = {
            'workflow_id': workflow_id,
            'from_state': from_state,
            'to_state': to_state,
            'timestamp': datetime.now(),
            'result_summary': self._summarize_result(result)
        }
        
        self.state_history.append(transition_record)
        logger.info("State transition recorded", 
                   workflow_id=workflow_id, 
                   transition=f"{from_state} -> {to_state}")
    
    def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get state transition history for a workflow"""
        return [record for record in self.state_history 
                if record['workflow_id'] == workflow_id]
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of execution result"""
        return {
            'success': result.get('success', True),
            'key_outputs': list(result.keys())[:5],  # First 5 keys
            'has_errors': 'error' in result or 'errors' in result,
            'data_size': len(str(result))
        }
    
    def clear_workflow_context(self, workflow_id: str) -> None:
        """Clear context for a completed workflow"""
        if workflow_id in self.context_store:
            del self.context_store[workflow_id]
            logger.info("Workflow context cleared", workflow_id=workflow_id)
    
    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs"""
        return list(self.context_store.keys())