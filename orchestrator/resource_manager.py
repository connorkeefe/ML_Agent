"""
Resource Manager for MCP System
Manages resources across all servers and provides intelligent resource querying.
"""

from typing import Dict, List, Any, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class ResourceManager:
    """Manages resources across MCP servers"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.resource_cache: Dict[str, Any] = {}
        self.resource_index: Dict[str, Dict[str, Any]] = {}
        
    async def initialize_resources(self) -> None:
        """Discover and cache all available resources"""
        logger.info("Initializing resource manager")
        
        try:
            servers = await self.mcp_client.list_servers()
            
            for server in servers:
                await self._load_server_resources(server.name)
                
            logger.info("Resource initialization complete", 
                       total_resources=len(self.resource_cache))
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
    
    async def _load_server_resources(self, server_name: str) -> None:
        """Load all resources from a specific server"""
        try:
            resources = await self.mcp_client.list_resources(server_name)
            
            for resource in resources:
                resource_key = f"{server_name}:{resource.name}"
                
                # Get resource content
                content = await self.mcp_client.read_resource(
                    server_name, resource.uri
                )
                
                # Cache the content
                self.resource_cache[resource_key] = content
                
                # Build searchable index
                self.resource_index[resource.name] = {
                    'server': server_name,
                    'description': resource.description,
                    'mime_type': resource.mimeType,
                    'uri': resource.uri,
                    'keywords': self._extract_keywords(content)
                }
                
                logger.info(f"Loaded resource: {resource_key}")
                
        except Exception as e:
            logger.error(f"Failed to load resources from {server_name}: {e}")
    
    def _extract_keywords(self, content: Any) -> List[str]:
        """Extract keywords from resource content for indexing"""
        keywords = []
        
        if hasattr(content, 'text'):
            text = content.text.lower()
            # Simple keyword extraction
            words = text.split()
            keywords = [w for w in words if len(w) > 3 and w.isalpha()]
        
        return list(set(keywords))  # Remove duplicates
    
    async def query_resources(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Query resources based on current context"""
        relevant_resources = {}
        
        # Problem-type specific resources
        if 'problem_type' in query_context:
            problem_resources = self._get_problem_specific_resources(
                query_context['problem_type']
            )
            relevant_resources.update(problem_resources)
        
        # Stage-specific resources
        if 'stage' in query_context:
            stage_resources = self._get_stage_specific_resources(
                query_context['stage']
            )
            relevant_resources.update(stage_resources)
        
        return relevant_resources
    
    def _get_problem_specific_resources(self, problem_type: str) -> Dict[str, Any]:
        """Get resources specific to the ML problem type"""
        resource_mappings = {
            'supervised_classification': [
                'ml-classifier-server:ml_problem_definitions',
                'ml-decision-server:model_catalog',
                'training-server:evaluation_metrics'
            ],
            'supervised_regression': [
                'ml-classifier-server:ml_problem_definitions',
                'ml-decision-server:model_catalog',
                'training-server:evaluation_metrics'
            ],
            'unsupervised_clustering': [
                'ml-classifier-server:ml_problem_definitions',
                'ml-decision-server:model_catalog',
                'eda-server:visualization_configs'
            ]
        }
        
        relevant_keys = resource_mappings.get(problem_type, [])
        return {key: self.resource_cache.get(key) for key in relevant_keys 
                if key in self.resource_cache}
    
    def _get_stage_specific_resources(self, stage: str) -> Dict[str, Any]:
        """Get resources specific to the workflow stage"""
        stage_mappings = {
            'classification': [
                'ml-classifier-server:ml_problem_definitions',
                'ml-classifier-server:capability_matrix'
            ],
            'ingestion': [
                'data-ingestion-server:supported_formats',
                'data-ingestion-server:standardized_schema'
            ],
            'eda': [
                'eda-server:analysis_templates',
                'eda-server:visualization_configs'
            ],
            'decision': [
                'ml-decision-server:model_catalog',
                'ml-decision-server:feature_engineering_patterns'
            ],
            'preprocessing': [
                'preprocessing-server:preprocessing_library',
                'preprocessing-server:transformation_catalog'
            ],
            'training': [
                'training-server:training_templates',
                'training-server:evaluation_metrics'
            ],
            'export': [
                'export-server:export_templates',
                'export-server:documentation_templates'
            ]
        }
        
        relevant_keys = stage_mappings.get(stage, [])
        return {key: self.resource_cache.get(key) for key in relevant_keys 
                if key in self.resource_cache}
    
    def get_resource(self, resource_key: str) -> Any:
        """Get a specific resource by key"""
        return self.resource_cache.get(resource_key)
    
    def search_resources(self, keywords: List[str]) -> List[str]:
        """Search resources by keywords"""
        matching_resources = []
        
        for resource_name, resource_info in self.resource_index.items():
            resource_keywords = resource_info.get('keywords', [])
            
            # Check for keyword matches
            if any(keyword.lower() in resource_keywords for keyword in keywords):
                resource_key = f"{resource_info['server']}:{resource_name}"
                matching_resources.append(resource_key)
        
        return matching_resources
    
    def get_server_resources(self, server_name: str) -> Dict[str, Any]:
        """Get all resources for a specific server"""
        server_resources = {}
        
        for resource_key, content in self.resource_cache.items():
            if resource_key.startswith(f"{server_name}:"):
                server_resources[resource_key] = content
        
        return server_resources