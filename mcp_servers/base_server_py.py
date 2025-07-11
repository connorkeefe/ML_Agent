"""
Base MCP Server class for ML Processor agents.
Provides common functionality for all specialized servers.
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    CallToolRequest,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
)
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class ServerConfig(BaseModel):
    """Configuration for MCP server"""
    name: str
    description: str
    version: str = "1.0.0"
    port: int = 8000
    host: str = "localhost"
    transport: str = "websocket"  # websocket, http, stdio
    log_level: str = "INFO"
    resource_path: Optional[str] = None


class BaseMLServer(ABC):
    """Base class for all ML processor MCP servers"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.server = Server(config.name)
        self.resources: Dict[str, Any] = {}
        self.tools: Dict[str, Tool] = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        # Initialize server
        self._setup_server()
        
    def _setup_server(self):
        """Set up the MCP server with handlers"""
        
        # Register handlers
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List all available resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri=f"resource://{self.config.name}/{name}",
                        name=name,
                        description=resource.get("description", ""),
                        mimeType=resource.get("mimeType", "application/json")
                    )
                    for name, resource in self.resources.items()
                ]
            )
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource"""
            # Extract resource name from URI
            resource_name = uri.split("/")[-1]
            
            if resource_name not in self.resources:
                raise ValueError(f"Resource '{resource_name}' not found")
            
            resource_data = self.resources[resource_name]
            
            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text=json.dumps(resource_data.get("content", {}), indent=2)
                    )
                ]
            )
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List all available tools"""
            return ListToolsResult(
                tools=list(self.tools.values())
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Call a specific tool"""
            if name not in self.tools:
                raise ValueError(f"Tool '{name}' not found")
            
            try:
                # Log tool call
                self.logger.info(
                    "Tool called",
                    tool_name=name,
                    arguments=arguments,
                    timestamp=datetime.now().isoformat()
                )
                
                # Execute tool
                result = await self._execute_tool(name, arguments)
                
                # Log result
                self.logger.info(
                    "Tool completed",
                    tool_name=name,
                    success=True,
                    timestamp=datetime.now().isoformat()
                )
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(result, indent=2, default=str)
                        )
                    ]
                )
                
            except Exception as e:
                # Log error
                self.logger.error(
                    "Tool failed",
                    tool_name=name,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": str(e),
                                "tool_name": name,
                                "timestamp": datetime.now().isoformat()
                            }, indent=2)
                        )
                    ]
                )
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        # Get tool handler
        handler_name = f"handle_{tool_name}"
        
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            return await handler(arguments)
        else:
            raise ValueError(f"No handler found for tool '{tool_name}'")
    
    def load_resources(self, resource_path: Optional[str] = None) -> None:
        """Load resources from JSON files"""
        if resource_path is None:
            resource_path = self.config.resource_path
        
        if resource_path is None:
            # Default to resources directory relative to server file
            base_path = Path(__file__).parent / self.__class__.__name__.lower().replace("server", "") / "resources"
        else:
            base_path = Path(resource_path)
        
        if not base_path.exists():
            self.logger.warning(f"Resource path does not exist: {base_path}")
            return
        
        # Load all JSON files in resources directory
        for json_file in base_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                resource_name = json_file.stem
                self.resources[resource_name] = {
                    "content": content,
                    "description": content.get("description", f"Resource: {resource_name}"),
                    "mimeType": "application/json",
                    "loaded_at": datetime.now().isoformat()
                }
                
                self.logger.info(f"Loaded resource: {resource_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load resource {json_file}: {e}")
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the server"""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, name: str, content: Any, description: str = "", mime_type: str = "application/json") -> None:
        """Register a resource with the server"""
        self.resources[name] = {
            "content": content,
            "description": description,
            "mimeType": mime_type,
            "loaded_at": datetime.now().isoformat()
        }
        self.logger.info(f"Registered resource: {name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the server (load resources, register tools, etc.)"""
        pass
    
    @abstractmethod
    def get_tool_definitions(self) -> List[Tool]:
        """Return list of tools provided by this server"""
        pass
    
    async def start(self) -> None:
        """Start the MCP server"""
        await self.initialize()
        
        # Load resources
        self.load_resources()
        
        # Register tools
        for tool in self.get_tool_definitions():
            self.register_tool(tool)
        
        self.logger.info(
            f"Starting {self.config.name} server",
            host=self.config.host,
            port=self.config.port,
            transport=self.config.transport
        )
        
        # Start server based on transport type
        if self.config.transport == "websocket":
            await self._start_websocket_server()
        elif self.config.transport == "stdio":
            await self._start_stdio_server()
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server"""
        from mcp.server.websocket import WebSocketServer
        
        websocket_server = WebSocketServer(self.server)
        await websocket_server.start(host=self.config.host, port=self.config.port)
    
    async def _start_stdio_server(self) -> None:
        """Start stdio server"""
        from mcp.server.stdio import StdioServer
        
        stdio_server = StdioServer(self.server)
        await stdio_server.start()
    
    def validate_parameters(self, parameters: Dict[str, Any], required_params: List[str]) -> None:
        """Validate that required parameters are present"""
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
    
    def get_resource_content(self, resource_name: str) -> Any:
        """Get content of a specific resource"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource '{resource_name}' not found")
        
        return self.resources[resource_name]["content"]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the server"""
        return {
            "status": "healthy",
            "server_name": self.config.name,
            "version": self.config.version,
            "uptime": datetime.now().isoformat(),
            "resources_loaded": len(self.resources),
            "tools_registered": len(self.tools)
        }