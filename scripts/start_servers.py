#!/usr/bin/env python3
"""
Start all MCP servers for the ML Processor system.
This script launches each server in its own process.
"""

import asyncio
import json
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Dict, List
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Manages the lifecycle of MCP servers"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/mcp_config.json"
        self.processes: Dict[str, subprocess.Popen] = {}
        self.config = {}
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_config(self) -> None:
        """Load server configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def start_all_servers(self) -> None:
        """Start all configured servers"""
        servers = self.config.get("servers", {})
        
        if not servers:
            logger.warning("No servers configured")
            return
        
        logger.info(f"Starting {len(servers)} servers...")
        
        for server_name, server_config in servers.items():
            try:
                self.start_server(server_name, server_config)
                time.sleep(1)  # Small delay between server starts
            except Exception as e:
                logger.error(f"Failed to start server {server_name}: {e}")
    
    def start_server(self, server_name: str, server_config: Dict) -> None:
        """Start a single server"""
        logger.info(f"Starting server: {server_name}")
        
        # Build command
        python_path = sys.executable
        module_path = server_config.get("module_path")
        
        if not module_path:
            logger.error(f"No module_path specified for server {server_name}")
            return
        
        # Build command arguments
        cmd = [
            python_path, "-m", module_path,
            "--host", server_config.get("host", "localhost"),
            "--port", str(server_config.get("port", 8000)),
            "--transport", server_config.get("transport", "websocket"),
            "--log-level", server_config.get("log_level", "INFO")
        ]
        
        # Add environment variables if specified
        env = server_config.get("environment", {})
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **env} if env else None,
                text=True
            )
            
            self.processes[server_name] = process
            logger.info(f"Server {server_name} started with PID {process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
    
    def check_server_health(self) -> Dict[str, bool]:
        """Check health of all running servers"""
        health_status = {}
        
        for server_name, process in self.processes.items():
            try:
                # Check if process is still running
                return_code = process.poll()
                if return_code is None:
                    health_status[server_name] = True
                else:
                    health_status[server_name] = False
                    logger.warning(f"Server {server_name} has stopped (return code: {return_code})")
            except Exception as e:
                health_status[server_name] = False
                logger.error(f"Error checking health of server {server_name}: {e}")
        
        return health_status
    
    def restart_server(self, server_name: str) -> None:
        """Restart a specific server"""
        logger.info(f"Restarting server: {server_name}")
        
        # Stop the server first
        self.stop_server(server_name)
        
        # Wait a moment
        time.sleep(2)
        
        # Start it again
        server_config = self.config["servers"].get(server_name)
        if server_config:
            self.start_server(server_name, server_config)
        else:
            logger.error(f"No configuration found for server {server_name}")
    
    def stop_server(self, server_name: str) -> None:
        """Stop a specific server"""
        if server_name not in self.processes:
            logger.warning(f"Server {server_name} is not running")
            return
        
        process = self.processes[server_name]
        logger.info(f"Stopping server: {server_name}")
        
        try:
            # Send SIGTERM
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Forcefully killing server {server_name}")
                process.kill()
                process.wait()
            
            del self.processes[server_name]
            logger.info(f"Server {server_name} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server {server_name}: {e}")
    
    def stop_all_servers(self) -> None:
        """Stop all running servers"""
        logger.info("Stopping all servers...")
        
        server_names = list(self.processes.keys())
        for server_name in server_names:
            self.stop_server(server_name)
        
        logger.info("All servers stopped")
    
    def monitor_servers(self) -> None:
        """Monitor server health and restart if needed"""
        logger.info("Starting server monitoring...")
        
        while self.running:
            try:
                health_status = self.check_server_health()
                
                # Restart any failed servers
                for server_name, is_healthy in health_status.items():
                    if not is_healthy:
                        logger.warning(f"Server {server_name} is unhealthy, restarting...")
                        self.restart_server(server_name)
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in server monitoring: {e}")
                time.sleep(10)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.stop_all_servers()
        sys.exit(0)
    
    def get_status(self) -> Dict:
        """Get status of all servers"""
        health_status = self.check_server_health()
        
        status = {
            "total_servers": len(self.config.get("servers", {})),
            "running_servers": sum(health_status.values()),
            "server_status": {}
        }
        
        for server_name, server_config in self.config.get("servers", {}).items():
            is_running = health_status.get(server_name, False)
            process = self.processes.get(server_name)
            
            status["server_status"][server_name] = {
                "running": is_running,
                "pid": process.pid if process and is_running else None,
                "host": server_config.get("host", "localhost"),
                "port": server_config.get("port", 8000),
                "transport": server_config.get("transport", "websocket")
            }
        
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Servers Manager")
    parser.add_argument("--config", "-c", help="Configuration file path", 
                       default="config/mcp_config.json")
    parser.add_argument("--monitor", "-m", action="store_true", 
                       help="Enable server monitoring and auto-restart")
    parser.add_argument("--status", "-s", action="store_true", 
                       help="Show server status and exit")
    parser.add_argument("--stop", action="store_true", 
                       help="Stop all servers and exit")
    
    args = parser.parse_args()
    
    # Create server manager
    manager = ServerManager(args.config)
    manager.load_config()
    
    if args.status:
        # Show status and exit
        status = manager.get_status()
        print("\n=== MCP Servers Status ===")
        print(f"Total servers configured: {status['total_servers']}")
        print(f"Running servers: {status['running_servers']}")
        print("\nServer Details:")
        for name, info in status["server_status"].items():
            status_str = "RUNNING" if info["running"] else "STOPPED"
            pid_str = f" (PID: {info['pid']})" if info["pid"] else ""
            print(f"  {name}: {status_str}{pid_str} - {info['host']}:{info['port']} ({info['transport']})")
        return
    
    if args.stop:
        # Stop all servers and exit
        manager.stop_all_servers()
        return
    
    # Start all servers
    manager.start_all_servers()
    
    if args.monitor:
        # Monitor servers
        try:
            manager.monitor_servers()
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted")
    else:
        # Just start and show status
        print("\n=== Servers Started ===")
        status = manager.get_status()
        for name, info in status["server_status"].items():
            if info["running"]:
                print(f"✓ {name} running on {info['host']}:{info['port']} (PID: {info['pid']})")
            else:
                print(f"✗ {name} failed to start")
        
        print("\nServers are running. Press Ctrl+C to stop all servers.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all_servers()


if __name__ == "__main__":
    import os
    main()