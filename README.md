# ML Processor MCP System

A distributed machine learning processing system built on the Model Context Protocol (MCP). This system uses specialized agents to handle different aspects of the ML pipeline, coordinated by a central orchestrator that Claude can interact with.

## 🏗️ Architecture Overview

The system consists of 7 specialized MCP servers, each handling a specific aspect of ML processing:

1. **Classification Server** (`port 8001`) - Problem type identification
2. **Data Ingestion Server** (`port 8002`) - Data loading and standardization  
3. **EDA Server** (`port 8003`) - Exploratory data analysis
4. **Decision Server** (`port 8004`) - Feature and model recommendations
5. **Preprocessing Server** (`port 8005`) - Data preprocessing
6. **Training Server** (`port 8006`) - Model training and evaluation
7. **Export Server** (`port 8007`) - Pipeline export and packaging

All coordinated by the **Workflow Orchestrator** (`port 8000`).

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ml_processor_mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Setup

```bash
# Create necessary directories
mkdir -p logs data/workflows data/models

# Copy environment template
cp .env.example .env

# Edit configuration if needed
nano config/mcp_config.json
```

### 3. Start the System

```bash
# Start all MCP servers
python scripts/start_servers.py --monitor

# In another terminal, run the orchestrator
python orchestrator/main.py --interactive
```

## 📋 Usage

### For Claude Integration

The main interface for Claude is the `claude_interface` function in `orchestrator/main.py`:

```python
from orchestrator.main import claude_interface

# Process an ML workflow
result = claude_interface(
    prompt="I want to predict house prices based on features like size, location, etc.",
    filepath="/path/to/housing_data.csv"
)
```

### Command Line Usage

```bash
# Interactive mode
python orchestrator/main.py --interactive

# Single request
python orchestrator/main.py --prompt "Classify customer segments" --filepath "data.csv"

# Check system status
python orchestrator/main.py --status

# Server management
python scripts/start_servers.py --status  # Check server status
python scripts/start_servers.py --stop    # Stop all servers
```

## 🔧 Configuration

### Server Configuration (`config/mcp_config.json`)

```json
{
  "servers": {
    "ml-classifier-server": {
      "host": "localhost",
      "port": 8001,
      "transport": "websocket"
    }
    // ... other servers
  },
  "data_limits": {
    "max_file_size_mb": 1000,
    "max_rows": 1000000
  }
}
```

### Supported Data Formats

- **CSV** (`.csv`, `.tsv`) - Most common format
- **Excel** (`.xlsx`, `.xls`) - Spreadsheet files
- **JSON** (`.json`) - Structured data
- **Parquet** (`.parquet`) - Columnar format
- **Feather** (`.feather`) - Fast binary format

## 🎯 Workflow Example

Here's what happens when you process an ML problem:

1. **Problem Classification**: "I want to predict customer churn" → Supervised Classification
2. **Data Ingestion**: Load `customer_data.csv` → Standardize format
3. **EDA**: Analyze 10,000 rows, 15 features → Identify target variable
4. **Decision Making**: Recommend Random Forest → User confirms
5. **Preprocessing**: Handle missing values → Scale features
6. **Training**: Train model → Achieve 85% accuracy
7. **Export**: Generate complete pipeline code + documentation

## 🔍 System Components

### MCP Servers

Each server is a standalone MCP server with:
- **Tools**: Functions that can be called (e.g., `classify_problem`, `load_data`)
- **Resources**: Knowledge bases (e.g., model catalog, preprocessing recipes)
- **State Management**: Independent operation with shared context

### Orchestrator

The orchestrator manages:
- **Workflow State Machine**: Transitions between processing stages
- **Parameter Discovery**: Automatically builds tool parameters from context
- **Resource Management**: Provides relevant knowledge to each agent
- **Error Handling**: Graceful recovery and user feedback

### Data Flow

```
User Input → Classification → Ingestion → EDA → Decision → 
Preprocessing → Training → Evaluation → Export → Results
```

## 🛠️ Development

### Adding New Servers

1. Create server directory: `mcp_servers/new_server/`
2. Implement server class extending `BaseMLServer`
3. Define tools, resources, and handlers
4. Add to configuration: `config/mcp_config.json`
5. Update orchestrator workflow logic

### Server Template

```python
class NewServer(BaseMLServer):
    def get_tool_definitions(self) -> List[Tool]:
        return [
            Tool(name="new_tool", description="...", inputSchema={...})
        ]
    
    async def handle_new_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation
        return {"result": "success"}
```

### Testing

```bash
# Run tests
python -m pytest tests/

# Test specific server
python -m pytest tests/test_servers/test_classification_server.py

# Integration tests
python scripts/run_tests.py --integration
```

## 📊 Monitoring

### Server Health

```bash
# Check all servers
python scripts/start_servers.py --status

# Monitor with auto-restart
python scripts/start_servers.py --monitor
```

### Workflow Monitoring

```bash
# Real-time status
python orchestrator/main.py --status

# Logs
tail -f logs/ml_processor.log
```

## 🔒 Security Considerations

- **Local Files Only**: System only processes local files by default
- **File Type Validation**: Only allowed extensions are processed
- **Size Limits**: Configurable limits on file size and memory usage
- **Sandboxed Execution**: Each server runs independently

## 🐛 Troubleshooting

### Common Issues

1. **Server Connection Failed**
   ```bash
   # Check if servers are running
   python scripts/start_servers.py --status
   
   # Restart specific server
   python scripts/start_servers.py --restart ml-classifier-server
   ```

2. **Memory Issues with Large Data**
   ```python
   # Adjust limits in config
   "data_limits": {
     "max_file_size_mb": 500,
     "max_memory_usage_gb": 4
   }
   ```

3. **Port Conflicts**
   ```json
   // Change ports in config/mcp_config.json
   "port": 8001  // Use different port
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python orchestrator/main.py --interactive
```

## 🤝 Integration with Claude

Claude can interact with this system through several methods:

### Direct Function Call
```python
# In Claude's environment
result = claude_interface(
    "Predict sales based on historical data", 
    "sales_data.csv"
)
```

### Structured Workflow
```python
# Step-by-step interaction
orchestrator = WorkflowOrchestrator(config)
await orchestrator.initialize()

context = await orchestrator.start_workflow({
    'prompt': 'Customer segmentation analysis',
    'filepath': 'customer_data.csv'
})
```

## 📈 Performance

### Benchmarks
- **Small datasets** (< 1K rows): ~30 seconds end-to-end
- **Medium datasets** (1K-100K rows): ~2-5 minutes  
- **Large datasets** (100K-1M rows): ~10-30 minutes

### Optimization Tips
- Use Parquet format for large datasets
- Enable data sampling for initial exploration
- Configure appropriate memory limits
- Use SSD storage for model artifacts

## 🗺️ Roadmap

### Planned Features
- [ ] Deep learning support expansion
- [ ] Real-time model serving
- [ ] Database connectivity
- [ ] Distributed processing
- [ ] Advanced AutoML capabilities
- [ ] Model monitoring and drift detection

### Recent Updates
- ✅ Initial MCP architecture implementation
- ✅ Complete workflow orchestration
- ✅ Multi-format data ingestion
- ✅ Automated feature engineering
- ✅ Model export and packaging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in `logs/ml_processor.log`  
3. Open an issue with system status and error details

---

**Happy ML Processing! 🤖✨**