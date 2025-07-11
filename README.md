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

