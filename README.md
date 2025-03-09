# Mem0llama

A specialized fork of [Mem0](https://mem0.ai) optimized for small local Large Language Models (LLMs) running through Ollama.

## Overview

Mem0llama enhances the compatibility of Mem0's graph memory system with small local LLMs while preserving its integration with Qdrant (for vector storage/RAG) and Neo4j (for graph relationships). This implementation ensures structured and predictable outputs from LLMs using Ollama's format argument and Pydantic models.

## Key Features

- **Structured Output Support**: Implemented Pydantic models to standardize LLM outputs for reliable entity and relationship extraction
- **LLM Formatting Utilities**: Created utilities for structured prompts and response parsing with robust error handling
- **Ollama Integration**: Optimized for local LLMs running through Ollama with format parameter configuration
- **Neo4j Community Edition Support**: Enhanced compatibility with Neo4j Community Edition
- **Preserved Functionality**: Maintains all existing Mem0 features including graph memory, search, and retrieval mechanisms

## Components

### Core Enhancements

1. **Pydantic Models** (`models.py`)
   - Standardized models for entities, relationships, and memory operations
   - Ensures consistent data structures for LLM outputs

2. **LLM Formatting Utilities** (`llm_formatter.py`)
   - Functions to create structured prompts for various operations
   - Response parsing with comprehensive error handling
   - Ollama format parameter configuration

3. **Graph Memory Improvements** (`graph_memory.py`)
   - Modified node retrieval and relationship establishment
   - Enhanced entity and relationship extraction
   - Added LLM configuration adaptation

4. **Bug Fixes**
   - Fixed "relatationship" typo in multiple files
   - Improved error handling for inconsistent LLM responses

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama (for local LLMs)
- Neo4j (Community Edition compatible)
- Qdrant (for vector storage)

### Installation

#### Docker servers

##### qdrant

```bash
cd servers/qdrant
./qdrant_run_docker.bat
```

##### neo4j

```bash
cd servers/neo4j
./neo4j_run_docker.bat
```

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with the following variables:

```
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=
LLM_MODEL=llama3
EMBEDDER_MODEL=nomic-embed-text
QDRANT_HOST=localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Basic Usage

```python
from mem0 import Memory

# Initialize Mem0 client
memory_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1024,
            "api_key": qdrant_api_key,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "ollama_base_url": "http://localhost:11434",
            "model": "llama3",
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
    },
}

memory = Memory.from_config(memory_config)

# Add memories
messages = [
    {"role": "user", "content": "I like to program in Python"},
    {"role": "assistant", "content": "That's great! Python is a versatile language."}
]
memory.add(messages, user_id="user123")

# Search memories
results = memory.search(query="What programming languages do I know?", user_id="user123")
```

## Testing

Run the test script to validate the structured output capabilities:

```bash
python ./mem0llama_structured_output.py
```

For a more interactive experience, try the terminal chatbot:

```bash
python ./mem0llama_chat_cli.py
```

Add the `--debug` flag to see detailed memory and relation information:

```bash
python ./mem0llama_chat_cli.py --debug
```

## License

This project is licensed under the same terms as the original Mem0 project.

## Acknowledgements

- [Mem0](https://mem0.ai) - The original memory layer for AI agents
- [Ollama](https://ollama.ai) - For running local LLMs
- [Neo4j](https://neo4j.com) - Graph database for relationship storage
- [Qdrant](https://qdrant.tech) - Vector database for RAG