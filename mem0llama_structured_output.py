"""
Test script for validating the enhanced structured output capabilities of Mem0llama.
This script tests the integration of Pydantic models and LLM formatting utilities
with small local LLMs using Ollama.
"""

import os
import asyncio
from dotenv import load_dotenv
from rich import print
from mem0llama import Memory
from mem0llama.memory.models import EntityExtraction, RelationshipExtraction
from mem0llama.memory.llm_formatter import parse_llm_response, create_entity_extraction_prompt, create_relationship_extraction_prompt

load_dotenv(override=True)

async def test_entity_extraction():
    """Test entity extraction with structured output."""
    print("\n[bold blue]Testing Entity Extraction[/bold blue]")
    
    # Retrieve environment variables
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    
    # Initialize Mem0llama client with minimal config for testing
    memory_config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "ollama_base_url": base_url,
                "api_key": api_key,
                "model": model_name.replace("ollama/", ""),
                "temperature": 0.1,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL", "nomic-embed-text").replace("ollama/", ""),
                "ollama_base_url": base_url,
                "api_key": api_key,
            },
        },
        "history_db_path": "./data/test_structured_output.db",
    }
    
    memory = Memory.from_config(memory_config)
    
    # Test text for entity extraction
    test_text = "John works at Microsoft as a software engineer. He uses Python and JavaScript to build applications."
    
    # Create a structured prompt
    prompt = create_entity_extraction_prompt(test_text, "test_user")
    
    # Generate response using the LLM with Pydantic model schema
    response = memory.llm.generate_response(
        messages=[
            {"role": "system", "content": "You are an AI assistant that extracts entities and their types from text."},
            {"role": "user", "content": prompt}
        ],
        response_format=EntityExtraction.model_json_schema()
    )
    
    print(f"[bold]Raw LLM Response:[/bold]\n{response}")
    
    # Parse the response
    try:
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = response
            
        extraction_result = parse_llm_response(content, EntityExtraction)
        print("\n[bold green]Parsed Entities:[/bold green]")
        for entity in extraction_result.entities:
            print(f"  - Entity: {entity.entity}, Type: {entity.entity_type}")
    except Exception as e:
        print(f"[bold red]Error parsing response:[/bold red] {e}")


async def test_relationship_extraction():
    """Test relationship extraction with structured output."""
    print("\n[bold blue]Testing Relationship Extraction[/bold blue]")
    
    # Retrieve environment variables
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    
    # Initialize Mem0llama client with minimal config for testing
    memory_config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "ollama_base_url": base_url,
                "api_key": api_key,
                "model": model_name.replace("ollama/", ""),
                "temperature": 0.1,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": os.getenv("EMBEDDER_MODEL", "nomic-embed-text").replace("ollama/", ""),
                "ollama_base_url": base_url,
                "api_key": api_key,
            },
        },
        "history_db_path": "./data/test_structured_output.db",
    }
    
    memory = Memory.from_config(memory_config)
    
    # Test text for relationship extraction
    test_text = "John works at Microsoft as a software engineer. He uses Python and JavaScript to build applications."
    entities = ["John", "Microsoft", "software_engineer", "Python", "JavaScript", "applications"]
    
    # Create a structured prompt
    prompt = create_relationship_extraction_prompt(test_text, entities, "test_user")
    
    # Generate response using the LLM with Pydantic model schema
    response = memory.llm.generate_response(
        messages=[
            {"role": "system", "content": "You are an AI assistant that identifies relationships between entities."},
            {"role": "user", "content": prompt}
        ],
        response_format=RelationshipExtraction.model_json_schema()
    )
    
    print(f"[bold]Raw LLM Response:[/bold]\n{response}")
    
    # Parse the response
    try:
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = response
            
        extraction_result = parse_llm_response(content, RelationshipExtraction)
        print("\n[bold green]Parsed Relationships:[/bold green]")
        for relationship in extraction_result.entities:
            print(f"  - {relationship.source} -- {relationship.relationship} --> {relationship.destination}")
    except Exception as e:
        print(f"[bold red]Error parsing response:[/bold red] {e}")


async def test_full_memory_flow():
    """Test the full memory flow with structured output."""
    print("\n[bold blue]Testing Full Memory Flow[/bold blue]")
    
    # Retrieve environment variables
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    embedder_model_name = os.getenv("EMBEDDER_MODEL", "nomic-embed-text")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    neo4j_url = os.getenv("NEO4J_URL")
    neo4j_username = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    user_id = "TestUser"
    
    # Initialize Mem0llama client with full config
    memory_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_test",
                "host": "ai.netcraft.fr",
                "port": 6333,
                "embedding_model_dims": 1024,
                "api_key": qdrant_api_key,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "ollama_base_url": base_url,
                "api_key": api_key,
                "model": model_name.replace("ollama/", ""),
                "temperature": 0.1,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embedder_model_name.replace("ollama/", ""),
                "ollama_base_url": base_url,
                "api_key": api_key,
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_username,
                "password": neo4j_password,
            },
        },
        "history_db_path": "./data/test_structured_output.db",
    }
    
    memory = Memory.from_config(memory_config)
    
    # Clear existing memories for clean test
    try:
        memory.delete_all(user_id=user_id)
        print("[green]Cleared existing memories for clean test[/green]")
    except Exception as e:
        print(f"[yellow]Warning: Could not clear memories: {e}[/yellow]")
    
    # Test adding a memory
    test_message = [
        {"role": "user", "content": "John works at Microsoft as a software engineer. He uses Python and JavaScript to build applications."},
        {"role": "assistant", "content": "That's interesting! John sounds like a skilled software engineer at Microsoft with experience in both Python and JavaScript."}
    ]
    
    print("\n[bold]Adding memory...[/bold]")
    add_result = memory.add(test_message, user_id=user_id)
    print(f"[green]Memory added: {add_result}[/green]")
    
    # Test searching for memories
    print("\n[bold]Searching for memories about 'John'...[/bold]")
    search_results = memory.search(query="Tell me about John", user_id=user_id)
    print(f"[green]Search results:[/green]")
    print(search_results)
    
    # Test getting all memories
    print("\n[bold]Getting all memories...[/bold]")
    all_memories = memory.get_all(user_id=user_id)
    print(f"[green]All memories:[/green]")
    print(all_memories)


async def main():
    """Run all tests."""
    await test_entity_extraction()
    await test_relationship_extraction()
    await test_full_memory_flow()


if __name__ == "__main__":
    asyncio.run(main())
