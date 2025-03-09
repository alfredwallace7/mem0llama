"""
Unit tests for validating the enhanced structured output capabilities of Mem0llama.
This script tests the integration of Pydantic models and LLM formatting utilities
with small local LLMs using Ollama.
"""

import os
import unittest
import asyncio
from dotenv import load_dotenv
from mem0llama import Memory
from mem0llama.memory.models import EntityExtraction, RelationshipExtraction
from mem0llama.memory.llm_formatter import (
    parse_llm_response, 
    create_entity_extraction_prompt, 
    create_relationship_extraction_prompt
)

load_dotenv(override=True)


class TestStructuredOutput(unittest.IsolatedAsyncioTestCase):
    """Test case for structured output capabilities."""

    async def asyncSetUp(self):
        """Set up test environment."""
        # Ensure we have the required environment variables
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "llama3")
        self.embedder_model = os.getenv("EMBEDDER_MODEL", "nomic-embed-text")
        # Remove protocol from QDRANT_HOST if present
        qdrant_host = os.getenv("QDRANT_HOST", "localhost:6333")
        self.qdrant_host = qdrant_host.replace("http://", "").replace("https://", "")
        if ":" not in self.qdrant_host:
            self.qdrant_host += ":6333"
        self.qdrant_port = int(self.qdrant_host.split(":")[1]) if ":" in self.qdrant_host else 6333
        self.qdrant_host = self.qdrant_host.split(":")[0]
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.neo4j_url = os.getenv("NEO4J_URL")
        self.neo4j_username = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Test data
        self.test_message = "John works at Microsoft. He is a Software Engineer. " \
                           "He uses Python and JavaScript to build applications."
        self.user_id = "TestUser"
        
        # Configure memory with proper settings for testing
        self.memory_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "host": self.qdrant_host,
                    "port": self.qdrant_port,
                    "embedding_model_dims": 1024,
                    "api_key": self.qdrant_api_key,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "ollama_base_url": self.base_url,
                    "api_key": self.api_key,
                    "model": self.model_name.replace("ollama/", ""),
                    "temperature": 0.1,
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.embedder_model.replace("ollama/", ""),
                    "ollama_base_url": self.base_url,
                    "api_key": self.api_key,
                },
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": self.neo4j_url,
                    "username": self.neo4j_username,
                    "password": self.neo4j_password,
                },
            },
            "history_db_path": "./data/unittest_history.db",
            "version": "v1.1",
        }
        
        # Initialize memory with config
        try:
            self.memory = Memory.from_config(self.memory_config)
        except Exception as e:
            self.fail(f"Memory initialization failed: {e}")
        
        # Clean up any existing test data
        try:
            self.memory.delete_all(user_id=self.user_id)
        except Exception as e:
            print(f"Could not clear memories: {e}")

    async def test_entity_extraction(self):
        """Test entity extraction with structured output."""
        # Create entity extraction prompt
        prompt = create_entity_extraction_prompt(self.test_message, self.user_id)
        
        # Generate response using the LLM with Pydantic model schema
        try:
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that extracts entities and their types from text."},
                    {"role": "user", "content": prompt}
                ],
                response_format=EntityExtraction.model_json_schema()
            )
        except Exception as e:
            self.fail(f"Entity extraction failed: {e}")
        
        # Parse response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = response
            
        try:
            result = parse_llm_response(content, EntityExtraction)
        except Exception as e:
            self.fail(f"Failed to parse entity extraction response: {e}")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EntityExtraction)
        self.assertTrue(len(result.entities) > 0)
        
        # Check for expected entities
        entity_names = [entity.entity.lower() for entity in result.entities]
        entity_types = [entity.entity_type.lower() for entity in result.entities]
        
        self.assertIn("john", entity_names)
        self.assertIn("microsoft", entity_names)
        self.assertTrue("person" in entity_types or "individual" in entity_types or "human" in entity_types)
        self.assertTrue("organization" in entity_types or "company" in entity_types or "corporation" in entity_types)
        
        return result.entities

    async def test_relationship_extraction(self):
        """Test relationship extraction with structured output."""
        # First extract entities or use predefined entities
        entities = ["John", "Microsoft", "Software Engineer", "Python", "JavaScript", "applications"]
        
        # Create relationship extraction prompt
        prompt = create_relationship_extraction_prompt(self.test_message, entities, self.user_id)
        
        # Generate response using the LLM with Pydantic model schema
        try:
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that identifies relationships between entities."},
                    {"role": "user", "content": prompt}
                ],
                response_format=RelationshipExtraction.model_json_schema()
            )
        except Exception as e:
            self.fail(f"Relationship extraction failed: {e}")
        
        # Parse response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = response
            
        try:
            result = parse_llm_response(content, RelationshipExtraction)
        except Exception as e:
            self.fail(f"Failed to parse relationship extraction response: {e}")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, RelationshipExtraction)
        self.assertTrue(len(result.entities) > 0)
        
        # Check for expected relationships
        relationships = [(rel.source.lower(), rel.relationship.lower(), rel.destination.lower()) 
                        for rel in result.entities]
        
        # Look for expected relationships with more flexible matching
        work_relationship_found = False
        use_relationship_found = False
        
        for rel in relationships:
            # Check for John works at Microsoft relationship
            if "john" in rel[0] and any(work_term in rel[1] for work_term in ["work", "employ"]) and "microsoft" in rel[2]:
                work_relationship_found = True
            
            # Check for John uses Python/JavaScript relationship
            if "john" in rel[0] and any(use_term in rel[1] for use_term in ["use", "code", "program", "develop"]) and any(lang in rel[2] for lang in ["python", "javascript"]):
                use_relationship_found = True
        
        self.assertTrue(work_relationship_found, "Work relationship not found")
        self.assertTrue(use_relationship_found, "Use relationship not found")

    async def test_full_memory_flow(self):
        """Test the full memory flow with structured output."""
        # Test adding a memory
        test_message = [
            {"role": "user", "content": self.test_message},
            {"role": "assistant", "content": "That's interesting! John sounds like a skilled software engineer at Microsoft with experience in both Python and JavaScript."}
        ]
        
        # Clear existing memories for clean test
        self.memory.delete_all(user_id=self.user_id)
        
        # Add memory
        try:
            add_result = self.memory.add(test_message, user_id=self.user_id)
        except Exception as e:
            self.fail(f"Failed to add memory: {e}")
        
        # Assertions for add
        self.assertIsNotNone(add_result)
        self.assertTrue(len(add_result) > 0)
        
        # Get all memories
        try:
            all_memories = self.memory.get_all(user_id=self.user_id)
        except Exception as e:
            self.fail(f"Failed to get all memories: {e}")
        
        # Assertions for get_all
        self.assertIsNotNone(all_memories)
        self.assertTrue(len(all_memories) > 0)
        
        # Search memories
        try:
            search_results = self.memory.search("Microsoft", user_id=self.user_id)
        except Exception as e:
            self.fail(f"Failed to search memories: {e}")
        
        # Assertions for search
        self.assertIsNotNone(search_results)
        self.assertTrue(len(search_results) > 0)
        
        # Check if graph relations were created (if graph is enabled)
        if hasattr(self.memory, 'graph') and self.memory.graph:
            try:
                # Use get_all method with user_id filter
                relations = self.memory.graph.get_all({"user_id": self.user_id})
                self.assertIsNotNone(relations)
                # Check that we have some results
                self.assertTrue(len(relations) > 0)
                
                # Clean up graph
                self.memory.graph.delete_all({"user_id": self.user_id})
            except Exception as e:
                self.fail(f"Failed to get graph relations: {e}")
        
        # Clean up
        self.memory.delete_all(user_id=self.user_id)

    async def asyncTearDown(self):
        """Clean up after tests."""
        try:
            if hasattr(self, 'memory') and self.memory:
                self.memory.delete_all(user_id=self.user_id)
        except Exception as e:
            print(f"Could not clear memories during teardown: {e}")


if __name__ == "__main__":
    unittest.main()
