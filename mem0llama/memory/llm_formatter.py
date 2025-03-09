"""
Utilities for formatting LLM prompts and parsing responses for Mem0's graph memory system.
This module provides functions to create structured prompts for Ollama and other LLMs,
and to parse their responses into standardized formats using Pydantic models.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from mem0llama.memory.models import (
    AddMemory,
    AddMemoryOutput,
    DeleteMemory,
    DeleteMemoryOutput,
    Entity,
    EntityExtraction,
    NoopOutput,
    Relationship,
    RelationshipExtraction,
    UpdateMemory,
    UpdateMemoryOutput,
)

logger = logging.getLogger(__name__)


def create_ollama_format_template(model_class):
    """
    Create a JSON schema format template for Ollama based on a Pydantic model.
    
    Args:
        model_class: A Pydantic model class to generate the schema from
        
    Returns:
        dict: A dictionary containing the JSON schema for the Ollama format parameter
    """
    schema = model_class.model_json_schema()
    return {
        "schema": schema
    }


def parse_llm_response(response: str, model_class) -> Any:
    """
    Parse an LLM response into a Pydantic model.
    
    Args:
        response: The raw response string from the LLM
        model_class: The Pydantic model class to parse the response into
        
    Returns:
        An instance of the specified Pydantic model class
    """
    if not response:
        logger.warning("Empty response from LLM")
        return None
    
    # Handle different response formats
    if isinstance(response, dict):
        # If response is already a dict, it might be from Ollama's direct response
        if "message" in response and "content" in response["message"]:
            content = response["message"]["content"]
        elif "content" in response:
            content = response["content"]
        else:
            # It might already be the parsed JSON
            content = response
    else:
        # Assume it's a string
        content = response
    
    # If content is a string, try to parse it as JSON
    if isinstance(content, str):
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        # Try to parse as JSON
        try:
            # Try direct model validation first
            try:
                return model_class.model_validate_json(content)
            except Exception as e:
                # If direct validation fails, try parsing to dict first
                parsed_content = json.loads(content)
                return model_class.model_validate(parsed_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw content: {content}")
            # Return a default instance as fallback
            return model_class()
    
    # If content is already a dict, validate it directly
    try:
        return model_class.model_validate(content)
    except Exception as e:
        logger.error(f"Failed to validate model: {e}")
        logger.debug(f"Content for validation: {content}")
        # Return a default instance as fallback
        return model_class()


def create_entity_extraction_prompt(text: str, user_id: str) -> str:
    """
    Create a prompt for entity extraction.
    
    Args:
        text: The text to extract entities from
        user_id: The user ID for reference
        
    Returns:
        str: A formatted prompt for entity extraction
    """
    return f"""You are a smart assistant who understands entities and their types in a given text.
If user message contains self-reference such as 'I', 'me', 'my' etc. then use {user_id} as the source entity.
Extract all the entities from the text. DO NOT answer the question itself if the given text is a question.

Extract entities in the following JSON format:
{{
  "entities": [
    {{
      "entity": "entity_name",
      "entity_type": "entity_type"
    }}
  ]
}}

Text: {text}"""


def create_relationship_extraction_prompt(text: str, entities: List[str], user_id: str) -> str:
    """
    Create a prompt for relationship extraction.
    
    Args:
        text: The text to extract relationships from
        entities: A list of entity names
        user_id: The user ID for reference
        
    Returns:
        str: A formatted prompt for relationship extraction
    """
    return f"""You are a smart assistant who understands relationships between entities in text.
Establish relationships among the entities based on the provided text.

Rules:
1. Only establish relationships that are explicitly mentioned or strongly implied in the text.
2. If the text mentions that a relationship no longer exists, do not include it.
3. If the user refers to themselves (using 'I', 'me', 'my', etc.), use '{user_id}' as the entity.

List of entities: {entities}

Text: {text}

Return the relationships in the following JSON format:
{{
  "entities": [
    {{
      "source": "source_entity",
      "relationship": "relationship_type",
      "destination": "destination_entity"
    }}
  ]
}}"""


def create_delete_memory_prompt(existing_memories: str, new_text: str, user_id: str) -> str:
    """
    Create a prompt for identifying memories to delete.
    
    Args:
        existing_memories: String representation of existing memories
        new_text: New text that may contradict existing memories
        user_id: The user ID for reference
        
    Returns:
        str: A formatted prompt for memory deletion
    """
    return f"""You are an assistant managing a knowledge graph of memories.
Based on new information, identify which existing relationships should be deleted because they are:
1. Directly contradicted by the new information
2. No longer true or valid based on the new information
3. Superseded by more accurate or updated information

Existing relationships:
{existing_memories}

New information:
{new_text}

If the user refers to themselves (using 'I', 'me', 'my', etc.), use '{user_id}' as the entity.

Return the relationships to delete in the following JSON format:
{{
  "entities": [
    {{
      "source": "source_entity",
      "relationship": "relationship_type",
      "destination": "destination_entity"
    }}
  ]
}}

If no relationships need to be deleted, return an empty list of entities:
{{
  "entities": []
}}"""


def adapt_llm_provider_for_structured_output(llm_provider: str, llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt the LLM configuration based on the provider to support structured outputs.
    
    Args:
        llm_provider: The LLM provider name (e.g., 'ollama', 'openai')
        llm_config: The original LLM configuration
        
    Returns:
        Dict[str, Any]: Updated LLM configuration
    """
    config = llm_config.copy()
    
    if llm_provider.lower() == 'ollama':
        # For Ollama, we need to handle format differently since BaseLlmConfig doesn't accept it
        # We'll set an environment variable or use a different approach during LLM initialization
        # For now, just ensure temperature is appropriate
        if 'temperature' not in config or config['temperature'] > 0.2:
            config['temperature'] = 0.1
            
        # Add a note in the log about format parameter
        logger.info("For Ollama structured output, format=json should be set during LLM initialization")
    
    elif llm_provider.lower() in ['openai', 'azure_openai']:
        # OpenAI already has good structured output capabilities
        if 'response_format' not in config:
            config['response_format'] = {'type': 'json_object'}
        
        # Ensure temperature is appropriate for structured tasks
        if 'temperature' not in config or config['temperature'] > 0.3:
            config['temperature'] = 0.2
    
    return config
