"""
Pydantic models for standardizing LLM outputs in Mem0's graph memory system.
These models ensure structured and predictable outputs from LLMs, especially
when using small local models with Ollama's format argument.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Model representing an entity extracted from text."""
    entity: str = Field(..., description="The name or identifier of the entity")
    entity_type: str = Field(..., description="The type or category of the entity")


class Relationship(BaseModel):
    """Model representing a relationship between two entities."""
    source: str = Field(..., description="The source entity of the relationship")
    relationship: str = Field(..., description="The relationship between the source and destination entities")
    destination: str = Field(..., description="The destination entity of the relationship")


class EntityExtraction(BaseModel):
    """Model representing the output of entity extraction."""
    entities: List[Entity] = Field(..., description="List of entities extracted from text")


class RelationshipExtraction(BaseModel):
    """Model representing the output of relationship extraction."""
    entities: List[Relationship] = Field(..., description="List of relationships between entities")


class DeleteMemory(BaseModel):
    """Model representing a memory to be deleted."""
    source: str = Field(..., description="The source entity in the relationship")
    relationship: str = Field(..., description="The relationship between entities")
    destination: str = Field(..., description="The destination entity in the relationship")


class DeleteMemoryOutput(BaseModel):
    """Model representing the output of memory deletion."""
    entities: List[DeleteMemory] = Field(default_factory=list, description="List of memories to delete")


class UpdateMemory(BaseModel):
    """Model representing a memory to be updated."""
    source: str = Field(..., description="The source entity in the relationship")
    relationship: str = Field(..., description="The relationship between entities")
    destination: str = Field(..., description="The destination entity in the relationship")


class UpdateMemoryOutput(BaseModel):
    """Model representing the output of memory updates."""
    entities: List[UpdateMemory] = Field(default_factory=list, description="List of memories to update")


class AddMemory(BaseModel):
    """Model representing a memory to be added."""
    source: str = Field(..., description="The source entity in the relationship")
    destination: str = Field(..., description="The destination entity in the relationship")
    relationship: str = Field(..., description="The relationship between entities")
    source_type: str = Field(..., description="The type of the source entity")
    destination_type: str = Field(..., description="The type of the destination entity")


class AddMemoryOutput(BaseModel):
    """Model representing the output of memory addition."""
    entities: List[AddMemory] = Field(default_factory=list, description="List of memories to add")


class NoopOutput(BaseModel):
    """Model representing a no-operation output."""
    message: str = Field(default="No operation needed", description="Message indicating no operation is needed")
