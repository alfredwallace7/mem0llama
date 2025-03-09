from dotenv import load_dotenv
load_dotenv(override=True)
import os
import argparse
from litellm import acompletion
from mem0llama import Memory
import asyncio
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
import json
import copy
from collections import deque

console = Console()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mem0llama Terminal Chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to display detailed memory and relation information")
    args = parser.parse_args()
    
    # Retrieve environment variables
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    embedder_model_name = os.getenv("EMBEDDER_MODEL")
    # Remove protocol from QDRANT_HOST if present
    qdrant_host = os.getenv("QDRANT_HOST", "localhost:6333")
    qdrant_host = qdrant_host.replace("http://", "").replace("https://", "")
    if ":" not in qdrant_host:
        qdrant_host += ":6333"
    qdrant_port = int(qdrant_host.split(":")[1]) if ":" in qdrant_host else 6333
    qdrant_host = qdrant_host.split(":")[0]
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    neo4j_url = os.getenv("NEO4J_URL")
    neo4j_username = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    user_id = "Alfred"
#     system_prompt = """You are a helpful assistant with access to a memory system that synthesizes past interactions and relationships to provide contextually relevant responses. Use this synthesized memory naturally in conversation, integrating relevant details where helpful.

# When responding:
# - Incorporate relevant past interactions and relationships naturally, without forcing recall.
# - Do not infer the user's thoughts or emotions unless explicitly stated.
# - Maintain your role as an assistant—do not speak on behalf of the user.
# - Ask open-ended questions to encourage engagement rather than making assumptions.

# {memories_context}"""
    system_prompt = """You are a helpful assistant with access to a memory system that synthesizes past interactions and relationships to provide contextually relevant responses. 
Use this synthesized memory naturally in conversation, integrating relevant details where helpful.

When responding:
- Incorporate relevant past interactions and relationships naturally, without forcing recall.
- Prioritize the last few conversational turns to maintain continuity before retrieving deeper memories.
- Do not infer the user's thoughts or emotions unless explicitly stated.
- Maintain your role as an assistant—do not speak on behalf of the user.
- Ask open-ended questions to encourage engagement rather than making assumptions.

Memory retrieval strategy:
1. **Short-term memory:** Recall the last 3 turns to maintain recent context.
2. **Graph-based relationships:** Leverage structured connections between concepts.
3. **Long-term vector memory:** Retrieve older interactions only when relevant.

Recent conversation:
{last_3_turns}

Context from memory:
{memories_context}"""

    # Initialize Mem0 client
    memory_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "host": qdrant_host,
                "port": qdrant_port,
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
                # "temperature": 0.2,
                # "max_tokens": 1500,
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
        "history_db_path": "./data/mem0_global_history.db",
        "version": "v1.1",
    }

    # Initialize Mem0 client
    memory = Memory.from_config(memory_config)
    conversation_history = deque(maxlen=3)  # Keep last 3 turns
    
    def format_conversation_history():
        if not conversation_history:
            return "No previous conversation."
        formatted_turns = []
        for i, (user_msg, ai_msg) in enumerate(conversation_history, 1):
            formatted_turns.extend([
                f"Turn {i}:",
                f"User: {user_msg}",
                f"Assistant: {ai_msg}",
                ""  # Empty line between turns
            ])
        return "\n".join(formatted_turns)
    
    debug_mode = "enabled" if args.debug else "disabled"
    console.print(Panel.fit(
        f"[bold blue]Mem0llama Terminal Chatbot (Debug mode: {debug_mode})[/bold blue]\n"
        "[green]Type your message to interact with the assistant.[/green]\n"
        "[yellow]Commands:[/yellow]\n"
        "  [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit the application\n"
        "  [cyan]/clear[/cyan] or [cyan]/reset[/cyan] - Clear memory\n"
        "[yellow]Debug Mode:[/yellow]\n"
        "  Run with [cyan]--debug[/cyan] flag to display detailed memory and relation information",
        title="Welcome",
        border_style="blue"
    ))

    while True:
        user_input = input("\n[User]: ")
        
        if not user_input.strip():
            continue
        elif user_input.lower() in ["/exit", "/quit"]:
            console.print("[bold red]Exiting...[/bold red]")
            break
        elif user_input.lower() in ["/clear", "/reset"]:
            memory.delete_all(user_id=user_id)
            conversation_history.clear()
            console.print("[bold yellow]Memory cleared.[/bold yellow]")
            continue

        # Retrieve relevant memories from vector store
        console.print("[dim]Searching for relevant memories...[/dim]")
        memories_str = ""
        relevant_memories = memory.search(query=user_input, user_id=user_id)
        if relevant_memories and len(relevant_memories) > 0:
            from datetime import datetime
            
            def format_date(date_str):
                if not date_str:
                    return "unknown date"
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return dt.strftime("%B %d, %Y, %H:%M")
                except (ValueError, AttributeError):
                    return "unknown date"
            
            # Sort memories by date (newest first)
            sorted_memories = sorted(
                relevant_memories['results'],
                key=lambda x: x.get('updated_at', '') or x.get('created_at', ''),
                reverse=True
            )
            
            memory_entries = []
            for entry in sorted_memories:
                date = entry.get('updated_at') or entry.get('created_at')
                date_str = format_date(date)
                memory_entries.append(f"- {entry['memory']} (from {date_str})")
            memories_str = "\n".join(memory_entries)
            console.print(f"[dim]Found {len(relevant_memories['results'])} relevant memories[/dim]")
            if args.debug:
                # Create a deep copy of relevant_memories for debug display to avoid modifying the original
                debug_memories = copy.deepcopy(relevant_memories)
                
                # Remove 'relations' key if it exists to avoid duplication with Graph Relationships display
                if isinstance(debug_memories, dict):
                    if 'relations' in debug_memories:
                        debug_memories.pop('relations')
                    # Also check if relations might be nested in results
                    if 'results' in debug_memories and isinstance(debug_memories['results'], list):
                        for item in debug_memories['results']:
                            if isinstance(item, dict) and 'relations' in item:
                                item.pop('relations')
                
                console.print(Panel.fit(
                    Syntax(json.dumps(debug_memories, indent=4), "json", word_wrap=True),
                    title="[bold yellow]Debug: Vector Memories[/bold yellow]",
                    border_style="yellow"
                ))
        else:
            console.print("[dim]No relevant vector memories found.[/dim]")

        # Retrieve graph relationships
        console.print("[dim]Retrieving graph relationships...[/dim]")
        graph_relations = memory.graph.search(query=user_input, filters={"user_id": user_id})
        relations_str = ""
        if graph_relations and len(graph_relations) > 0:
            relations_str = "\n".join(
                f"- {relation['source']} {relation['relationship']} {relation['destination']}"
                for relation in graph_relations
            )
            console.print(f"[dim]Found {len(graph_relations)} relevant relationships[/dim]")
            if args.debug:
                console.print(Panel.fit(
                    Syntax(json.dumps(graph_relations, indent=4), "json", word_wrap=True),
                    title="[bold green]Debug: Graph Relationships[/bold green]",
                    border_style="green"
                ))
        else:
            console.print("[dim]No relevant graph relationships found.[/dim]")

        # Build the memory context for system prompt
        context_parts = []
        if memories_str:
            context_parts.append(f"Vector Memories:\n{memories_str}")
        if relations_str:
            context_parts.append(f"Graph Relationships:\n{relations_str}")

        memories_context = f"\nContext from memory:\n{'\n\n'.join(context_parts)}" if context_parts else ""
        current_system_prompt = system_prompt.format(
            memories_context=memories_context,
            last_3_turns=format_conversation_history()
        )

        if args.debug:
            console.print(Panel.fit(
                current_system_prompt,
                title="[bold cyan]Debug: System Prompt[/bold cyan]",
                border_style="cyan"
            ))

        prompt = f"User:\n{user_input}"

        # Show a spinner while waiting for the response
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            response = await acompletion(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                messages=[
                    {"role": "system", "content": current_system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

        content = response["choices"][0]["message"]["content"]
        
        # Display the response with markdown formatting
        console.print(Panel(
            Markdown(content),
            title=f"[bold]{model_name.replace('ollama/', '')}[/bold]",
            border_style="green"
        ))

        # Update conversation history
        conversation_history.append((user_input, content))

        # Save the interaction to memory
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": content}
        ]

        memory.add(messages, user_id=user_id, prompt=user_input)
        console.print("[dim]Conversation saved to memory[/dim]")

if __name__ == "__main__":
    asyncio.run(main())