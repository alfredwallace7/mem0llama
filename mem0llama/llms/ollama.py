from typing import Dict, List, Optional

try:
    from ollama import Client
except ImportError:
    raise ImportError("The 'ollama' library is required. Please install it using 'pip install ollama'.")

from mem0llama.configs.llms.base import BaseLlmConfig
from mem0llama.llms.base import LLMBase


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1:70b"
        self.client = Client(host=self.config.ollama_base_url)
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """
        Ensure the specified model exists locally. If not, pull it from Ollama.
        """
        local_models = self.client.list()["models"]
        if not any(model.get("name") == self.config.model for model in local_models):
            self.client.pull(self.config.model)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response["message"]["content"],
                "tool_calls": [],
            }

            if response["message"].get("tool_calls"):
                for tool_call in response["message"]["tool_calls"]:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        }
                    )

            return processed_response
        else:
            return response["message"]["content"]

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Ollama.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Can be a Pydantic model schema.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str or dict: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        
        # Handle structured output formatting
        if response_format:
            # If response_format is a dict, it could be a Pydantic schema or OpenAI format
            if isinstance(response_format, dict):
                if "type" in response_format and response_format.get("type") == "json_object":
                    # OpenAI-style format parameter
                    params["format"] = "json"
                else:
                    # Assume it's a Pydantic schema
                    params["format"] = response_format
            elif response_format == "json":
                params["format"] = "json"
            
            # Add JSON format instruction to system message if not already present
            has_json_instruction = False
            for msg in messages:
                if msg["role"] == "system" and "JSON" in msg["content"]:
                    has_json_instruction = True
                    break
                    
            if not has_json_instruction and messages and messages[0]["role"] == "system":
                messages[0]["content"] += " Respond using valid JSON format only."

        if tools:
            params["tools"] = tools

        response = self.client.chat(**params)
        return self._parse_response(response, tools)
