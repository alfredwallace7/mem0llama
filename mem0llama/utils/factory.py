import importlib

from mem0llama.configs.embeddings.base import BaseEmbedderConfig
from mem0llama.configs.llms.base import BaseLlmConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "mem0llama.llms.ollama.OllamaLLM",
        "openai": "mem0llama.llms.openai.OpenAILLM",
        "groq": "mem0llama.llms.groq.GroqLLM",
        "together": "mem0llama.llms.together.TogetherLLM",
        "aws_bedrock": "mem0llama.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "mem0llama.llms.litellm.LiteLLM",
        "azure_openai": "mem0llama.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "mem0llama.llms.openai_structured.OpenAIStructuredLLM",
        "anthropic": "mem0llama.llms.anthropic.AnthropicLLM",
        "azure_openai_structured": "mem0llama.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
        "gemini": "mem0llama.llms.gemini.GeminiLLM",
        "deepseek": "mem0llama.llms.deepseek.DeepSeekLLM",
        "xai": "mem0llama.llms.xai.XAILLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "mem0llama.embeddings.openai.OpenAIEmbedding",
        "ollama": "mem0llama.embeddings.ollama.OllamaEmbedding",
        "huggingface": "mem0llama.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "mem0llama.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "mem0llama.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "mem0llama.embeddings.vertexai.VertexAIEmbedding",
        "together": "mem0llama.embeddings.together.TogetherEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0llama.vector_stores.qdrant.Qdrant",
        "chroma": "mem0llama.vector_stores.chroma.ChromaDB",
        "pgvector": "mem0llama.vector_stores.pgvector.PGVector",
        "milvus": "mem0llama.vector_stores.milvus.MilvusDB",
        "azure_ai_search": "mem0llama.vector_stores.azure_ai_search.AzureAISearch",
        "redis": "mem0llama.vector_stores.redis.RedisDB",
        "elasticsearch": "mem0llama.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "mem0llama.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "mem0llama.vector_stores.opensearch.OpenSearchDB",
        "supabase": "mem0llama.vector_stores.supabase.Supabase",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")
