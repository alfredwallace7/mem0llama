import importlib.metadata

__version__ = importlib.metadata.version("mem0llama")

from mem0llama.client.main import MemoryClient, AsyncMemoryClient  # noqa
from mem0llama.memory.main import Memory  # noqa
