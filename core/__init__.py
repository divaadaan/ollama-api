"""Core package """

from .llm_client import LLMClient, create_llm_client
from .agents import BasicAgent, create_basic_agent
from .smolagents_adapter import SmolOllamaAdapter, create_smol_ollama_adapter
from .tools import get_default_tools, AVAILABLE_TOOLS

__version__ = "1.0.0"

__all__ = [
    # LLM Client
    'LLMClient',
    'create_llm_client',

    # Agents
    'BasicAgent',
    'create_basic_agent',

    # Adapter
    'SmolOllamaAdapter',
    'create_smol_ollama_adapter'
]