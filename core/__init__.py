"""Core package """

from .agents import BasicAgent, create_basic_agent
from .llm_client import LLMClient, create_llm_client
from .smolagents_adapter import SmolOllamaAdapter, create_smol_ollama_adapter
from .tools import AVAILABLE_TOOLS, get_default_tools

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