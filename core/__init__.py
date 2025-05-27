"""Core package """

from .llm_client import LLMClient, create_llm_client
from .agents import BasicAgent, create_basic_agent
from .smolagents_adapter import SmolOllamaAdapter, create_smol_ollama_adapter

__version__ = "1.0.0"

__all__ = [
    # LLM Client
    'LLMClient',
    'create_llm_client',

    # Agents
    'BasicAgent',
    'LLMClientAdapter',
    'create_basic_agent'
]