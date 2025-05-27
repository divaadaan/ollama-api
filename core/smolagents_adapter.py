"""Smolagents adapter for LLMClient."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from smolagents import Model

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Proper response object for smolagents compatibility."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    model: str = "unknown"

    def __post_init__(self):
        """Set up token usage dict for compatibility."""
        self.token_usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }
        self.usage = self.token_usage


class SmolOllamaAdapter(Model):
    """Smolagents adapter for LLMClient"""

    def __init__(self, llm_client, **kwargs):
        """
        Initialize adapter with LLM client.

        Args:
            llm_client: LLMClient instance ()
            **kwargs: Additional Model parameters
        """
        super().__init__(**kwargs)
        self.llm_client = llm_client
        logger.info(
            f"SmolOllamaAdapter initialized with telemetry {'enabled' if llm_client.telemetry_enabled else 'disabled'}")

    def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Call interface for smolagents compatibility."""
        prompt = self._format_messages(messages)
        result = self.llm_client.generate(prompt, **kwargs)

        if "error" in result:
            raise RuntimeError(f"LLM generation failed: {result['error']}")

        # Return dict with ALL expected smolagents fields
        response_dict =  {
            "content": result["content"],
            "input_tokens": result.get("prompt_tokens", 0),
            "output_tokens": result.get("completion_tokens", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "token_usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            },
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            },
            "finish_reason": result.get("finish_reason", "stop"),
            "model": result.get("model", "unknown")
        }

        logger.info(f"__call__ returning dict with keys: {list(response_dict.keys())}")
        logger.info(f"__call__ input_tokens value: {response_dict.get('input_tokens', 'MISSING')}")
        return response_dict

    def generate(self, messages: List[Dict], temperature: float = 0.5, stop_sequences: Optional[List[str]] = None,
                 **kwargs) -> LLMResponse:
        prompt = self._format_messages(messages)
        result = self.llm_client.generate(prompt=prompt, temperature=temperature, **kwargs)

        if "error" in result:
            raise RuntimeError(f"LLM generation failed: {result['error']}")

        prompt_tokens = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)
        total_tokens = result.get("total_tokens", 0)

        response = LLMResponse(
            content=result["content"],
            input_tokens=prompt_tokens,  # smolagents expects this
            output_tokens=completion_tokens,  # smolagents expects this
            prompt_tokens=prompt_tokens,  # traditional name
            completion_tokens=completion_tokens,  # traditional name
            total_tokens=total_tokens,  # traditional name
            finish_reason=result.get("finish_reason", "stop"),
            model=result.get("model", "unknown")
        )

        logger.info(f"generate() returning LLMResponse with input_tokens: {response.input_tokens}")
        logger.info(f"generate() LLMResponse attributes: {dir(response)}")

        return response


    def _format_messages(self, messages: List[Dict]) -> str:
        """
        Convert smolagents message format to a single prompt string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        formatted_parts = []

        for message in messages:
            if isinstance(message, dict):
                # Handle different message formats
                if "content" in message:
                    content = message["content"]
                    role = message.get("role", "")
                    if role:
                        # Format with role prefix
                        formatted_parts.append(f"{role}: {content}")
                    else:
                        formatted_parts.append(content)
                elif "text" in message:
                    # Alternative format
                    formatted_parts.append(message["text"])
                else:
                    # Fallback: convert dict to string
                    formatted_parts.append(str(message))
            else:
                # If it's already a string, use it directly
                formatted_parts.append(str(message))

        return "\n".join(formatted_parts)

    @property
    def telemetry_enabled(self) -> bool:
        """Check if the underlying LLM client has telemetry enabled."""
        return self.llm_client.telemetry_enabled

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check through the underlying LLM client.

        Returns:
            Dictionary with health status information
        """
        try:
            # Use the LLMClient's health check
            llm_health = self.llm_client.health_check()

            # Add adapter-specific information
            return {
                "status": llm_health["status"],
                "adapter": "SmolOllamaAdapter",
                "llm_client": llm_health,
                "telemetry_enabled": self.telemetry_enabled
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "adapter": "SmolOllamaAdapter",
                "error": f"Adapter health check failed: {str(e)}",
                "telemetry_enabled": self.telemetry_enabled
            }


# Factory function for easy creation
def create_smol_ollama_adapter(
        llm_client,
        **kwargs
) -> SmolOllamaAdapter:
    """
    Factory function to create SmolOllamaAdapter.

    Args:
        llm_client: LLMClient instance
        **kwargs: Additional Model parameters

    Returns:
        Configured SmolOllamaAdapter instance
    """
    return SmolOllamaAdapter(llm_client=llm_client, **kwargs)