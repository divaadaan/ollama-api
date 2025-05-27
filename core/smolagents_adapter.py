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
    token_usage: Dict[str, int]
    finish_reason: str = "stop"
    model: str = "unknown"

    @property
    def usage(self) -> Dict[str, int]:
        """Alias for token_usage for compatibility."""
        return self.token_usage


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

    def __call__(self, messages: List[Dict], **kwargs) -> Dict[str, str]:
        """Call interface for smolagents compatibility."""
        prompt = self._format_messages(messages)
        result = self.llm_client.generate(prompt, **kwargs)

        if "error" in result:
            raise RuntimeError(f"LLM generation failed: {result['error']}")

        return {"content": result["content"]}

    def generate(self, messages: List[Dict], temperature: float = 0.5, stop_sequences: Optional[List[str]] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate interface for smolagents compatibility.

        Args:
            messages: List of message dictionaries in smolagents format
            temperature: Temperature for generation
            stop_sequences: Stop sequences (not implemented in Ollama)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object with proper smolagents attributes
        """
        prompt = self._format_messages(messages)

        # Call the  LLMClient
        result = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            **kwargs
        )

        if "error" in result:
            raise RuntimeError(f"LLM generation failed: {result['error']}")

        # Convert LLMClient response to smolagents format
        return LLMResponse(
            content=result["content"],
            token_usage={
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            },
            finish_reason=result.get("finish_reason", "stop"),
            model=result.get("model", "unknown")
        )

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