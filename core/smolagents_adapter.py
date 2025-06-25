"""Smolagents adapter for LLMClient."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from smolagents import Model

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def prompt_tokens(self):
        return self.input_tokens

    @property
    def completion_tokens(self):
        return self.output_tokens

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    """Response object that matches what smolagents expects."""
    content: str
    token_usage: TokenUsage
    finish_reason: str = "stop"
    model: str = "unknown"

    @property
    def input_tokens(self):
        return self.token_usage.input_tokens

    @property
    def output_tokens(self):
        return self.token_usage.output_tokens

    @property
    def usage(self):
        return {
            "prompt_tokens": self.token_usage.input_tokens,
            "completion_tokens": self.token_usage.output_tokens,
            "total_tokens": self.token_usage.input_tokens + self.token_usage.output_tokens
        }


class SmolOllamaAdapter(Model):
    """Smolagents adapter for LLMClient"""

    def __init__(self, llm_client, **kwargs):
        """
        Initialize adapter with LLM client.

        Args:
            llm_client: LLMClient instance
            **kwargs: Additional Model parameters
        """
        super().__init__(**kwargs)
        self.llm_client = llm_client
        logger.info(
            f"SmolOllamaAdapter initialized with telemetry {'enabled' if llm_client.telemetry_enabled else 'disabled'}")

    def __call__(self, messages: List[Dict], **kwargs) -> ModelResponse:
        """Call interface for smolagents compatibility."""
        return self._generate_response(messages, **kwargs)

    def generate(self, messages: List[Dict], temperature: float = 0.5, stop_sequences: Optional[List[str]] = None,
                 **kwargs) -> ModelResponse:
        """Generate interface for smolagents compatibility."""
        return self._generate_response(messages, temperature=temperature, stop_sequences=stop_sequences, **kwargs)

    def _generate_response(self, messages: List[Dict], temperature: Optional[float] = None,
                          stop_sequences: Optional[List[str]] = None, **kwargs) -> ModelResponse:
        """Common generation logic for both __call__ and generate methods."""
        prompt = self._format_messages(messages)

        gen_kwargs = kwargs.copy()
        if temperature is not None:
            gen_kwargs['temperature'] = temperature

        result = self.llm_client.generate(prompt=prompt, **gen_kwargs)

        if "error" in result:
            raise RuntimeError(f"LLM generation failed: {result['error']}")

        content = result.get("content", "")

        token_usage = TokenUsage(
            input_tokens=result.get("prompt_tokens", 0),
            output_tokens=result.get("completion_tokens", 0)
        )

        response = ModelResponse(
            content=content,
            token_usage=token_usage,
            finish_reason=result.get("finish_reason", "stop"),
            model=result.get("model", "unknown")
        )

        logger.debug(f"Returning ModelResponse with content length: {len(content)}, "
                    f"input_tokens: {token_usage.input_tokens}, "
                    f"output_tokens: {token_usage.output_tokens}")

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