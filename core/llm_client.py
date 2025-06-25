"""LLM Client for Ollama communication."""
import json
import logging
import time
from typing import Optional, Dict, Any, Iterator
import os

import requests

from telemetry import TelemetryConfig, SpanStatus

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client for Ollama with optional telemetry support."""

    def __init__(
            self,
            api_url: str,
            default_model: str = "qwen2.5-coder:7b",
            telemetry: Optional[TelemetryConfig] = None
    ):
        """
        Initialize LLM client.
        Args:
            api_url: URL for the LLM API endpoint
            default_model: Default model to use for requests
            telemetry: Optional telemetry configuration. If None, telemetry is disabled.
        """
        self.api_url = api_url
        self.default_model = default_model

        # Store telemetry components
        if telemetry is None:
            from telemetry import TelemetryConfig
            telemetry = TelemetryConfig(enabled=False)

        self._telemetry = telemetry
        self._tracer = telemetry.tracer

        # Pre-fetch metrics for performance
        self._request_counter = telemetry.llm_request_counter
        self._response_time = telemetry.llm_response_time
        self._token_counter = telemetry.llm_token_counter

        self.max_timeout = int(os.getenv("MAX_TIMEOUT", "60"))

        logger.info(
            f"LLMClient initialized with telemetry {'enabled' if telemetry.enabled else 'disabled'}"
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the provided model.
        Args:
            prompt: Text prompt for generation
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional parameters
        Returns:
            Dictionary with response data:
            {
                "content": str,           # Generated text
                "model": str,            # Model used
                "prompt_tokens": int,    # Estimated input tokens
                "completion_tokens": int, # Estimated output tokens
                "total_tokens": int,     # Total tokens
                "finish_reason": str,    # Completion reason
                "error": str (optional)  # Error message if failed
            }
        """
        model = model or self.default_model

        with self._tracer.start_span("llm_generate") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.api_url", self.api_url)

            # Build request payload
            payload = {"model": model, "prompt": prompt, "stream": True}

            if max_tokens:
                payload.setdefault("options", {})["num_predict"] = max_tokens
                span.set_attribute("llm.max_tokens", max_tokens)

            if temperature is not None:
                payload.setdefault("options", {})["temperature"] = temperature
                span.set_attribute("llm.temperature", temperature)

            try:
                logger.debug(f"Sending request to {self.api_url} with model: {model}")
                response = requests.post(
                    self.api_url,
                    json=payload,
                    stream=True,
                    timeout=self.max_timeout
                )
                response.raise_for_status()

                # Process streaming response
                content = ""
                completion_tokens = 0

                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        try:
                            data = json.loads(chunk)
                            if "response" in data:
                                content += data["response"]
                                completion_tokens += len(data["response"].split())
                            elif "error" in data:
                                error_msg = data["error"]
                                logger.error(f"LLM API error: {error_msg}")
                                span.set_status(SpanStatus.ERROR, error_msg)
                                self._request_counter.add(1, {"model": model, "status": "error"})
                                return {"error": error_msg}
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON: {chunk}")
                            continue

                # Calculate token estimates
                prompt_tokens = len(prompt.split())
                total_tokens = prompt_tokens + completion_tokens
                duration = time.time() - start_time

                # Record telemetry
                self._request_counter.add(1, {"model": model, "status": "success"})
                self._response_time.record(duration, {"model": model})
                self._token_counter.add(completion_tokens, {"model": model, "type": "output"})
                self._token_counter.add(prompt_tokens, {"model": model, "type": "input"})

                # Set successful span attributes
                span.set_attribute("llm.response_length", len(content))
                span.set_attribute("llm.completion_tokens", completion_tokens)
                span.set_attribute("llm.prompt_tokens", prompt_tokens)
                span.set_attribute("llm.total_tokens", total_tokens)
                span.set_attribute("llm.duration_seconds", duration)
                span.set_status(SpanStatus.OK)

                logger.debug(f"Generated {len(content)} characters in {duration:.2f}s")

                return {
                    "content": content.strip(),
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "finish_reason": "stop"
                }

            except requests.exceptions.Timeout:
                error_msg = "Request timed out"
                logger.error(error_msg)
                span.set_status(SpanStatus.ERROR, error_msg)
                self._request_counter.add(1, {"model": model, "status": "timeout"})
                return {"error": error_msg}

            except requests.exceptions.ConnectionError:
                error_msg = "Failed to connect to LLM API"
                logger.error(error_msg)
                span.set_status(SpanStatus.ERROR, error_msg)
                self._request_counter.add(1, {"model": model, "status": "connection_error"})
                return {"error": error_msg}

            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error: {e.response.status_code}"
                logger.error(error_msg)
                span.set_status(SpanStatus.ERROR, error_msg)
                self._request_counter.add(1, {"model": model, "status": "http_error"})
                return {"error": error_msg}

            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                logger.error(error_msg)
                span.set_status(SpanStatus.ERROR, error_msg)
                self._request_counter.add(1, {"model": model, "status": "request_error"})
                return {"error": error_msg}

    def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming response.

        Args:
            prompt: Text prompt for generation
            model: Model to use (defaults to default_model)
            **kwargs: Additional parameters

        Yields:
            str: Streaming text chunks
        """
        model = model or self.default_model

        with self._tracer.start_span("llm_stream_generate") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))

            payload = {"model": model, "prompt": prompt, "stream": True}

            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    stream=True,
                    timeout=self.max_timeout
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        try:
                            data = json.loads(chunk)
                            if "response" in data:
                                yield data["response"]
                            elif "error" in data:
                                span.set_status(SpanStatus.ERROR, data["error"])
                                raise RuntimeError(f"LLM API error: {data['error']}")
                        except json.JSONDecodeError:
                            continue

                span.set_status(SpanStatus.OK)

            except Exception as e:
                span.set_status(SpanStatus.ERROR, str(e))
                raise

    def health_check(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a health check of the LLM API.

        Args:
            timeout: Timeout for health check request

        Returns:
            Dictionary with health status information
        """
        if timeout is None:
            timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))

        with self._tracer.start_span("llm_health_check") as span:
            span.set_attribute("llm.api_url", self.api_url)
            span.set_attribute("llm.timeout", timeout)

            try:
                # Store original timeout and temporarily change it
                original_timeout = self.max_timeout
                self.max_timeout = timeout

                result = self.generate("Hello", max_tokens=1)

                # Restore original timeout
                self.max_timeout = original_timeout

                if "error" not in result and result.get("content"):
                    span.set_status(SpanStatus.OK)
                    return {
                        "status": "healthy",
                        "api_url": self.api_url,
                        "default_model": self.default_model,
                        "response_preview": result.get("content", "")[:50]
                    }
                else:
                    span.set_status(SpanStatus.ERROR, "Health check failed")
                    return {
                        "status": "unhealthy",
                        "api_url": self.api_url,
                        "error": result.get("error", "No response")
                    }

            except Exception as e:
                error_msg = f"Health check exception: {str(e)}"
                span.set_status(SpanStatus.ERROR, error_msg)
                return {
                    "status": "unhealthy",
                    "api_url": self.api_url,
                    "error": error_msg
                }

    @property
    def telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled for this client."""
        return self._telemetry.enabled


# Factory function for easy creation
def create_llm_client(
        api_url: str,
        default_model: str = "mistral",
        telemetry_enabled: bool = True,
        telemetry_config: Optional[TelemetryConfig] = None
) -> LLMClient:
    """
    Factory function to create LLMClient with optional telemetry.

    Args:
        api_url: LLM API endpoint URL
        default_model: Default model name
        telemetry_enabled: Whether to enable telemetry
        telemetry_config: Pre-configured telemetry, overrides telemetry_enabled

    Returns:
        Configured LLMClient instance
    """
    if telemetry_config is None:
        from telemetry import get_telemetry_config
        telemetry_config = get_telemetry_config(enabled=telemetry_enabled)

    return LLMClient(
        api_url=api_url,
        default_model=default_model,
        telemetry=telemetry_config
    )