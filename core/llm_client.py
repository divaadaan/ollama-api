"""LLM Client """

import json
import logging
import time
from typing import Optional, Dict, Any
import os
import requests

from telemetry import TelemetryConfig, SpanStatus

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client with optional telemetry support via dependency injection."""
    def __init__(
            self,
            api_url: str,
            default_model: str = "mistral",
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
            # Create a disabled telemetry config if none provided
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

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text using the provided model.

        Args:
            prompt: Text prompt for generation
            model: Model to use (defaults to default_model)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            Dictionary with 'response' key containing generated text, or 'error' key if failed
        """
        model = model or self.default_model

        # Start telemetry span (no-op if telemetry disabled)
        with self._tracer.start_span("llm_generate") as span:
            start_time = time.time()

            # Set span attributes (no-op if telemetry disabled)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.api_url", self.api_url)

            # Build request payload
            payload = {"model": model, "prompt": prompt, "stream": True}

            if "max_tokens" in kwargs and kwargs["max_tokens"]:
                payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]
                span.set_attribute("llm.max_tokens", kwargs["max_tokens"])

            if "temperature" in kwargs and kwargs["temperature"] is not None:
                payload.setdefault("options", {})["temperature"] = kwargs["temperature"]
                span.set_attribute("llm.temperature", kwargs["temperature"])

            try:
                logger.info(f"Sending request to {self.api_url} with model: {model}")
                response = requests.post(self.api_url, json=payload, stream=True, timeout=self.max_timeout)
                response.raise_for_status()

                # Process streaming response
                output = ""
                token_count = 0

                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        try:
                            data = json.loads(chunk)
                            if "response" in data:
                                output += data["response"]
                                token_count += len(data["response"].split())
                            elif "error" in data:
                                error_msg = data["error"]
                                logger.error(f"LLM API error: {error_msg}")

                                # Record error in telemetry (no-op if disabled)
                                span.set_status(SpanStatus.ERROR, error_msg)
                                self._request_counter.add(1, {"model": model, "status": "error"})

                                return {"error": error_msg}
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON: {chunk}")
                            continue

                # Record successful completion
                duration = time.time() - start_time

                # All telemetry calls are no-ops if telemetry is disabled
                self._request_counter.add(1, {"model": model, "status": "success"})
                self._response_time.record(duration, {"model": model})
                self._token_counter.add(token_count, {"model": model, "type": "output"})
                self._token_counter.add(len(prompt.split()), {"model": model, "type": "input"})

                # Set successful span attributes
                span.set_attribute("llm.response_length", len(output))
                span.set_attribute("llm.token_count", token_count)
                span.set_attribute("llm.duration_seconds", duration)
                span.set_status(SpanStatus.OK)

                logger.info(f"Successfully generated {len(output)} characters in {duration:.2f}s")
                return {"response": output.strip(), "model": model}

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

    def health_check(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a quick health check of the LLM API.

        Args:
            timeout: Timeout for health check request - should be set by .env

        Returns:
            Dictionary with health status information
        """
        if timeout is None:
            timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))

        with self._tracer.start_span("llm_health_check") as span:
            span.set_attribute("llm.api_url", self.api_url)
            span.set_attribute("llm.timeout", timeout)

            try:
                result = self.generate("Hello", max_tokens=1)
                logger.info(f"Reponse from model to Hello {result}")
                if "error" not in result and result.get("response"):
                    span.set_status(SpanStatus.OK)
                    return {
                        "status": "healthy",
                        "api_url": self.api_url,
                        "default_model": self.default_model,
                        "response_preview": result.get("response", "")[:50]
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