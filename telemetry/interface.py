"""Telemetry abstraction interface for optional monitoring support."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from enum import Enum


class SpanStatus(Enum):
    """Span status enumeration"""
    OK = "ok"
    ERROR = "error"


@runtime_checkable
class TelemetrySpan(Protocol):
    """Protocol for telemetry spans"""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span"""
        ...

    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        """Set the status of the span"""
        ...


@runtime_checkable
class TelemetryTracer(Protocol):
    """Protocol for telemetry tracers"""

    @contextmanager
    def start_span(self, name: str) -> TelemetrySpan:
        """Start a new span context manager"""
        ...


@runtime_checkable
class TelemetryCounter(Protocol):
    """Protocol for telemetry counters"""

    def add(self, amount: int = 1, attributes: Optional[Dict[str, str]] = None) -> None:
        """Add to the counter"""
        ...


@runtime_checkable
class TelemetryHistogram(Protocol):
    """Protocol for telemetry histograms"""

    def record(self, amount: float, attributes: Optional[Dict[str, str]] = None) -> None:
        """Record a value in the histogram"""
        ...


@runtime_checkable
class TelemetryProvider(Protocol):
    """Protocol for the main telemetry provider"""

    def get_tracer(self, name: str) -> TelemetryTracer:
        """Get a tracer instance"""
        ...

    def create_counter(self, name: str, description: str = "") -> TelemetryCounter:
        """Create a counter metric"""
        ...

    def create_histogram(self, name: str, description: str = "") -> TelemetryHistogram:
        """Create a histogram metric"""
        ...


# Null implementations for when telemetry is disabled

class NullSpan:
    """No-op implementation of TelemetrySpan"""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        pass


class NullTracer:
    """No-op implementation of TelemetryTracer"""

    @contextmanager
    def start_span(self, name: str) -> TelemetrySpan:
        yield NullSpan()


class NullCounter:
    """No-op implementation of TelemetryCounter"""

    def add(self, amount: int = 1, attributes: Optional[Dict[str, str]] = None) -> None:
        pass


class NullHistogram:
    """No-op implementation of TelemetryHistogram"""

    def record(self, amount: float, attributes: Optional[Dict[str, str]] = None) -> None:
        pass


class NullTelemetryProvider:
    """No-op implementation of TelemetryProvider"""

    def get_tracer(self, name: str) -> TelemetryTracer:
        return NullTracer()

    def create_counter(self, name: str, description: str = "") -> TelemetryCounter:
        return NullCounter()

    def create_histogram(self, name: str, description: str = "") -> TelemetryHistogram:
        return NullHistogram()


# Factory function for creating telemetry providers

def create_telemetry_provider(enabled: bool = True) -> TelemetryProvider:
    """
    Factory function to create appropriate telemetry provider.

    Args:
        enabled: Whether to enable real telemetry or use null implementation

    Returns:
        TelemetryProvider instance (real or null implementation)
    """
    if not enabled:
        return NullTelemetryProvider()

    # Import OpenTelemetry only when needed
    try:
        from .opentelemetry_provider import OpenTelemetryProvider
        return OpenTelemetryProvider()
    except ImportError as e:
        import logging
        logging.warning(f"OpenTelemetry not available, using null provider: {e}")
        return NullTelemetryProvider()


# Convenience class for dependency injection

class TelemetryConfig:
    """Configuration and factory for telemetry components"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._provider = create_telemetry_provider(enabled)

        # Pre-create commonly used components
        self.tracer = self._provider.get_tracer(__name__)

        # Common metrics
        self.llm_request_counter = self._provider.create_counter(
            "llm_requests_total",
            "Total number of LLM requests"
        )
        self.llm_response_time = self._provider.create_histogram(
            "llm_response_time_seconds",
            "LLM response time in seconds"
        )
        self.llm_token_counter = self._provider.create_counter(
            "llm_tokens_total",
            "Total tokens processed"
        )
        self.agent_request_counter = self._provider.create_counter(
            "agent_requests_total",
            "Total agent requests"
        )
        self.agent_response_time = self._provider.create_histogram(
            "agent_response_time_seconds",
            "Agent response time in seconds"
        )

    @property
    def provider(self) -> TelemetryProvider:
        return self._provider