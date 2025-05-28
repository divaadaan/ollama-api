"""
OpenTelemetry implementation of the telemetry abstraction interface.
This module is optionally imported depending on if telemetry is enabled
"""

from contextlib import contextmanager
from typing import Dict, Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from prometheus_client import start_http_server

from .interface import (
    TelemetryTracer,
    TelemetryCounter,
    TelemetryHistogram,
    SpanStatus
)


class OpenTelemetrySpan:
    """OpenTelemetry implementation of TelemetrySpan"""
    def __init__(self, otel_span):
        self._span = otel_span

    def set_attribute(self, key: str, value) -> None:
        self._span.set_attribute(key, value)

    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        if status == SpanStatus.OK:
            self._span.set_status(trace.Status(trace.StatusCode.OK, description))
        elif status == SpanStatus.ERROR:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, description))


class OpenTelemetryTracer:
    """OpenTelemetry implementation of TelemetryTracer"""
    def __init__(self, otel_tracer):
        self._tracer = otel_tracer

    @contextmanager
    def start_span(self, name: str):
        with self._tracer.start_as_current_span(name) as span:
            yield OpenTelemetrySpan(span)


class OpenTelemetryCounter:
    """OpenTelemetry implementation of TelemetryCounter"""
    def __init__(self, otel_counter):
        self._counter = otel_counter

    def add(self, amount: int = 1, attributes: Optional[Dict[str, str]] = None) -> None:
        self._counter.add(amount, attributes or {})


class OpenTelemetryHistogram:
    """OpenTelemetry implementation of TelemetryHistogram"""
    def __init__(self, otel_histogram):
        self._histogram = otel_histogram

    def record(self, amount: float, attributes: Optional[Dict[str, str]] = None) -> None:
        self._histogram.record(amount, attributes or {})


class OpenTelemetryProvider:
    """OpenTelemetry implementation of TelemetryProvider"""
    def __init__(self, enable_console_export: bool = False, metrics_port: int = 8001):
        """
        Initialize OpenTelemetry provider with full configuration.
        Args:
            enable_console_export: Whether to enable console span export for development
            metrics_port: Port for Prometheus metrics server
        """
        self._setup_tracing(enable_console_export)
        self._setup_metrics()
        self._setup_instrumentation()
        self._metrics_port = metrics_port
        self._metrics_server_started = False

    def _setup_tracing(self, enable_console_export: bool) -> None:
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())

        if enable_console_export:
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics with Prometheus export"""
        metric_reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        self._meter = metrics.get_meter(__name__)

    def _setup_instrumentation(self) -> None:
        """Setup automatic instrumentation for FastAPI and requests"""
        RequestsInstrumentor().instrument()
        # Note: FastAPI instrumentation is applied to the app instance separately

    def get_tracer(self, name: str) -> TelemetryTracer:
        """Get a tracer instance"""
        otel_tracer = trace.get_tracer(name)
        return OpenTelemetryTracer(otel_tracer)

    def create_counter(self, name: str, description: str = "") -> TelemetryCounter:
        """Create a counter metric"""
        otel_counter = self._meter.create_counter(name, description=description)
        return OpenTelemetryCounter(otel_counter)

    def create_histogram(self, name: str, description: str = "") -> TelemetryHistogram:
        """Create a histogram metric"""
        otel_histogram = self._meter.create_histogram(name, description=description)
        return OpenTelemetryHistogram(otel_histogram)

    def instrument_fastapi_app(self, app):
        """Instrument a FastAPI application"""
        FastAPIInstrumentor.instrument_app(app)

    def start_metrics_server(self) -> None:
        """Start the Prometheus metrics server"""
        if not self._metrics_server_started:
            start_http_server(self._metrics_port)
            self._metrics_server_started = True

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Prometheus metrics server started on port {self._metrics_port}")
            logger.info(f"Metrics available at http://localhost:{self._metrics_port}/metrics")


# Utility functions for easy setup
def setup_telemetry(
        app=None,
        enable_console_export: bool = False,
        metrics_port: int = 8001,
        start_metrics_server: bool = True
) -> OpenTelemetryProvider:
    """
    Setup OpenTelemetry with common configuration.
    Args:
        app: FastAPI application instance to instrument (optional)
        enable_console_export: Enable console span export for development
        metrics_port: Port for Prometheus metrics server
        start_metrics_server: Whether to start the metrics server immediately
    Returns:
        Configured OpenTelemetryProvider instance
    """
    provider = OpenTelemetryProvider(
        enable_console_export=enable_console_export,
        metrics_port=metrics_port
    )

    if app is not None:
        provider.instrument_fastapi_app(app)

    if start_metrics_server:
        provider.start_metrics_server()

    return provider