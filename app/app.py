"""This module provides a FastAPI application that interacts with a local LLM API
Dependencies:
- """
import json
import os
import logging
import sys
import time
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import gradio as gr
from smolagents import CodeAgent, DuckDuckGoSearchTool, Model

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from prometheus_client import start_http_server

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add console exporter for development (optional)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Initialize metrics with Prometheus exporter
metric_reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
meter = metrics.get_meter(__name__)

# Custom metrics
llm_request_counter = meter.create_counter(
    "llm_requests_total",
    description="Total number of LLM requests"
)
llm_response_time = meter.create_histogram(
    "llm_response_time_seconds",
    description="LLM response time in seconds"
)
llm_token_counter = meter.create_counter(
    "llm_tokens_total",
    description="Total tokens processed"
)
agent_request_counter = meter.create_counter(
    "agent_requests_total",
    description="Total agent requests"
)
agent_response_time = meter.create_histogram(
    "agent_response_time_seconds",
    description="Agent response time in seconds"
)

# FastAPI instantiation
app = FastAPI(
    title="Local LLM API",
    description="FastAPI application for interacting with local LLM models",
    version="1.0.0"
)

# Auto-instrument FastAPI and requests
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

class PromptRequest(BaseModel):
    """Request model for POST endpoints"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class LLMClient:
    """Class to abstract over LLM interfaces"""
    def __init__(self, api_url: str, default_model: str = "mistral"):
        self.api_url = api_url
        self.default_model = default_model

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text using the provided model."""
        model = model or self.default_model

        # Start OpenTelemetry span
        with tracer.start_as_current_span("llm_generate") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.api_url", self.api_url)

            payload = {"model": model, "prompt": prompt, "stream": True}
            if "max_tokens" in kwargs and kwargs["max_tokens"]:
                payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]
                span.set_attribute("llm.max_tokens", kwargs["max_tokens"])
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                payload.setdefault("options", {})["temperature"] = kwargs["temperature"]
                span.set_attribute("llm.temperature", kwargs["temperature"])

            try:
                logger.info(f"Sending request to {self.api_url} with model: {model}")
                response = requests.post(self.api_url, json=payload, stream=True, timeout=30)
                response.raise_for_status()

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
                                logger.error(f"LLM API error: {data['error']}")
                                span.set_status(trace.Status(trace.StatusCode.ERROR, data['error']))
                                llm_request_counter.add(1, {"model": model, "status": "error"})
                                return {"error": data["error"]}
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON: {chunk}")
                            continue

                # Record successful metrics
                duration = time.time() - start_time
                llm_request_counter.add(1, {"model": model, "status": "success"})
                llm_response_time.record(duration, {"model": model})
                llm_token_counter.add(token_count, {"model": model, "type": "output"})
                llm_token_counter.add(len(prompt.split()), {"model": model, "type": "input"})

                # Set span attributes for response
                span.set_attribute("llm.response_length", len(output))
                span.set_attribute("llm.token_count", token_count)
                span.set_attribute("llm.duration_seconds", duration)
                span.set_status(trace.Status(trace.StatusCode.OK))

                return {"response": output.strip(), "model": model}

            except requests.exceptions.Timeout:
                logger.error("Request timed out")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Request timed out"))
                llm_request_counter.add(1, {"model": model, "status": "timeout"})
                return {"error": "Request timed out"}
            except requests.exceptions.ConnectionError:
                logger.error("Failed to connect to LLM API")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Connection error"))
                llm_request_counter.add(1, {"model": model, "status": "connection_error"})
                return {"error": "Failed to connect to LLM API"}
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e.response.status_code}")
                span.set_status(trace.Status(trace.StatusCode.ERROR, f"HTTP {e.response.status_code}"))
                llm_request_counter.add(1, {"model": model, "status": "http_error"})
                return {"error": f"HTTP error: {e.response.status_code}"}
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                llm_request_counter.add(1, {"model": model, "status": "request_error"})
                return {"error": f"Request failed: {str(e)}"}


class LLMClientAdapter(Model):
    def __init__(self, client: LLMClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def __call__(self, messages: list[dict], **kwargs) -> dict:
        prompt = self._format_messages(messages)
        result = self.client.generate(prompt, **kwargs)
        return {"content": result.get("response", "")}

    def generate(self, messages: list[dict], stop_sequences: list[str] = None, **kwargs):
        prompt = self._format_messages(messages)
        result = self.client.generate(prompt, stop_sequences=stop_sequences, **kwargs)
        return type('Response', (object,), {'content': result.get("response", "")})()

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert list of message dictionaries to a single prompt string."""
        formatted_parts = []
        for message in messages:
            if isinstance(message, dict):
                # Handle different message formats
                if "content" in message:
                    content = message["content"]
                    role = message.get("role", "")
                    if role:
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

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=LLMClientAdapter(llm_client))
        SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and
                finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
                YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
                list of numbers and/or strings.
                If you are asked for a number, don't use comma to write your number neither use units such as $ or
                percent sign unless specified otherwise.
                If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the
                digits in plain text unless specified otherwise.
                If you are asked for a comma separated list, apply the above rules depending of whether the element
                to be put in the list is a number or a string.
                """
        self.agent.prompt_templates["system_prompt"] += SYSTEM_PROMPT

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # Add telemetry for agent calls
        with tracer.start_as_current_span("agent_run") as span:
            start_time = time.time()
            span.set_attribute("agent.question_length", len(question))

            try:
                final_answer = self.agent.run(question)

                # Record successful agent metrics
                duration = time.time() - start_time
                agent_request_counter.add(1, {"status": "success"})
                agent_response_time.record(duration)

                span.set_attribute("agent.response_length", len(final_answer))
                span.set_attribute("agent.duration_seconds", duration)
                span.set_status(trace.Status(trace.StatusCode.OK))

                print(f"Agent returning final answer: {final_answer}")
                return final_answer

            except Exception as e:
                # Record agent errors
                agent_request_counter.add(1, {"status": "error"})
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.error(f"Agent error: {str(e)}")
                raise

def check_ollama_health(client: LLMClient, timeout: int = 30, retry_interval: int = 5) -> bool:
    """Check if Ollama is running and responding to requests."""
    logger.info(f"Checking Ollama health at {client.api_url}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Send a simple test prompt
            test_result = client.generate("Hello", max_tokens=5)

            if "error" not in test_result and test_result.get("response"):
                logger.info("✅ Ollama health check passed")
                return True
            else:
                logger.warning(f"Ollama returned error: {test_result.get('error', 'No response')}")

        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")

        logger.info(f"Retrying Ollama health check in {retry_interval} seconds...")
        time.sleep(retry_interval)

    return False

def startup_health_checks():
    """Perform all startup health checks."""
    logger.info("Starting application health checks...")

    # Check Ollama
    if not check_ollama_health(llm_client):
        logger.error("Ollama health check failed - is Ollama running and model loaded?")
        logger.error(f"Expected Ollama at: {llm_client.api_url}")
        logger.error("Try: ollama serve && ollama pull mistral")
        sys.exit(1)

    logger.info("✅ All health checks passed - starting application")

llm_client = LLMClient(
    api_url=os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate"),
    default_model=os.getenv("DEFAULT_MODEL", "mistral")
)

basic_agent = BasicAgent()

startup_health_checks()

@app.get("/")
def root():
    return {"message": "Local LLM API is running", "endpoints": {"/ask": "GET", "/generate": "POST", "/health": "GET", "/metrics": "GET"}}

@app.get("/health")
def health_check():
    return {"status": "healthy", "api_url": llm_client.api_url}

@app.get("/metrics")
def metrics_endpoint():
    """Redirect to Prometheus metrics endpoint"""
    return {"message": "Metrics available at :8001/metrics"}

@app.get("/ask")
def ask(
    prompt: str = Query(...),
    model: Optional[str] = Query(None),
    max_tokens: Optional[int] = Query(None),
    temperature: Optional[float] = Query(None)
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    result = llm_client.generate(prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/generate")
def generate(request: PromptRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    result = llm_client.generate(
        prompt=request.prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/code-agent")
def run_agent(request: PromptRequest):
    result = basic_agent(request.prompt)
    return {"response": result}

# Start Prometheus metrics server on a separate port
@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server on port 8001
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")
    logger.info("Metrics available at http://localhost:8001/metrics")

if __name__ == "__main__":
    def run_agent_ui(prompt: str) -> str:
        return basic_agent(prompt)

    iface = gr.Interface(fn=run_agent_ui, inputs="text", outputs="text")
    iface.launch()