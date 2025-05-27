"""FastAPI application with modular telemetry support."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

# Core functionality imports - no telemetry dependencies
from core.llm_client import create_llm_client
from core.agents import create_basic_agent

# Telemetry imports - gracefully handle missing dependencies
from telemetry import get_telemetry_config, is_telemetry_available

# Conditionally import quick_setup only if available
try:
    from telemetry import quick_setup
except ImportError:
    quick_setup = None

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# Request models
class PromptRequest(BaseModel):
    """Request model for POST endpoints"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# Global variables for clients (initialized in lifespan)
llm_client = None
basic_agent = None
telemetry_config = None

def startup_health_checks():
    """Perform all startup health checks."""
    logger.info("Starting application health checks...")

    startup_timeout = int(os.getenv("STARTUP_TIMEOUT", "120"))

    # Check LLM client
    health_result = llm_client.health_check()
    if health_result["status"] != "healthy":
        logger.error("LLM client health check failed")
        logger.error(f"Expected LLM API at: {llm_client.api_url}")
        logger.error("Try: ollama serve && ollama pull mistral")
        logger.error(f"Health check result: {health_result}")
        sys.exit(1)

    # Check agent
    agent_health = basic_agent.health_check()
    if agent_health["status"] != "healthy":
        logger.error("Agent health check failed")
        logger.error(f"Agent health result: {agent_health}")
        sys.exit(1)

    logger.info("✅ All health checks passed - starting application")

def initialize_telemetry(app: FastAPI) -> None:
    """
    Initialize telemetry based on configuration.

    This function encapsulates all telemetry setup, making it easy to
    conditionally enable/disable monitoring.
    """
    global telemetry_config

    # Get telemetry configuration from environment
    telemetry_config = get_telemetry_config()

    if telemetry_config.enabled:
        if is_telemetry_available():
            logger.info("Initializing telemetry with OpenTelemetry")

            # Set up telemetry with FastAPI instrumentation
            if quick_setup:
                quick_setup(
                    app=app,
                    enable_console_export=os.getenv("TELEMETRY_CONSOLE_EXPORT", "false").lower() == "true",
                    metrics_port=int(os.getenv("TELEMETRY_METRICS_PORT", "8001")),
                    start_metrics_server=True
                )
                logger.info("✅ Telemetry initialized successfully")
            else:
                logger.info("✅ disabled - quick_setup not available")
        else:
            logger.warning("Telemetry enabled but OpenTelemetry dependencies not available")
            logger.warning("Install with: pip install opentelemetry-api opentelemetry-sdk ...")
            # telemetry_config will use null implementations
    else:
        logger.info("Telemetry disabled via configuration")


def initialize_clients() -> None:
    """Initialize LLM client and agent with shared telemetry configuration."""
    global llm_client, basic_agent

    # Create LLM client with telemetry
    llm_client = create_llm_client(
        api_url=os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate"),
        default_model=os.getenv("DEFAULT_MODEL", "mistral"),
        telemetry_config=telemetry_config
    )

    # Create agent with same telemetry configuration
    basic_agent = create_basic_agent(
        llm_client=llm_client,
        telemetry_config=telemetry_config
    )

    logger.info("✅ Clients initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown logic."""

    # Startup
    logger.info("🚀 Starting FastAPI application")

    # Initialize telemetry first
    initialize_telemetry(app)

    # Initialize clients
    initialize_clients()

    # Run health checks
    startup_health_checks()

    logger.info("🎉 Application startup complete")

    yield

    # Shutdown
    logger.info("🛑 Application shutting down")
    # Could add cleanup logic here if needed


# FastAPI app with lifespan
app = FastAPI(
    title="Local LLM API",
    description="FastAPI application for interacting with local LLM models with optional telemetry",
    version="1.0.0",
    lifespan=lifespan
)


# API Endpoints
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Local LLM API is running",
        "endpoints": {
            "/ask": "GET - Simple question endpoint",
            "/generate": "POST - Text generation endpoint",
            "/code-agent": "POST - AI agent endpoint",
            "/health": "GET - Health check",
            "/metrics": "GET - Metrics information",
            "/telemetry": "GET - Telemetry status"
        },
        "telemetry_enabled": telemetry_config.enabled if telemetry_config else False
    }


@app.get("/health")
def health_check():
    """Comprehensive health check endpoint."""
    llm_health = llm_client.health_check()
    agent_health = basic_agent.health_check()

    overall_status = "healthy" if (
        llm_health["status"] == "healthy" and
        agent_health["status"] == "healthy"
    ) else "unhealthy"

    return {
        "status": overall_status,
        "llm_client": llm_health,
        "agent": agent_health,
        "telemetry": {
            "enabled": telemetry_config.enabled,
            "available": is_telemetry_available()
        }
    }


@app.get("/telemetry")
def telemetry_status():
    """Get telemetry configuration and status."""
    from telemetry import get_telemetry_info

    info = get_telemetry_info()
    info.update({
        "current_config_enabled": telemetry_config.enabled,
        "llm_client_telemetry": llm_client.telemetry_enabled,
        "agent_telemetry": basic_agent.telemetry_enabled
    })

    return info


@app.get("/metrics")
def metrics_endpoint():
    """Metrics endpoint information."""
    if telemetry_config.enabled and is_telemetry_available():
        metrics_port = os.getenv("TELEMETRY_METRICS_PORT", "8001")
        return {
            "message": f"Metrics available at http://localhost:{metrics_port}/metrics",
            "prometheus_port": metrics_port,
            "telemetry_enabled": True
        }
    else:
        return {
            "message": "Metrics not available - telemetry disabled or OpenTelemetry not installed",
            "telemetry_enabled": False
        }


@app.get("/ask")
def ask(
    prompt: str = Query(..., description="Text prompt for the LLM"),
    model: Optional[str] = Query(None, description="Model to use"),
    max_tokens: Optional[int] = Query(None, description="Maximum tokens to generate"),
    temperature: Optional[float] = Query(None, description="Temperature for generation")
):
    """Simple GET endpoint for LLM queries."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = llm_client.generate(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/generate")
def generate(request: PromptRequest):
    """POST endpoint for LLM text generation."""
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
    """POST endpoint for AI agent execution."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        result = basic_agent(request.prompt)
        return {"response": result}
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.get("/stats")
def get_stats():
    """Get runtime statistics about the application."""
    return {
        "llm_client": {
            "api_url": llm_client.api_url,
            "default_model": llm_client.default_model,
            "telemetry_enabled": llm_client.telemetry_enabled
        },
        "agent": basic_agent.get_stats(),
        "telemetry": {
            "enabled": telemetry_config.enabled,
            "available": is_telemetry_available(),
            "metrics_port": os.getenv("TELEMETRY_METRICS_PORT", "8001")
        }
    }


# Gradio interface for development/testing
def run_agent_ui(prompt: str) -> str:
    """Gradio interface function for agent testing."""
    try:
        return basic_agent(prompt)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # For development - run Gradio interface
    import uvicorn

    # Initialize everything manually for standalone script
    telemetry_config = get_telemetry_config()
    initialize_clients()

    # Option 1: Run Gradio interface
    iface = gr.Interface(
        fn=run_agent_ui,
        inputs="text",
        outputs="text",
        title="LLM Agent Interface",
        description="Test the AI agent with your questions"
    )
    iface.launch()

    # Option 2: Run FastAPI server
    # uvicorn.run(app, host="0.0.0.0", port=8000)