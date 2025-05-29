"""FastAPI application with modular telemetry support."""

import logging
import os
import sys
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from core.llm_client import create_llm_client
from core.agents import create_basic_agent

from telemetry import get_telemetry_config, is_telemetry_available

# Configure logging and environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
load_dotenv()

class PromptRequest(BaseModel):
    """Request model for POST endpoints"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

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


async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown logic."""
    logger.info("🚀 Starting FastAPI application")
    initialize_clients()
    startup_health_checks()

    if telemetry_config.enabled and is_telemetry_available():
        try:
            provider = telemetry_config.provider

            if hasattr(provider, 'start_metrics_server'): #specific to OpenTelemetry?
                provider.start_metrics_server()
                logger.info("✅ Metrics server started")
            else:
                logger.warning("Provider does not support metrics server")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

    logger.info("🎉 Application startup complete")

    yield

    logger.info("🛑 Application shutting down")

#get telemetry config
telemetry_config = get_telemetry_config()

app = FastAPI(
    title="Local LLM API",
    description="FastAPI application for interacting with local LLM models with optional telemetry",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument FastAPI if telemetry is enabled
if telemetry_config.enabled and is_telemetry_available():
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("✅ FastAPI instrumented with OpenTelemetry")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")


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
            "/health/simple": "GET - Simple health check - no agent execution",
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

@app.get("/health/simple")
def simple_health_check():
    """Simple health check without agent execution."""
    try:
        return {
            "status": "healthy",
            "components": {
                "llm_client_initialized": llm_client is not None,
                "agent_initialized": basic_agent is not None,
                "ollama_url": llm_client.api_url if llm_client else None,
                "default_model": llm_client.default_model if llm_client else None
            },
            "telemetry": {
                "enabled": telemetry_config.enabled if telemetry_config else False,
                "available": is_telemetry_available()
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/telemetry")
def telemetry_status():
    """Get telemetry configuration and status."""
    info = {
        "enabled": telemetry_config.enabled,
        "available": is_telemetry_available(),
        "config_source": "environment" if os.getenv("TELEMETRY_ENABLED") else "default"
    }

    if telemetry_config:
        info.update({
            "current_config_enabled": telemetry_config.enabled,
            "llm_client_telemetry": llm_client.telemetry_enabled if llm_client else None,
            "agent_telemetry": basic_agent.telemetry_enabled if basic_agent else None
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
    """ GET endpoint for LLM queries."""
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


# Gradio interface
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

    iface = gr.Interface(
        fn=run_agent_ui,
        inputs="text",
        outputs="text",
        title="LLM Agent Interface",
        description="Test the AI agent with your questions"
    )
    iface.launch()