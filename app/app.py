"""
This module provides a FastAPI application that interacts with a local LLM API

Dependencies:
- requests
- dotenv
- fastapi
- uvicorn
"""
import json
import os
import requests
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(
    title="Local LLM API",
    description="FastAPI application for interacting with local LLM models",
    version="1.0.0"
)

class PromptRequest(BaseModel):
    """Request model for POST endpoints"""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class LLMClient:
    """
    Class to abstract over LLM interfaces
    """
    def __init__(self, api_url: str, default_model: str = "mistral"):
        self.api_url = api_url
        self.default_model = default_model

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text using the specified LLM model."""
        model = model or self.default_model

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }

        # Add optional parameters
        if "max_tokens" in kwargs and kwargs["max_tokens"]:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = kwargs["max_tokens"]

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = kwargs["temperature"]

        try:
            logger.info(f"Sending request to {self.api_url} with model: {model}")
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=30  # Increased timeout for longer responses
            )
            response.raise_for_status()  # Raise exception for HTTP errors

            output = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    try:
                        data = json.loads(chunk)
                        if "response" in data:
                            output += data["response"]
                        elif "error" in data:
                            logger.error(f"LLM API error: {data['error']}")
                            return {"error": data["error"]}
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON: {chunk}")
                        continue

            return {"response": output.strip(), "model": model}

        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to LLM API"
            logger.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e.response.status_code}"
            logger.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

# Initialize LLM client
llm_client = LLMClient(
    api_url=os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate"),
    default_model=os.getenv("DEFAULT_MODEL", "mistral")
)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Local LLM API is running",
        "endpoints": {
            "/ask": "GET - Send a prompt via query parameter",
            "/generate": "POST - Send a prompt via request body",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "api_url": llm_client.api_url}

@app.get("/ask")
def ask(
    prompt: str = Query(..., description="Prompt to send to the model"),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    max_tokens: Optional[int] = Query(None, description="Maximum tokens to generate"),
    temperature: Optional[float] = Query(None, description="Temperature for generation (0.0-1.0)")
):
    """
    Send a prompt to the local LLM model via GET request

    Args:
        prompt: The text prompt to send to the model
        model: Optional model name to use
        max_tokens: Optional maximum number of tokens to generate
        temperature: Optional temperature for text generation

    Returns:
        dict: Model response or error message
    """
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
    """
    Send a prompt to the local LLM model via POST request

    Args:
        request: PromptRequest containing prompt and optional parameters

    Returns:
        dict: Model response or error message
    """
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

# The Dockerfile uses: uvicorn app.app:app --host 0.0.0.0 --port 8000