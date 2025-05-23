'''''
This module provides a FastAPI application that interacts with a local LLM API

Dependencies:
- requests
- dotenv
- fastapi
- uvicorn
- gradio
- smolagents
'''''
import json
import os
import logging
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import gradio as gr
import pandas as pd
import inspect
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, Model

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# FastAPI instantiation
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
        payload = {"model": model, "prompt": prompt, "stream": True}
        if "max_tokens" in kwargs and kwargs["max_tokens"]:
            payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload.setdefault("options", {})["temperature"] = kwargs["temperature"]
        try:
            logger.info(f"Sending request to {self.api_url} with model: {model}")
            response = requests.post(self.api_url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
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
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON: {chunk}")
                        continue
            return {"response": output.strip(), "model": model}
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return {"error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to LLM API")
            return {"error": "Failed to connect to LLM API"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            return {"error": f"HTTP error: {e.response.status_code}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}

class LLMClientAdapter():
    def __init__(self, client: LLMClient):
        self.client = client

    def __call__(self, messages: list[dict], **kwargs) -> dict:
        prompt = "\n".join([m["content"] for m in messages])
        result = self.client.generate(prompt, **kwargs)
        return {"content": result.get("response", "")}

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Mimics the expected generate method for CodeAgent to work correctly.
        It uses the LLMClient to generate the response.
        """
        return self.client.generate(prompt, **kwargs)

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=LLMClientAdapter(llm_client))
        SYSTEM_PROMPT = """You are a general AI assistant..."""
        self.agent.prompt_templates["system_prompt"] += SYSTEM_PROMPT
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        final_answer = self.agent.run(question)
        print(f"Agent returning final answer: {final_answer}")
        return final_answer

llm_client = LLMClient(
    api_url=os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate"),
    default_model=os.getenv("DEFAULT_MODEL", "mistral")
)

basic_agent = BasicAgent()

@app.get("/")
def root():
    return {"message": "Local LLM API is running", "endpoints": {"/ask": "GET", "/generate": "POST", "/health": "GET"}}

@app.get("/health")
def health_check():
    return {"status": "healthy", "api_url": llm_client.api_url}

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
    # Note: ensure parameter name matches usage below
    result = basic_agent(request.prompt)
    return {"response": result}

if __name__ == "__main__":
    def run_agent_ui(prompt: str) -> str:
        return basic_agent(prompt)

    iface = gr.Interface(fn=run_agent_ui, inputs="text", outputs="text")
    iface.launch()
