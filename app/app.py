"""
This module provides a FastAPI application that interacts with a local LLM API

Dependencies:
- requests
- dotenv
- fastapi
"""
import json
import os
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, Query

load_dotenv()
app = FastAPI()

class LLMClient:
    """
    Class to abstract over LLM interfaces
    """
    def __init__(self, api_url, default_model="mistral"):
        self.api_url = api_url
        self.default_model = default_model

    def generate(self, prompt, model=None):
        """Generate text using the specified LLM model."""
        model = model or self.default_model
        try:
            response = requests.post(
                self.api_url,
                json={"model": model, "prompt": prompt},
                stream=True,
                timeout=10
            )
            output = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    try:
                        data = json.loads(chunk)
                        output += data.get("response", "")
                    except json.JSONDecodeError:
                        pass  # optionally log the line or raise an error

            return {"response": output}
        except requests.exceptions.Timeout:
            return {"error": "response timed out"}
        except requests.exceptions.RequestException as error_code:
            return {"error": str(error_code)}

llm_client = LLMClient(
    api_url=os.getenv("LLM_API_URL", "http://host.docker.internal:11434/api/generate"),
    default_model=os.getenv("DEFAULT_MODEL", "mistral")
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/ask")
def ask(prompt: str = Query(..., description="Prompt to send to the model")):
    """
    Prompts the local model
    :param prompt:
    :return: dict model response
    """
    return llm_client.generate(prompt)
