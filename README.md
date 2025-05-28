You need ollama running locally serving a model on the default port, or another model. 
Make the specifications in .env

LLM_API_URL=http://host.docker.internal:11434/api/generate
DEFAULT_MODEL=qwen2.5-coder:14b

I have tried mistral7b but it struggles to adopt the required response format.
I will want to tre qwen7b too