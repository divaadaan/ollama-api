"""
LLMClientAdapter class for abstracting to local LLM ollama instance
"""

import logging
import time
from typing import Optional

from smolagents import CodeAgent, DuckDuckGoSearchTool, Model

from telemetry import TelemetryConfig, SpanStatus

logger = logging.getLogger(__name__)


class LLMClientAdapter(Model):
    """
    Adapter to make LLMClient compatible with smolagents Model interface.

    This adapter also handles telemetry forwarding from the agent to the LLM client.
    """

    def __init__(self, llm_client, **kwargs):
        """
        Initialize adapter with LLM client.

        Args:
            llm_client: LLMClient instance (with or without telemetry)
            **kwargs: Additional Model parameters
        """
        super().__init__(**kwargs)
        self.llm_client = llm_client

    def __call__(self, messages: list[dict], **kwargs) -> dict:
        """Call interface for smolagents compatibility."""
        prompt = self._format_messages(messages)
        result = self.llm_client.generate(prompt, **kwargs)
        return {"content": result.get("response", "")}

    def generate(self, messages: list[dict], stop_sequences: list[str] = None, **kwargs):
        """Generate interface for smolagents compatibility."""
        prompt = self._format_messages(messages)
        result = self.llm_client.generate(prompt, stop_sequences=stop_sequences, **kwargs)
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

    @property
    def telemetry_enabled(self) -> bool:
        """Check if the underlying LLM client has telemetry enabled."""
        return getattr(self.llm_client, 'telemetry_enabled', False)


class BasicAgent:
    """
    AI Agent with optional telemetry support via dependency injection.

    The agent works identically whether telemetry is enabled or disabled,
    with zero performance overhead when telemetry is off.
    """

    def __init__(
            self,
            llm_client,
            telemetry: Optional[TelemetryConfig] = None,
            tools: Optional[list] = None
    ):
        """
        Initialize BasicAgent.

        Args:
            llm_client: LLMClient instance for language model access
            telemetry: Optional telemetry configuration. If None, uses llm_client's telemetry
            tools: Optional list of tools for the agent. Defaults to DuckDuckGoSearchTool
        """
        self.llm_client = llm_client

        # Debug logging
        #logger.info(f"LLM client telemetry enabled: {getattr(llm_client, 'telemetry_enabled', 'unknown')}")

        # Set up telemetry - inherit from LLM client if not provided
        if telemetry is None:
            if hasattr(llm_client, '_telemetry'):
                telemetry = llm_client._telemetry
            else:
                # Fall back to disabled telemetry
                from telemetry import TelemetryConfig
                telemetry = TelemetryConfig(enabled=False)

        self._telemetry = telemetry
        #logger.info(f"BasicAgent final telemetry enabled: {telemetry.enabled}")

        self._tracer = telemetry.tracer

        # Pre-fetch metrics for performance
        self._request_counter = telemetry.agent_request_counter
        self._response_time = telemetry.agent_response_time

        # Set up tools
        if tools is None:
            tools = [DuckDuckGoSearchTool()]

        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(
            tools=tools,
            model=LLMClientAdapter(llm_client)
        )

        # Configure system prompt
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
        self.agent.system_prompt += SYSTEM_PROMPT

        #logger.info(f"BasicAgent initialized with telemetry {'enabled' if telemetry.enabled else 'disabled'}")

    def __call__(self, question: str) -> str:
        """
        Execute agent on a question.

        Args:
            question: Question or task for the agent to process

        Returns:
            Agent's response/answer
        """
        logger.info(f"Agent received question (first 50 chars): {question[:50]}...")

        # Start telemetry span (no-op if telemetry disabled)
        with self._tracer.start_span("agent_run") as span:
            start_time = time.time()

            # Set span attributes (no-op if telemetry disabled)
            span.set_attribute("agent.question_length", len(question))
            span.set_attribute("agent.tools_count", len(self.agent.tools))
            span.set_attribute("agent.model_telemetry", self.llm_client.telemetry_enabled)

            try:
                # Execute the agent
                final_answer = self.agent.run(question)

                # Record successful completion
                duration = time.time() - start_time

                # All telemetry calls are no-ops if telemetry is disabled
                self._request_counter.add(1, {"status": "success"})
                self._response_time.record(duration)

                # Set successful span attributes
                span.set_attribute("agent.response_length", len(str(final_answer)))
                span.set_attribute("agent.duration_seconds", duration)
                span.set_status(SpanStatus.OK)

                logger.info(f"Agent completed successfully in {duration:.2f}s")
                logger.info(f"Agent returning final answer: {final_answer}")
                return final_answer

            except Exception as e:
                # Record agent errors
                duration = time.time() - start_time
                error_msg = f"Agent execution failed: {str(e)}"

                # Telemetry for errors (no-op if disabled)
                self._request_counter.add(1, {"status": "error"})
                span.set_attribute("agent.duration_seconds", duration)
                span.set_attribute("agent.error_message", str(e))
                span.set_status(SpanStatus.ERROR, error_msg)

                logger.error(error_msg)
                raise

    def run(self, question: str) -> str:
        """Alias for __call__ method for explicit usage."""
        return self.__call__(question)

    def health_check(self) -> dict:
        """
        Perform a health check of the agent and its dependencies.

        Returns:
            Dictionary with health status information
        """
        with self._tracer.start_span("agent_health_check") as span:
            try:
                # Check LLM client health
                llm_health = self.llm_client.health_check()

                # Quick agent test
                test_result = self("What is 2+2?")

                span.set_status(SpanStatus.OK)
                return {
                    "status": "healthy",
                    "llm_client_status": llm_health.get("status", "unknown"),
                    "tools_count": len(self.agent.tools),
                    "telemetry_enabled": self._telemetry.enabled,
                    "test_response_length": len(test_result)
                }

            except Exception as e:
                error_msg = f"Agent health check failed: {str(e)}"
                span.set_status(SpanStatus.ERROR, error_msg)
                return {
                    "status": "unhealthy",
                    "error": error_msg,
                    "telemetry_enabled": self._telemetry.enabled
                }

    @property
    def telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled for this agent."""
        return self._telemetry.enabled

    def get_stats(self) -> dict:
        """
        Get runtime statistics about the agent.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "tools": [tool.__class__.__name__ for tool in self.agent.tools],
            "model_adapter": self.agent.model.__class__.__name__,
            "llm_client_type": self.llm_client.__class__.__name__,
            "telemetry_enabled": self.telemetry_enabled,
            "llm_telemetry_enabled": self.llm_client.telemetry_enabled
        }

# Factory function for easy creation
def create_basic_agent(
        llm_client,
        tools: Optional[list] = None,
        telemetry_enabled: Optional[bool] = None,
        telemetry_config: Optional[TelemetryConfig] = None
) -> BasicAgent:
    """
    Factory function to create BasicAgent with optional telemetry.

    Args:
        llm_client: LLMClient instance
        tools: Optional list of tools for the agent
        telemetry_enabled: Whether to enable telemetry (overrides llm_client setting)
        telemetry_config: Pre-configured telemetry, overrides other telemetry settings

    Returns:
        Configured BasicAgent instance
    """
    if telemetry_config is None:
        if telemetry_enabled is not None:
            from telemetry import get_telemetry_config
            telemetry_config = get_telemetry_config(enabled=telemetry_enabled)
        # Otherwise, inherit from llm_client (handled in BasicAgent.__init__)

    return BasicAgent(
        llm_client=llm_client,
        telemetry=telemetry_config,
        tools=tools
    )