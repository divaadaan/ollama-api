"""AI Agent using the new architecture with clean separation of concerns."""

import logging
import time
from typing import Optional

from smolagents import CodeAgent, DuckDuckGoSearchTool

from telemetry import TelemetryConfig, SpanStatus
from .smolagents_adapter import SmolOllamaAdapter

logger = logging.getLogger(__name__)


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

        # Set up telemetry - inherit from LLM client if not provided
        if telemetry is None:
            if hasattr(llm_client, '_telemetry'):
                telemetry = llm_client._telemetry
            else:
                # Fall back to disabled telemetry
                from telemetry import TelemetryConfig
                telemetry = TelemetryConfig(enabled=False)

        self._telemetry = telemetry
        self._tracer = telemetry.tracer

        # Pre-fetch metrics for performance
        self._request_counter = telemetry.agent_request_counter
        self._response_time = telemetry.agent_response_time

        # Set up tools
        if tools is None:
            tools = [DuckDuckGoSearchTool()]

        # Create the smolagents adapter
        self.model_adapter = SmolOllamaAdapter(llm_client)

        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(
            tools=tools,
            model=self.model_adapter
        )

        # Configure system prompt
        SYSTEM_PROMPT = """
        ADDITIONAL INSTRUCTIONS for this specific agent:

        You are a general AI assistant. When solving tasks, follow this approach:
        1. Think through the problem step by step
        2. Use code to perform calculations or operations when needed
        3. Always call final_answer() with a string argument containing your result

        For your final answer format:
        - If asked for a number: provide just the number as a string (no commas, no units like $ or % unless specified)
        - If asked for a string: use few words as possible, no articles, no abbreviations, write digits in plain text
        - If asked for a list: provide comma separated values following the above rules

        Examples:
        - For "What is 2+2?": final_answer("4")
        - For "Capital of France?": final_answer("Paris")  
        - For "List first 3 primes": final_answer("2, 3, 5")

        CRITICAL:
        - Always convert your result to a string before calling final_answer()
        - For math: final_answer(str(calculation_result))
        - Keep code simple and direct
        - Use the standard smolagents format with <end_code>
        """
        self.agent.system_prompt += SYSTEM_PROMPT

        logger.info(f"BasicAgent initialized with telemetry {'enabled' if telemetry.enabled else 'disabled'}")

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
                # Check the adapter health first
                adapter_health = self.model_adapter.health_check()

                if adapter_health["status"] != "healthy":
                    span.set_status(SpanStatus.ERROR, "Adapter health check failed")
                    return {
                        "status": "unhealthy",
                        "error": "Adapter health check failed",
                        "adapter_health": adapter_health,
                        "telemetry_enabled": self._telemetry.enabled
                    }

                # Quick agent test
                test_result = self("Hello, just respond with 'Agent OK'")

                span.set_status(SpanStatus.OK)
                return {
                    "status": "healthy",
                    "tools_count": len(self.agent.tools),
                    "telemetry_enabled": self._telemetry.enabled,
                    "test_response_length": len(str(test_result)),
                    "adapter_health": adapter_health
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
            "model_adapter": self.model_adapter.__class__.__name__,
            "llm_client_type": self.llm_client.__class__.__name__,
            "telemetry_enabled": self.telemetry_enabled,
            "llm_telemetry_enabled": self.llm_client.telemetry_enabled,
            "architecture": "Option2_CleanSeparation"
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