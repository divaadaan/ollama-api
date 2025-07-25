"""
AI Agent implementation using smolagents with optional telemetry support.
"""

import logging
import time
from typing import Optional

from smolagents import CodeAgent

from telemetry import SpanStatus, TelemetryConfig
from .smolagents_adapter import SmolOllamaAdapter

logger = logging.getLogger(__name__)

class BasicAgent:
    """This is a BasicAgent """

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
            telemetry: Optional telemetry configuration
            tools: Optional list of tools for the agent
        """
        self.llm_client = llm_client

        # Set up telemetry
        if telemetry is None:
            if hasattr(llm_client, '_telemetry'):
                telemetry = llm_client._telemetry
            else:
                # Fall back to disabled telemetry
                from telemetry import TelemetryConfig
                telemetry = TelemetryConfig(enabled=False)

        self._telemetry = telemetry

        self._tracer = telemetry.tracer

        self._request_counter = telemetry.agent_request_counter
        self._response_time = telemetry.agent_response_time

        if tools is None:
            from .tools import get_default_tools
            tools = get_default_tools()

        self.agent = CodeAgent(
            tools=tools,
            model=SmolOllamaAdapter(llm_client),
            additional_authorized_imports=[
                'csv', 'pandas', 'json', 'os', 'pathlib', 'tempfile',
                'urllib', 'requests', 'numpy', 'io', 'base64', 'uuid, re, chess'
            ]
        )

        # Configure system prompt
        SYSTEM_PROMPT = """You are an AI assistant that solves tasks using available tools. Provide clear, direct answers that appropriately address the question.
        
        REQUIRED PATTERN:
        1. Gather information (use tools if needed)
        2. Call final_answer("your_result_as_string") 
        CRITICAL: You MUST call final_answer("result") to complete any task. 
        
        AVAILABLE TOOLS:
        - web_search(query="...") - Search the web
        - visit_webpage(url="...") - Visit webpages 
        - web_scrape(url="...", selector="...", wait_time=3) - website scraping 
        - search_wikipedia(query="...") - Search Wikipedia
        - file_download(url="...") - Download files
        - file_reader(file_path="...") - Read files
        - speech_to_text(file_path="...") - Transcribe audio to text
        - image_to_text(file_path="...", prompt="...") - Analyze images into text
        - file_reader(file_path="...", analyze_with_llm=True, analysis_query="...") - Read files with optional LLM analysis for CSV/Excel
        - extract_text_ocr(image_path="...", language="eng") - Extract text from images using OCR
        
        Remember: Every response must call final_answer() with your result as a string.
        """

        self.agent.system_prompt += SYSTEM_PROMPT

    def __call__(self, question: str) -> str:
        """
        Execute agent on a question.
        Args:
            question: Question or task for the agent to process
        Returns:
            Agent's response/answer
        """
        max_retries =3
        logger.info(f"Agent received question (first 50 chars): {question[:50]}...")

        for attempt in range(max_retries) :
            logger.info(f"Attempt: {attempt+1}")

            # Start telemetry span (no-op if telemetry disabled)
            with self._tracer.start_span("agent_run") as span:
                start_time = time.time()

                span.set_attribute("agent.question_length", len(question))
                span.set_attribute("agent.tools_count", len(self.agent.tools))
                span.set_attribute("agent.model_telemetry", self.llm_client.telemetry_enabled)
                span.set_attribute("agent.max_retries", max_retries)

                try:
                    # Execute the agent
                    raw_answer = self.agent.run(question)

                    if not isinstance(raw_answer, str):
                        raw_answer = str(raw_answer)

                    duration = time.time() - start_time
                    execution_logs = self.agent.memory.steps
                    reasoning_steps = self._extract_reasoning_from_logs(execution_logs)

                    span.set_attribute("agent.response_length", len(str(raw_answer)))
                    span.set_attribute("agent.duration_seconds", duration)

                    logger.info(f"Agent completed successfully in {duration:.2f}s") ##########

                    logger.info(f"Agent returning unvalidated answer: {raw_answer}")

                    validation_result = self._validate_answer(raw_answer, question, reasoning_steps)
                    span.set_attribute("agent.validation_passed", validation_result["valid"])


                    if validation_result["valid"]:
                        self._request_counter.add(1, {"status": "success", "attempt": str(attempt + 1)})
                        self._response_time.record(duration)
                        span.set_status(SpanStatus.OK)

                        logger.info(f"Validation success on attempt  {attempt + 1}")
                        return validation_result["final_answer"]
                    else:
                        logger.warning(f"Validation failed on attempt {attempt + 1}")
                        span.set_attribute("agent.validation_reason", validation_result.get("reason", "unknown"))

                        if attempt== max_retries -1:
                            span.set_status(SpanStatus.ERROR, "All validation attempts failed")
                            logger.error("All validation attempts failed, returning last raw answer")
                            return raw_answer


                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = f"Agent execution failed on attempt {attempt + 1}: {str(e)}"

                    self._request_counter.add(1, {"status": "error", "attempt": str(attempt + 1)})
                    span.set_attribute("agent.duration_seconds", duration)
                    span.set_attribute("agent.error_message", str(e))
                    span.set_status(SpanStatus.ERROR, error_msg)

                    logger.error(error_msg)
                    if attempt == max_retries - 1:
                        raise
                    else:
                        logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
        raise RuntimeError("Unexpected end of retry loop")

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
                test_result = self("Your task is to respond with the text 'Agent OK'")

                span.set_status(SpanStatus.OK)
                return {
                    "status": "healthy",
                    "tools_count": len(self.agent.tools),
                    "telemetry_enabled": self._telemetry.enabled,
                    "test_response_length": len(str(test_result))
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

    def _validate_answer(self, raw_answer: str, question: str, reasoning_steps: str) -> dict:
        """Validate answer using LLM client"""
        try:
            with self._tracer.start_span("agent validation") as span:
                final_answer = self._extract_final_answer(raw_answer)
                logger.info(f"Extracted final answer: {final_answer}")
                span.set_attribute("validation.raw_answer_length", len(raw_answer))
                span.set_attribute("validation.final_answer_length", len(final_answer))

                logger.info("Checking reasoning...")
                reasoning_valid = self._check_reasoning(question, reasoning_steps, final_answer)

                logger.info(f"Reasoning check result: {reasoning_valid}")
                is_valid = reasoning_valid

                logger.info(f"Validation result: {is_valid}")
                span.set_attribute("validation.result", is_valid)
                span.set_status(SpanStatus.OK)

                return {
                    "valid": is_valid,
                    "final_answer": final_answer,
                    "reasoning_steps": reasoning_steps
                }

        except Exception as e:
            return {
                "valid": False,
                "final_answer": raw_answer,
                "reason": f"validation_error: {str(e)}"
            }

    def _extract_final_answer(self, raw_answer: str) -> str:
        """Extract and format the final answer from the raw response."""
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        sep_token = "FINAL ANSWER:"

        if sep_token in raw_answer:
            formatted_answer = raw_answer.split(sep_token)[1].strip()
        else:
            formatted_answer = raw_answer.strip()

        # Only remove brackets if they appear to be formatting artifacts
        if formatted_answer.startswith('[') and formatted_answer.endswith(']'):
            formatted_answer = formatted_answer[1:-1].strip()

        return formatted_answer

    def _check_reasoning(self, question: str, reasoning_steps: str,final_answer: str) -> bool:
        """Check if the reasoning and results are correct"""
        logger.info("Performing reasoning validation...")
        try:
            with self._tracer.start_span("reasoning_check") as span:
                prompt = f"""You are validating an AI agent's reasoning process. 
                Task: {question}
                
                Agent's step-by-step reasoning and code execution:
                {reasoning_steps}
                
                Final answer: {final_answer}
                
                Evaluate if:
                1. The reasoning steps make sense for the task
                2. The code execution leads logically to the final answer
                3. For simple tasks, minimal reasoning is acceptable
                
                Respond with PASS if the reasoning is adequate for the task complexity, FAIL otherwise. Include a very brief explanation of how you came to this decision.
                If the Agent has returned an unsuccessful API call then FAIL the Agent.
                """


                result = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=500
                )

                if "error" in result:
                    logger.error(f"Reasoning check error: {result}")
                    span.set_status(SpanStatus.ERROR, "Reasoning check failed")
                    return False

                response = result.get("content", "")
                logger.debug(f"Reasoning check response: {response}...")
                span.set_attribute("reasoning_check.response", response[:200])

                is_valid = "PASS" in response and "FAIL" not in response
                logger.debug(f"Reasoning validation result: {is_valid}")
                span.set_attribute("reasoning_check.result", is_valid)
                span.set_status(SpanStatus.OK)

                return is_valid

        except Exception as e:
            logger.error(f"Reasoning check failed: {str(e)}")
            return False

    def _extract_reasoning_from_logs(self, logs: list) -> str:
        """Extract reasoning steps from logs"""
        reasoning_parts = []

        for step in self.agent.memory.steps:
            step_type = step.__class__.__name__
            if step_type == 'TaskStep':
                if hasattr(step, 'task'):
                    reasoning_parts.append(f"Task: {step.task}")

            elif step_type == 'ActionStep':
                # Extract code and result from execution step
                if hasattr(step, 'model_output'):
                    reasoning_parts.append(f"Agent reasoning: {step.model_output}")

                if hasattr(step, 'observations'):
                    reasoning_parts.append(f"Execution results: {step.observations}")

                if hasattr(step, 'tool_calls') and step.tool_calls:
                    reasoning_parts.append(f"Tools used: {step.tool_calls}")

        full_reason = "\n".join(reasoning_parts) if reasoning_parts else "No reasoning captured - simple task?"
        return full_reason

# Factory function
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


    return BasicAgent(
        llm_client=llm_client,
        telemetry=telemetry_config,
        tools=tools
    )