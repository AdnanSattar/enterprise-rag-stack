"""
LLM client wrapper with grounding and structured output support.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deployment.circuit_breaker import CircuitOpenError, get_llm_breaker
from shared.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response with metadata."""

    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any = None


class LLMClient:
    """
    LLM client with circuit breaker and retry logic.

    Supports:
    - OpenAI API
    - Structured JSON output
    - Temperature control for determinism
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.llm.model
        self.temperature = (
            temperature if temperature is not None else settings.llm.temperature
        )
        self.max_tokens = max_tokens or settings.llm.max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Override temperature (use 0 for high-risk tasks)
            max_tokens: Override max tokens
            json_mode: Request JSON output

        Returns:
            LLMResponse with content and metadata
        """
        breaker = get_llm_breaker()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = breaker.call(self.client.chat.completions.create, **kwargs)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                raw_response=response,
            )

        except CircuitOpenError:
            logger.error("LLM circuit breaker is open")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_structured(self, prompt: str, system_prompt: str = None) -> Dict:
        """
        Generate structured JSON output.

        Args:
            prompt: User prompt (should request JSON)
            system_prompt: System prompt

        Returns:
            Parsed JSON dict
        """
        response = self.generate(
            prompt=prompt, system_prompt=system_prompt, json_mode=True
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from response
            return self._extract_json(response.content)

    def _extract_json(self, text: str) -> Dict:
        """Try to extract JSON from text that may have extra content."""
        import re

        # Try to find JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"error": "Could not parse JSON", "raw": text}


class ModelRouter:
    """
    Route queries to appropriate model based on complexity.

    Cost optimization: use smaller models for simple queries,
    larger models for complex tasks.
    """

    MODELS = {
        "small": "gpt-4.1-mini",
        "medium": "gpt-4.1-mini",
        "large": "gpt-4.1",
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._clients: Dict[str, LLMClient] = {}

    def get_client(self, model_size: str) -> LLMClient:
        """Get or create client for model size."""
        if model_size not in self._clients:
            model = self.MODELS.get(model_size, self.MODELS["medium"])
            self._clients[model_size] = LLMClient(api_key=self.api_key, model=model)
        return self._clients[model_size]

    def route(self, query: str) -> str:
        """
        Determine which model to use based on query.

        Simple heuristic:
        - Short queries: small model
        - Legal/policy queries: large model
        - Default: medium model
        """
        query_lower = query.lower()

        # Short simple queries
        if len(query) < 40:
            return "small"

        # Complex domain-specific queries
        complex_keywords = [
            "legal",
            "policy",
            "compliance",
            "contract",
            "regulation",
            "liability",
            "indemnification",
        ]
        if any(kw in query_lower for kw in complex_keywords):
            return "large"

        # Multi-part questions
        if query.count("?") > 1 or " and " in query_lower:
            return "large"

        return "medium"

    def generate(
        self, prompt: str, query: str = None, force_model: str = None, **kwargs
    ) -> LLMResponse:
        """
        Generate response with automatic model routing.

        Args:
            prompt: Full prompt
            query: Original query for routing (optional)
            force_model: Override routing (small/medium/large)
            **kwargs: Additional args for LLM client

        Returns:
            LLMResponse
        """
        if force_model:
            model_size = force_model
        elif query:
            model_size = self.route(query)
        else:
            model_size = "medium"

        logger.info(f"Routing to {model_size} model")
        client = self.get_client(model_size)
        return client.generate(prompt, **kwargs)


# Global LLM client
_llm_client: Optional[LLMClient] = None
_model_router: Optional[ModelRouter] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_model_router() -> ModelRouter:
    """Get global model router."""
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
