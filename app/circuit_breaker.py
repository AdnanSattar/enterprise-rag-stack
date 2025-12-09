"""
Circuit breaker pattern for resilient LLM calls.
Prevents cascading failures when external services are degraded.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker(failure_threshold=5)
        result = breaker.call(external_api_call, *args)
    """

    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_max_calls: int = 3

    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failures: int = field(default=0)
    successes: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if (
                self.last_failure_time
                and (time.time() - self.last_failure_time) >= self.reset_timeout
            ):
                self._transition_to_half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit breaker OPEN: {self.failures} failures in succession")
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.successes = 0

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info("Circuit breaker CLOSED: service recovered")
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.half_open_calls = 0

    def _record_success(self):
        """Record a successful call."""
        self.failures = 0

        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.half_open_max_calls:
                self._transition_to_closed()

    def _record_failure(self):
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failures >= self.failure_threshold:
            self._transition_to_open()

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            fn: Function to call
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Original exception from function
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"Circuit is {self.state.value}. "
                f"Wait {self.reset_timeout - (time.time() - (self.last_failure_time or 0)):.0f}s"
            )

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    async def call_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Async version of call."""
        if not self._should_allow_request():
            raise CircuitOpenError(f"Circuit is {self.state.value}")

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failures": self.failures,
            "successes": self.successes,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator to wrap a function with circuit breaker.

    Usage:
        breaker = CircuitBreaker()

        @with_circuit_breaker(breaker)
        def call_external_api():
            ...
    """

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return breaker.call(fn, *args, **kwargs)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call_async(fn, *args, **kwargs)

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    return decorator


# Global circuit breakers for different services
_llm_breaker: Optional[CircuitBreaker] = None
_reranker_breaker: Optional[CircuitBreaker] = None


def get_llm_breaker() -> CircuitBreaker:
    """Get circuit breaker for LLM calls."""
    global _llm_breaker
    if _llm_breaker is None:
        _llm_breaker = CircuitBreaker(
            failure_threshold=3, reset_timeout=30.0, half_open_max_calls=2
        )
    return _llm_breaker


def get_reranker_breaker() -> CircuitBreaker:
    """Get circuit breaker for reranker calls."""
    global _reranker_breaker
    if _reranker_breaker is None:
        _reranker_breaker = CircuitBreaker(
            failure_threshold=5, reset_timeout=60.0, half_open_max_calls=3
        )
    return _reranker_breaker
