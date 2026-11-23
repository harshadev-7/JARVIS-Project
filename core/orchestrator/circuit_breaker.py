"""JARVIS Multi-Agent Orchestrator — Fault Tolerance & Circuit Breakers
=======================================================================
Implements resilience patterns including the Circuit Breaker state machine
(CLOSED -> OPEN -> HALF_OPEN), exponential retry backoff, and automatic
fallback agent diversion to prevent cascading swarm failures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"        # Healthy: requests pass through normally
    OPEN = "open"            # Tripped: requests fail fast or divert to fallback
    HALF_OPEN = "half_open"  # Testing: allowing single canary probe to test recovery


class CircuitBreakerOpenError(RuntimeError):
    """Raised when an operation is attempted while the circuit is OPEN and no fallback succeeds."""
    pass


@dataclass
class CircuitBreaker:
    """Monitors failures for a specific agent and manages tripping/recovery."""
    agent_key: str
    failure_threshold: int = 3
    recovery_timeout_sec: float = 10.0
    fallback_agent_key: Optional[str] = None
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    total_trips: int = 0

    def can_execute(self) -> bool:
        """Determines if a request should be dispatched to this agent."""
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if now - self.last_failure_time >= self.recovery_timeout_sec:
                logger.info("Circuit breaker for [%s] transitioning from OPEN -> HALF_OPEN (probing recovery).", self.agent_key)
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow canary request
            return True
        return False

    def record_success(self) -> None:
        """Records a successful execution, resetting failure metrics."""
        now = time.time()
        if self.state in {CircuitState.HALF_OPEN, CircuitState.OPEN}:
            logger.info("Circuit breaker for [%s] successfully recovered: HALF_OPEN -> CLOSED.", self.agent_key)
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_state_change = now

    def record_failure(self, error_message: str = "") -> None:
        """Records a failed execution, tripping to OPEN if threshold is crossed."""
        now = time.time()
        self.consecutive_failures += 1
        self.last_failure_time = now

        if self.state == CircuitState.HALF_OPEN:
            # Canary probe failed, trip back to OPEN
            logger.warning("Canary probe failed for [%s]. Tripping back to OPEN: %s", self.agent_key, error_message)
            self.state = CircuitState.OPEN
            self.total_trips += 1
            self.last_state_change = now
        elif self.state == CircuitState.CLOSED and self.consecutive_failures >= self.failure_threshold:
            logger.warning("Agent [%s] reached failure threshold (%d). Tripping circuit to OPEN.", self.agent_key, self.failure_threshold)
            self.state = CircuitState.OPEN
            self.total_trips += 1
            self.last_state_change = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_key": self.agent_key,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "fallback_agent_key": self.fallback_agent_key,
            "total_trips": self.total_trips,
        }


class CircuitBreakerRegistry:
    """Central registry of circuit breakers for all registered agents."""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_default_breakers()

    def _initialize_default_breakers(self) -> None:
        """Sets up default breakers and intelligent fallback routing paths."""
        fallbacks = {
            "programming": "problem_solving",       # Fallback to general deductive logic
            "mathematical": "problem_solving",      # Fallback to general reasoning
            "research": "task_planning",            # Fallback to planning if internet drops
            "social_media": "content_creation",     # Fallback to general content creation
            "cybersecurity": "problem_solving",     # Fallback to deductive analysis
        }
        for agent_key, fallback in fallbacks.items():
            self.breakers[agent_key] = CircuitBreaker(
                agent_key=agent_key,
                failure_threshold=3,
                recovery_timeout_sec=5.0,
                fallback_agent_key=fallback,
            )

    def get_breaker(self, agent_key: str) -> CircuitBreaker:
        """Retrieves or lazily instantiates a breaker for the agent."""
        clean = agent_key.lower().replace("agent", "").strip("_ ")
        if clean not in self.breakers:
            self.breakers[clean] = CircuitBreaker(agent_key=clean, failure_threshold=3, recovery_timeout_sec=5.0)
        return self.breakers[clean]

    async def execute_with_resilience(
        self,
        agent_key: str,
        execute_fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 2,
        base_delay_sec: float = 0.2,
        fallback_executor: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes a task through the circuit breaker with exponential retry backoff
        and fallback diversion when tripped.
        """
        breaker = self.get_breaker(agent_key)

        # 1. Check if circuit is OPEN
        if not breaker.can_execute():
            logger.warning("Circuit breaker for [%s] is OPEN. Attempting fallback diversion.", agent_key)
            if fallback_executor and breaker.fallback_agent_key:
                logger.info("Diverting task from [%s] to fallback [%s].", agent_key, breaker.fallback_agent_key)
                return await fallback_executor(breaker.fallback_agent_key, *args, **kwargs)
            raise CircuitBreakerOpenError(
                f"Agent [{agent_key}] circuit breaker is OPEN due to repeated failures ({breaker.consecutive_failures} failures). No fallback available."
            )

        # 2. Execute with exponential backoff
        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                result = await execute_fn(*args, **kwargs)
                # Check if result indicates soft failure
                if hasattr(result, "success") and not result.success:
                    err_msg = getattr(result, "error", "") or getattr(result, "summary", "Task failed")
                    breaker.record_failure(str(err_msg))
                    if attempt < max_retries:
                        attempt += 1
                        delay = base_delay_sec * (2 ** (attempt - 1)) + random.uniform(0.01, 0.05)
                        logger.debug("Soft failure for [%s]. Retrying in %.2fs (attempt %d/%d).", agent_key, delay, attempt, max_retries)
                        await asyncio.sleep(delay)
                        continue
                    return result

                breaker.record_success()
                return result

            except Exception as exc:
                last_error = exc
                breaker.record_failure(str(exc))
                if attempt < max_retries:
                    attempt += 1
                    delay = base_delay_sec * (2 ** (attempt - 1)) + random.uniform(0.01, 0.05)
                    logger.debug("Exception in [%s]: %s. Retrying in %.2fs (attempt %d/%d).", agent_key, exc, delay, attempt, max_retries)
                    await asyncio.sleep(delay)
                else:
                    break

        # 3. If retries exhausted and circuit tripped, try fallback
        if breaker.state == CircuitState.OPEN and fallback_executor and breaker.fallback_agent_key:
            logger.info("Retries exhausted for [%s]. Diverting to fallback agent [%s].", agent_key, breaker.fallback_agent_key)
            return await fallback_executor(breaker.fallback_agent_key, *args, **kwargs)

        raise last_error or RuntimeError(f"Task execution failed for [{agent_key}] after {max_retries} retries.")


# Global singleton registry
circuit_breaker_registry = CircuitBreakerRegistry()
