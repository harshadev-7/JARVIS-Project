"""JARVIS Multi-Agent Orchestrator — Consensus & Cross-Verification Engine
========================================================================
Implements multi-perspective agent voting, consensus verification protocols,
and cross-agent critiques to validate high-stakes decisions before commitment.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsensusStrategy(str, Enum):
    MAJORITY_VOTE = "majority_vote"            # > 50% agreement
    UNANIMOUS = "unanimous"                    # 100% agreement required
    WEIGHTED_CONFIDENCE = "weighted_confidence" # Multiplied by domain authority weights
    FIRST_VALID = "first_valid"                # First passing verification terminates early


@dataclass
class VerificationVote:
    """An individual agent's review and vote on a proposed outcome."""
    agent_key: str
    approved: bool
    confidence: float
    critique: str
    suggestions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_key": self.agent_key,
            "approved": self.approved,
            "confidence": round(self.confidence, 2),
            "critique": self.critique,
            "suggestions": self.suggestions,
        }


@dataclass
class ConsensusOutcome:
    """The synthesized outcome of a multi-agent consensus verification session."""
    agreed: bool
    strategy_used: ConsensusStrategy
    consensus_score: float
    verified_output: Any
    votes: List[VerificationVote]
    dissenting_opinions: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreed": self.agreed,
            "strategy_used": self.strategy_used.value,
            "consensus_score": round(self.consensus_score, 2),
            "votes_count": len(self.votes),
            "votes": [v.to_dict() for v in self.votes],
            "dissenting_opinions": self.dissenting_opinions,
            "summary": self.summary,
        }


class ConsensusEngine:
    """Orchestrates cross-agent verification reviews and evaluates consensus."""

    # Default authority weights across specialized domains
    DOMAIN_WEIGHTS: Dict[str, float] = {
        "cybersecurity": 1.5,     # Highest weight on security invariants
        "programming": 1.2,       # High weight on syntax and architecture
        "problem_solving": 1.1,   # General deductive reasoning weight
        "mathematical": 1.3,      # High weight on mathematical accuracy
        "research": 1.0,
        "data_analysis": 1.0,
        "task_planning": 1.0,
    }

    async def evaluate_proposal(
        self,
        proposal: str,
        verifying_agents: List[str],
        agent_executor: Any,
        strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConsensusOutcome:
        """
        Submits a proposed solution or plan to multiple verifying agents concurrently
        and calculates consensus according to the chosen strategy.
        """
        if not verifying_agents:
            return ConsensusOutcome(
                agreed=True,
                strategy_used=strategy,
                consensus_score=1.0,
                verified_output=proposal,
                votes=[],
                summary="Self-approved: zero verification agents requested.",
            )

        verification_prompt = (
            f"Review and verify the following proposal for correctness, security, and logical consistency:\n"
            f"PROPOSAL: {proposal}\n\n"
            f"State whether you APPROVE or REJECT, along with your reasoning and confidence level (0.0 to 1.0)."
        )

        async def query_verifier(key: str) -> VerificationVote:
            try:
                res = await agent_executor(key, verification_prompt, context=context)
                summary_text = getattr(res, "summary", "") or str(res)
                lower = summary_text.lower()

                # Parse approval status from agent summary
                is_rejected = any(w in lower for w in ["reject", "vulnerability", "flaw", "insecure", "incorrect", "invalid", "fail"])
                approved = not is_rejected
                confidence = 0.85 if approved else 0.70

                return VerificationVote(
                    agent_key=key,
                    approved=approved,
                    confidence=confidence,
                    critique=summary_text[:150],
                    suggestions=None,
                )
            except Exception as exc:
                logger.warning("Verification failed for agent [%s]: %s", key, exc)
                return VerificationVote(
                    agent_key=key,
                    approved=False,
                    confidence=0.5,
                    critique=f"Verification agent threw error: {exc}",
                )

        # Execute all verification queries in parallel
        votes: List[VerificationVote] = await asyncio.gather(*(query_verifier(k) for k in verifying_agents))

        # Evaluate consensus based on strategy
        agreed, score, dissent = self._calculate_consensus(votes, strategy)

        summary_msg = (
            f"Consensus reached (Score: {score:.2f}, Strategy: {strategy.value})"
            if agreed else
            f"Consensus rejected (Score: {score:.2f}, Strategy: {strategy.value}) with {len(dissent)} dissenting objection(s)."
        )

        return ConsensusOutcome(
            agreed=agreed,
            strategy_used=strategy,
            consensus_score=score,
            verified_output=proposal if agreed else None,
            votes=votes,
            dissenting_opinions=dissent,
            summary=summary_msg,
        )

    def _calculate_consensus(
        self,
        votes: List[VerificationVote],
        strategy: ConsensusStrategy,
    ) -> tuple[bool, float, List[str]]:
        """Calculates agreement boolean, numerical score, and dissenting critiques."""
        if not votes:
            return True, 1.0, []

        approvals = [v for v in votes if v.approved]
        dissent = [f"[{v.agent_key}] {v.critique}" for v in votes if not v.approved]

        if strategy == ConsensusStrategy.UNANIMOUS:
            agreed = len(approvals) == len(votes)
            score = len(approvals) / len(votes)
            return agreed, score, dissent

        elif strategy == ConsensusStrategy.MAJORITY_VOTE:
            score = len(approvals) / len(votes)
            agreed = score > 0.5
            return agreed, score, dissent

        elif strategy == ConsensusStrategy.WEIGHTED_CONFIDENCE:
            total_weight = 0.0
            approved_weight = 0.0
            for v in votes:
                w = self.DOMAIN_WEIGHTS.get(v.agent_key, 1.0) * v.confidence
                total_weight += w
                if v.approved:
                    approved_weight += w
            score = (approved_weight / total_weight) if total_weight > 0 else 0.0
            agreed = score >= 0.6
            return agreed, score, dissent

        elif strategy == ConsensusStrategy.FIRST_VALID:
            agreed = len(approvals) >= 1
            score = 1.0 if agreed else 0.0
            return agreed, score, dissent

        return False, 0.0, dissent


# Global singleton instance
consensus_engine = ConsensusEngine()
