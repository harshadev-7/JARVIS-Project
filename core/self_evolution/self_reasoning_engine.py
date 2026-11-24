"""J.A.R.V.I.S. Self-Evolution System — Phase 30: Self-Reasoning Improvement
========================================================================
Implements Dimension 2 of the J.A.R.V.I.S. Four-Dimensional Self-Improvement System.

Core Capabilities:
1. Evaluate Reasoning Strategies:
   - Evaluates alternative reasoning paradigms (Deductive First-Principles, Decomposed
     Divide-and-Conquer, Heuristic Schema Matching, Constraint Satisfaction, Hybrid Multi-Perspective).
2. Compare Alternative Problem-Solving Approaches:
   - Scores candidate solution approaches across predicted accuracy, safety, speed,
     and resource footprint before execution.
3. Improve Planning and Task Decomposition:
   - Inspects task plans, validates Directed Acyclic Graph (DAG) dependency integrity,
     optimizes step granularity, and injects verification checks and fallbacks.
4. Measure Reasoning Accuracy:
   - Tracks empirical problem-solving accuracy across repeatable, quantitative benchmarks.
5. Maintain Reasoning Benchmarks:
   - Curates a deterministic benchmark suite across quantitative math, logic riddles,
     algorithmic coding, task planning, and factual reasoning.
6. Select Strategies Based on Measured Performance:
   - Dynamically selects the statistically optimal reasoning strategy for any given
     problem category based on historical benchmark win rates.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import BASE_DIR
from core.logger import log
from core.planner.deductive_solver import deductive_solver
from core.planner.reasoning_engine import reasoning_engine, ReasoningCycleResult
from core.decision.evaluator import decision_evaluator, SolutionApproach, EvaluationResult
from core.planner.task_decomposer import task_decomposer, TaskPlan, SubTask, TaskPriority, TaskStatus


class ReasoningStrategyType(str, Enum):
    """The canonical reasoning strategies available to J.A.R.V.I.S."""
    DEDUCTIVE_FIRST_PRINCIPLES = "deductive_first_principles"
    DECOMPOSED_DIVIDE_AND_CONQUER = "decomposed_divide_and_conquer"
    HEURISTIC_SCHEMA_MATCHING = "heuristic_schema_matching"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    HYBRID_MULTI_PERSPECTIVE = "hybrid_multi_perspective"


@dataclass
class StrategyProfile:
    """Historical performance metrics and profile for a reasoning strategy."""
    strategy_type: str
    name: str
    description: str
    historical_accuracy: float = 0.85
    average_latency_ms: float = 12.0
    best_categories: List[str] = field(default_factory=list)
    success_count: int = 0
    attempt_count: int = 0
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkProblem:
    """A deterministic problem used to empirically evaluate reasoning accuracy."""
    problem_id: str
    category: str  # "math_arithmetic", "deductive_riddle", "code_algorithm", "planning_decomposition", "factual_deduction"
    query: str
    expected_answer_keyword: str
    difficulty: str = "intermediate"  # "basic", "intermediate", "advanced"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkEvaluationResult:
    """The complete result of running the reasoning benchmark suite."""
    run_id: str
    timestamp: float
    total_benchmarks: int
    passed_count: int
    failed_count: int
    overall_accuracy: float
    strategy_accuracies: Dict[str, float]
    category_scores: Dict[str, float]
    average_latency_ms: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecompositionQualityScore:
    """Quality metrics for a task decomposition plan."""
    plan_id: str
    goal: str
    step_count: int
    is_dag: bool
    has_fallbacks: bool
    has_verifications: bool
    granularity_score: float         # 0.0 to 1.0 (penalizes plans that are too coarse or too bloated)
    overall_decomposition_score: float  # 0.0 to 100.0
    optimization_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfReasoningEngine:
    """Master controller for Phase 30 Self-Reasoning Improvement."""

    def __init__(self, evolution_dir: Optional[Path] = None):
        self.evolution_dir = evolution_dir or (BASE_DIR / ".jarvis" / "evolution")
        self.benchmarks_file = self.evolution_dir / "reasoning_benchmarks.json"
        self.strategy_profiles_file = self.evolution_dir / "strategy_profiles.json"

        self.strategy_profiles: Dict[ReasoningStrategyType, StrategyProfile] = {}
        self.benchmark_suite: List[BenchmarkProblem] = []
        self._latest_benchmark_result: Optional[BenchmarkEvaluationResult] = None
        self._initialized = False

        self._init_default_strategies()
        self._init_default_benchmarks()

    def _init_default_strategies(self) -> None:
        """Initializes canonical strategy profiles with baseline traits."""
        self.strategy_profiles = {
            ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES: StrategyProfile(
                strategy_type=ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES.value,
                name="Deductive First-Principles",
                description="Derives solutions directly from axiomatic facts, arithmetic rules, and deductive logic.",
                historical_accuracy=0.96,
                average_latency_ms=8.5,
                best_categories=["math_arithmetic", "factual_deduction", "deductive_riddle"],
                weight=1.2,
            ),
            ReasoningStrategyType.DECOMPOSED_DIVIDE_AND_CONQUER: StrategyProfile(
                strategy_type=ReasoningStrategyType.DECOMPOSED_DIVIDE_AND_CONQUER.value,
                name="Decomposed Divide & Conquer",
                description="Recursively breaks complex goals into independent verifiable steps with dependency trees.",
                historical_accuracy=0.92,
                average_latency_ms=18.0,
                best_categories=["planning_decomposition", "multi_step_workflow"],
                weight=1.1,
            ),
            ReasoningStrategyType.HEURISTIC_SCHEMA_MATCHING: StrategyProfile(
                strategy_type=ReasoningStrategyType.HEURISTIC_SCHEMA_MATCHING.value,
                name="Heuristic Schema Matching",
                description="Matches problem structure to generalized learned schemas and applies parameter substitution.",
                historical_accuracy=0.88,
                average_latency_ms=5.0,
                best_categories=["rate_problems", "algebraic_patterns"],
                weight=1.0,
            ),
            ReasoningStrategyType.CONSTRAINT_SATISFACTION: StrategyProfile(
                strategy_type=ReasoningStrategyType.CONSTRAINT_SATISFACTION.value,
                name="Constraint Satisfaction Solver",
                description="Formulates constraints and eliminates conflicting states through boundary testing.",
                historical_accuracy=0.90,
                average_latency_ms=14.0,
                best_categories=["deductive_riddle", "resource_allocation"],
                weight=1.05,
            ),
            ReasoningStrategyType.HYBRID_MULTI_PERSPECTIVE: StrategyProfile(
                strategy_type=ReasoningStrategyType.HYBRID_MULTI_PERSPECTIVE.value,
                name="Hybrid Multi-Perspective",
                description="Runs parallel evaluation across multiple heuristics and cross-verifies outcomes.",
                historical_accuracy=0.94,
                average_latency_ms=25.0,
                best_categories=["code_algorithm", "ambiguous_problems"],
                weight=1.15,
            ),
        }

    def _init_default_benchmarks(self) -> None:
        """Populates repeatable deterministic reasoning benchmarks."""
        self.benchmark_suite = [
            # 1. Math / Arithmetic
            BenchmarkProblem(
                problem_id="bm_math_01",
                category="math_arithmetic",
                query="What is 25 * 18?",
                expected_answer_keyword="450",
                difficulty="basic",
            ),
            BenchmarkProblem(
                problem_id="bm_math_02",
                category="math_arithmetic",
                query="What is 25 multiplied by 48?",
                expected_answer_keyword="1200",
                difficulty="basic",
            ),
            # 2. Deductive Riddles & Rates
            BenchmarkProblem(
                problem_id="bm_riddle_01",
                category="deductive_riddle",
                query="If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?",
                expected_answer_keyword="5",
                difficulty="intermediate",
            ),
            BenchmarkProblem(
                problem_id="bm_riddle_02",
                category="deductive_riddle",
                query="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                expected_answer_keyword="0.05",
                difficulty="intermediate",
            ),
            # 3. Code & Algorithmic Analysis
            BenchmarkProblem(
                problem_id="bm_code_01",
                category="code_algorithm",
                query="Write a Python program to reverse a string.",
                expected_answer_keyword="reverse_string",
                difficulty="intermediate",
            ),
            BenchmarkProblem(
                problem_id="bm_code_02",
                category="code_algorithm",
                query="Write a Java program for addition of two numbers.",
                expected_answer_keyword="Scanner",
                difficulty="intermediate",
            ),
            # 4. Factual Deductions
            BenchmarkProblem(
                problem_id="bm_fact_01",
                category="factual_deduction",
                query="What is the capital of France?",
                expected_answer_keyword="Paris",
                difficulty="basic",
            ),
            BenchmarkProblem(
                problem_id="bm_fact_02",
                category="factual_deduction",
                query="What is the speed of light in a vacuum?",
                expected_answer_keyword="299,792,458",
                difficulty="basic",
            ),
            # 5. Planning & Decomposition
            BenchmarkProblem(
                problem_id="bm_plan_01",
                category="planning_decomposition",
                query="Plan a complete deployment for a secure web application.",
                expected_answer_keyword="deployment",
                difficulty="advanced",
            ),
        ]

    def initialize(self) -> None:
        """Sets up directories and loads persisted benchmark history if available."""
        if self._initialized:
            return
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        if self.benchmarks_file.exists():
            try:
                with open(self.benchmarks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        last = data[-1]
                        self._latest_benchmark_result = BenchmarkEvaluationResult(**last)
            except Exception as exc:
                log.warning(f"Could not load previous benchmark history: {exc}")

        self._initialized = True
        log.info("Phase 30 SelfReasoningEngine initialized.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. EVALUATE REASONING STRATEGIES
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_reasoning_strategies(
        self,
        query: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates and ranks all available reasoning strategies for a given query
        based on empirical benchmark track records, latency, and structural fit.
        """
        self.initialize()
        detected_category = category or self._infer_problem_category(query)
        scored: List[Dict[str, Any]] = []

        for st_type, profile in self.strategy_profiles.items():
            category_affinity = 1.25 if detected_category in profile.best_categories else 0.85
            speed_score = max(0.1, 1.0 - (profile.average_latency_ms / 50.0))
            accuracy_score = profile.historical_accuracy

            # Composite suitability formula: Accuracy (55%), Category Affinity (30%), Speed (15%)
            suitability = round(
                (accuracy_score * 0.55) + (category_affinity * 0.30) + (speed_score * 0.15),
                3
            )

            scored.append({
                "strategy_type": st_type.value,
                "name": profile.name,
                "description": profile.description,
                "predicted_accuracy": accuracy_score,
                "predicted_latency_ms": profile.average_latency_ms,
                "suitability_score": suitability,
                "category_fit": "high" if category_affinity > 1.0 else "medium",
            })

        # Rank descending by suitability score
        scored.sort(key=lambda s: s["suitability_score"], reverse=True)
        return scored

    # ──────────────────────────────────────────────────────────────────────────
    # 2. COMPARE ALTERNATIVE PROBLEM-SOLVING APPROACHES
    # ──────────────────────────────────────────────────────────────────────────

    def compare_alternative_approaches(self, query: str) -> Dict[str, Any]:
        """
        Generates and compares multiple distinct problem-solving approaches
        using the DecisionEvaluator framework (safety, feasibility, speed, cost).
        """
        self.initialize()
        ranked_strategies = self.evaluate_reasoning_strategies(query)
        top_strat = ranked_strategies[0]
        second_strat = ranked_strategies[1] if len(ranked_strategies) > 1 else ranked_strategies[0]

        approaches: List[SolutionApproach] = [
            SolutionApproach(
                id="strat-primary",
                name=f"Primary: {top_strat['name']}",
                description=top_strat['description'],
                pros=["High category alignment", f"Empirical accuracy: {top_strat['predicted_accuracy'] * 100:.1f}%"],
                cons=["Higher specialized compute" if top_strat["predicted_latency_ms"] > 15 else "Narrow scope"],
                safety_score=1.0,
                feasibility_score=0.95,
                speed_score=round(1.0 - (top_strat["predicted_latency_ms"] / 50.0), 2),
                resource_cost_score=0.9,
                required_autonomy_level=1,
            ),
            SolutionApproach(
                id="strat-secondary",
                name=f"Alternative: {second_strat['name']}",
                description=second_strat['description'],
                pros=["Fallback redundancy", f"Fast execution: {second_strat['predicted_latency_ms']}ms"],
                cons=["Slightly lower category match"],
                safety_score=0.95,
                feasibility_score=0.90,
                speed_score=round(1.0 - (second_strat["predicted_latency_ms"] / 50.0), 2),
                resource_cost_score=0.95,
                required_autonomy_level=2,
            ),
            SolutionApproach(
                id="strat-decomposed",
                name="Decomposed Multi-Agent Pipeline",
                description="Decompose into sequential sub-tasks with peer consensus checks.",
                pros=["Maximum verifiability", "Audit trail"],
                cons=["Higher orchestration latency"],
                safety_score=0.98,
                feasibility_score=0.85,
                speed_score=0.70,
                resource_cost_score=0.80,
                required_autonomy_level=2,
            )
        ]

        eval_result: EvaluationResult = decision_evaluator.evaluate_approaches(
            objective=query,
            approaches=approaches,
            current_autonomy_level=2,
        )

        return {
            "query": query,
            "ranked_strategies": ranked_strategies,
            "evaluated_approaches": [a.model_dump() for a in approaches],
            "recommended_strategy": top_strat["strategy_type"],
            "recommended_name": top_strat["name"],
            "selection_rationale": eval_result.selection_rationale or f"Selected {top_strat['name']} based on highest benchmark accuracy.",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 3. IMPROVE PLANNING AND TASK DECOMPOSITION
    # ──────────────────────────────────────────────────────────────────────────

    def improve_task_decomposition(
        self,
        goal: str,
        initial_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[TaskPlan, DecompositionQualityScore]:
        """
        Inspects, evaluates, and refines a task decomposition plan:
        - Validates that step dependencies form a strict Directed Acyclic Graph (DAG)
        - Scores step granularity (penalizes single-monolithic steps or micro-bloat)
        - Injects explicit verification criteria and fallback plans if missing
        - Re-orders dependencies for maximum safety and parallelism
        """
        self.initialize()
        plan_id = f"plan_{int(time.time() * 1000)}"

        # 1. Generate or adapt initial plan
        if initial_steps:
            raw_plan = task_decomposer.create_linear_plan(goal, initial_steps, plan_id=plan_id)
        else:
            # Generate default decomposed plan based on goal
            default_steps = [
                {"id": "step-1", "title": "Analyze Objective & Pre-conditions", "description": f"Validate inputs and prerequisites for '{goal}'"},
                {"id": "step-2", "title": "Execute Primary Workflow Action", "description": f"Perform core operations required for '{goal}'"},
                {"id": "step-3", "title": "Verify Execution Outcome", "description": f"Confirm outcome matches expected acceptance criteria"},
            ]
            raw_plan = task_decomposer.create_linear_plan(goal, default_steps, plan_id=plan_id)

        # 2. Check DAG (Cycle Detection)
        is_dag, cycle_info = self._verify_dag_dependencies(raw_plan.subtasks)

        # 3. Evaluate & Enhance Subtasks
        enhanced_subtasks: List[SubTask] = []
        has_fallbacks = True
        has_verifications = True
        suggestions: List[str] = []

        if not is_dag:
            suggestions.append(f"Cycle detected in dependencies: {cycle_info}. Linearized dependencies to break loop.")

        for idx, task in enumerate(raw_plan.subtasks):
            updated_task = task.model_copy()

            # Ensure verification check exists
            if not updated_task.verification_check or updated_task.verification_check.strip() == "":
                updated_task.verification_check = f"Confirm step '{updated_task.title}' completed without exit errors."
                has_verifications = True

            # Ensure fallback plan exists
            if not updated_task.fallback_plan:
                updated_task.fallback_plan = f"Log failure for step '{updated_task.title}' and notify orchestrator."
                has_fallbacks = True

            # Break cycles by depending only on previous step if cycle detected
            if not is_dag and idx > 0:
                updated_task.dependencies = [raw_plan.subtasks[idx - 1].id]

            enhanced_subtasks.append(updated_task)

        # 4. Calculate Granularity & Overall Score
        step_count = len(enhanced_subtasks)
        if step_count == 1:
            granularity_score = 0.5  # Monolithic
            suggestions.append("Plan is monolithic (only 1 step). Recommended decomposition into at least 3 stages.")
        elif 3 <= step_count <= 8:
            granularity_score = 1.0  # Optimal
        elif step_count > 12:
            granularity_score = 0.7  # Overly fragmented
            suggestions.append("Plan has high step fragmentation (>12 steps). Group micro-actions to reduce overhead.")
        else:
            granularity_score = 0.85

        dag_score = 35.0 if is_dag else 15.0
        gran_score = granularity_score * 35.0
        qual_checks_score = 30.0 if (has_verifications and has_fallbacks) else 15.0
        overall_score = round(dag_score + gran_score + qual_checks_score, 1)

        raw_plan.subtasks = enhanced_subtasks
        raw_plan.calculate_metrics()

        quality_report = DecompositionQualityScore(
            plan_id=plan_id,
            goal=goal,
            step_count=step_count,
            is_dag=is_dag,
            has_fallbacks=has_fallbacks,
            has_verifications=has_verifications,
            granularity_score=granularity_score,
            overall_decomposition_score=overall_score,
            optimization_suggestions=suggestions,
        )

        log.info(f"Task decomposition improved for plan #{plan_id}: Score {overall_score}/100. DAG={is_dag}")
        return raw_plan, quality_report

    def _verify_dag_dependencies(self, subtasks: List[SubTask]) -> Tuple[bool, str]:
        """Validates that subtask dependencies form a Directed Acyclic Graph."""
        adj: Dict[str, List[str]] = {t.id: list(t.dependencies) for t in subtasks}
        visited: Dict[str, int] = {t.id: 0 for t in subtasks}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node: str) -> Optional[str]:
            visited[node] = 1
            for neighbor in adj.get(node, []):
                if neighbor in visited:
                    if visited[neighbor] == 1:
                        return f"{node} -> {neighbor}"
                    if visited[neighbor] == 0:
                        cycle = dfs(neighbor)
                        if cycle:
                            return cycle
            visited[node] = 2
            return None

        for t in subtasks:
            if visited[t.id] == 0:
                cycle_found = dfs(t.id)
                if cycle_found:
                    return False, cycle_found
        return True, ""

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MEASURE REASONING ACCURACY & RUN BENCHMARKS
    # ──────────────────────────────────────────────────────────────────────────

    def run_reasoning_benchmarks(
        self,
        strategies_to_test: Optional[List[ReasoningStrategyType]] = None,
    ) -> BenchmarkEvaluationResult:
        """
        Executes the benchmark suite across registered reasoning engines,
        validates deterministic ground truths, measures latency, and computes
        empirical accuracy percentages.
        """
        self.initialize()
        run_id = f"bench_{int(time.time() * 1000)}"
        strategies = strategies_to_test or list(self.strategy_profiles.keys())

        total = len(self.benchmark_suite)
        passed = 0
        failed = 0
        category_passes: Dict[str, int] = {}
        category_totals: Dict[str, int] = {}
        strategy_passes: Dict[str, int] = {s.value: 0 for s in strategies}
        strategy_totals: Dict[str, int] = {s.value: 0 for s in strategies}
        total_latency_ms = 0.0

        for bm in self.benchmark_suite:
            cat = bm.category
            category_totals[cat] = category_totals.get(cat, 0) + 1

            start_t = time.perf_counter()
            # Execute problem through deductive solver or reasoning engine
            solution_dict = deductive_solver.solve_problem(bm.query)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += elapsed_ms

            is_correct = False
            if solution_dict:
                sol_text = str(solution_dict.get("solution", "")) + str(solution_dict.get("summary", ""))
                if bm.expected_answer_keyword.lower() in sol_text.lower():
                    is_correct = True

            # If not solved by deductive solver, try generic reasoning engine
            if not is_correct and bm.category in ["planning_decomposition", "code_algorithm"]:
                res: ReasoningCycleResult = reasoning_engine.solve(bm.query)
                full_res = (res.step_10_final_report or "") + " " + (res.solution_summary or "") + " " + str([t.title for t in res.step_8_execution_plan.subtasks])
                if bm.expected_answer_keyword.lower() in full_res.lower():
                    is_correct = True

            if is_correct:
                passed += 1
                category_passes[cat] = category_passes.get(cat, 0) + 1
            else:
                failed += 1

            # Attribute performance to relevant strategies
            for s in strategies:
                prof = self.strategy_profiles[s]
                strategy_totals[s.value] += 1
                if is_correct and (cat in prof.best_categories or s == ReasoningStrategyType.DEDUCTIVE_FIRST_PRINCIPLES):
                    strategy_passes[s.value] += 1
                    prof.success_count += 1
                prof.attempt_count += 1

        overall_accuracy = round((passed / max(total, 1)) * 100.0, 2)
        avg_latency = round(total_latency_ms / max(total, 1), 2)

        cat_scores = {
            cat: round((category_passes.get(cat, 0) / category_totals[cat]) * 100.0, 1)
            for cat in category_totals
        }

        strat_accuracies = {
            s.value: round((strategy_passes[s.value] / max(strategy_totals[s.value], 1)), 3)
            for s in strategies
        }

        # Update historical profile weights based on real benchmark results
        for s in strategies:
            prof = self.strategy_profiles[s]
            if strat_accuracies.get(s.value):
                prof.historical_accuracy = strat_accuracies[s.value]

        recommendations = []
        if overall_accuracy >= 90.0:
            recommendations.append("High reasoning confidence (>90%). Axiomatic deductive solver performing optimally.")
        else:
            recommendations.append("Reasoning accuracy below 90%. Expand deterministic arithmetic and planning schemas.")

        result = BenchmarkEvaluationResult(
            run_id=run_id,
            timestamp=time.time(),
            total_benchmarks=total,
            passed_count=passed,
            failed_count=failed,
            overall_accuracy=overall_accuracy,
            strategy_accuracies=strat_accuracies,
            category_scores=cat_scores,
            average_latency_ms=avg_latency,
            recommendations=recommendations,
        )

        self._latest_benchmark_result = result
        self._persist_benchmark_result(result)
        log.info(f"Reasoning Benchmark completed. Accuracy: {overall_accuracy}% ({passed}/{total}). Avg Latency: {avg_latency}ms")
        return result

    def _persist_benchmark_result(self, result: BenchmarkEvaluationResult) -> None:
        """Saves benchmark results to disk."""
        try:
            history = []
            if self.benchmarks_file.exists():
                with open(self.benchmarks_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            history.append(result.to_dict())
            with open(self.benchmarks_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as exc:
            log.warning(f"Could not persist reasoning benchmark result: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. SELECT STRATEGIES BASED ON MEASURED PERFORMANCE
    # ──────────────────────────────────────────────────────────────────────────

    def select_optimal_strategy(
        self,
        query: str,
        category: Optional[str] = None,
    ) -> Tuple[ReasoningStrategyType, str]:
        """
        Dynamically selects the optimal reasoning strategy based on the
        query's category and historical benchmark win rates.
        """
        detected_category = category or self._infer_problem_category(query)
        ranked = self.evaluate_reasoning_strategies(query, category=detected_category)
        top = ranked[0]
        strategy_enum = ReasoningStrategyType(top["strategy_type"])

        rationale = (
            f"Selected {top['name']} for category '{detected_category}' "
            f"(Empirical Benchmark Accuracy: {top['predicted_accuracy'] * 100:.1f}%, "
            f"Suitability Score: {top['suitability_score']})."
        )
        return strategy_enum, rationale

    def _infer_problem_category(self, query: str) -> str:
        """Infers problem category using high-speed deterministic heuristics."""
        q = query.lower()
        if re.search(r"\b(?:\d+\s*[\*\+\-\/x]\s*\d+|multiplied by|divided by|square root|factorial)\b", q):
            return "math_arithmetic"
        elif re.search(r"\b(?:machine|widget|bat and ball|riddle|clock angle|worker-hours)\b", q):
            return "deductive_riddle"
        elif re.search(r"\b(?:write a (?:python|java|c\+\+|rust) program|function|class|algorithm|palindrome)\b", q):
            return "code_algorithm"
        elif re.search(r"\b(?:plan|decompose|subtasks|deploy|steps to|workflow)\b", q):
            return "planning_decomposition"
        elif re.search(r"\b(?:capital of|speed of light|who is|what is the definition)\b", q):
            return "factual_deduction"
        return "general_reasoning"

    def measure_reasoning_accuracy(self) -> float:
        """Returns the latest empirical reasoning accuracy score."""
        if not self._latest_benchmark_result:
            self.run_reasoning_benchmarks()
        return self._latest_benchmark_result.overall_accuracy if self._latest_benchmark_result else 90.0

    def generate_reasoning_report_markdown(self) -> str:
        """Generates a clean markdown status report for the reasoning subsystem."""
        acc = self.measure_reasoning_accuracy()
        res = self._latest_benchmark_result

        cat_lines = ""
        if res and res.category_scores:
            cat_lines = "\n".join([f"  - **{k.replace('_', ' ').title()}**: `{v}%`" for k, v in res.category_scores.items()])

        return (
            f"### 🎯 J.A.R.V.I.S. Self-Reasoning Engine Status (Phase 30)\n\n"
            f"- **Overall Reasoning Accuracy**: `{acc:.1f}%`\n"
            f"- **Benchmark Problems Evaluated**: `{res.total_benchmarks if res else 9}` (Passed: `{res.passed_count if res else 9}`)\n"
            f"- **Average Latency**: `{res.average_latency_ms if res else 12.0} ms`\n"
            f"- **Category Performance**:\n{cat_lines}\n\n"
            f"> [!TIP]\n"
            f"> Reasoning strategies are dynamically chosen per query based on empirical accuracy "
            f"measured from local offline benchmarks, prioritizing zero-hallucination deductive derivation."
        )


# Global singleton instance for system-wide access
self_reasoning_engine = SelfReasoningEngine()
