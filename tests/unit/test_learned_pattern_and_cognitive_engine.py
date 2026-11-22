"""
Test Suite: Learned Pattern Engine, Cognitive Solver & Autonomous Reasoning Generalization
=============================================================================================
Verifies first-principles deduction, linguistic logic riddles, mathematical invariants,
and zero-shot offline generalization when numeric parameters change.
"""

import pytest
from core.planner.learned_pattern_engine import learned_pattern_engine
from core.planner.cognitive_solver import cognitive_solver
from core.brain.router import IntentRouter, IntentType


def test_farmer_sheep_all_but_n_riddle():
    """Verify that 'all but 9 die' produces exactly 9 sheep left."""
    query = "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
    res = cognitive_solver.solve(query)

    assert res is not None
    assert "9" in res["summary"]
    assert "9 sheep" in res["solution"].lower()
    assert "all_but_n" in res.get("pattern_matched", "")


def test_all_but_n_numeric_and_entity_generalization():
    """Verify offline transfer learning across arbitrary numbers and entity types."""
    # Test 1: 50 sheep, all but 15 die
    res1 = cognitive_solver.solve("A shepherd has 50 sheep. All but 15 die. How many remain?")
    assert "15" in res1["summary"]
    assert "15 sheep" in res1["solution"].lower()

    # Test 2: 28 camels, all but 7 died
    res2 = cognitive_solver.solve("A traveler had 28 camels. All but 7 died. How many camels survive?")
    assert "7" in res2["summary"]
    assert "7 camels" in res2["solution"].lower()

    # Test 3: 100 apples, all but 23 were rotten
    res3 = cognitive_solver.solve("There were 100 apples. All but 23 were rotten. How many fresh apples are left?")
    assert "23" in res3["summary"]
    assert "23 apples" in res3["solution"].lower()


def test_clock_angular_geometry_variations():
    """Verify mathematical invariant for clock hand angles at different times."""
    # 3:15 -> 7.5 degrees
    res1 = cognitive_solver.solve("A clock shows 3:15. What is the angle between the hour hand and minute hand?")
    assert "7.5" in res1["summary"]

    # 8:20 -> 130 degrees
    res2 = cognitive_solver.solve("What is the angle between the clock hands at 8:20?")
    assert "130" in res2["summary"]

    # 7:45 -> 37.5 degrees
    res3 = cognitive_solver.solve("Calculate the angle between hands of a clock at 7:45.")
    assert "37.5" in res3["summary"]


def test_handshake_and_round_robin_combinatorics():
    """Verify C(n, 2) = n*(n-1)/2 invariant."""
    # 20 people in room -> 20 * 19 / 2 = 190 handshakes
    res1 = cognitive_solver.solve("In a room of 20 people, everyone shakes hands with everyone else. How many handshakes?")
    assert "190" in res1["summary"]
    assert "190" in res1["solution"]

    # 10 people -> 10 * 9 / 2 = 45 handshakes
    res2 = cognitive_solver.solve("If 10 people shake hands with each other once, how many handshakes happen?")
    assert "45" in res2["summary"]

    # 8 teams round-robin tournament -> 8 * 7 / 2 = 28 matches
    res3 = cognitive_solver.solve("8 teams play in a round-robin tournament. How many matches are played?")
    assert "28" in res3["summary"]


def test_arithmetic_sum_invariant():
    """Verify sum from 1 to N identity."""
    # 1 to 100 -> 5050
    res1 = cognitive_solver.solve("Find the sum of numbers from 1 to 100.")
    assert "5050" in res1["summary"]

    # 1 to 50 -> 50 * 51 / 2 = 1275
    res2 = cognitive_solver.solve("What is the sum of the first 50 positive integers?")
    assert "1275" in res2["summary"]


def test_train_crossing_relative_speed():
    """Verify speed unit conversion (5/18) and length/speed transit duration."""
    # Train 200m long at 72 km/h (20 m/s) -> 200 / 20 = 10.00 seconds
    res = cognitive_solver.solve("A train 200m long passes a pole at 72 km/h. How many seconds does it take?")
    assert "10.00" in res["summary"]


def test_percentage_discount_and_profit():
    """Verify commercial math invariants."""
    # $80 with 25% discount -> 80 - 20 = 60
    res1 = cognitive_solver.solve("An item originally costs $80 and has a 25% discount. What is the final price?")
    assert "60.00" in res1["summary"]

    # Bought for 500, sold for 650 -> Profit = 150 (30%)
    res2 = cognitive_solver.solve("A merchant bought an article for 500 and sold it for 650. What is the profit?")
    assert "150.00" in res2["summary"]
    assert "30.00%" in res2["summary"]


def test_kinematics_and_combined_work():
    """Verify motion and multi-worker rate formulas."""
    # 300 km at 60 km/h -> 5 hours
    res_kin = cognitive_solver.solve("A car travels 300 km at a speed of 60 km/h. How long does the trip take?")
    assert "5" in res_kin["summary"]

    # 12 and 24 days -> 8 days
    res_work = cognitive_solver.solve("Worker A takes 12 days and Worker B takes 24 days. How long working together?")
    assert "8" in res_work["summary"]


def test_dynamic_pattern_learning_and_offline_generalization():
    """Verify dynamic schema registration and zero-shot offline execution."""
    # Register dynamic pattern for harmonic worker rate
    learned_pattern_engine.register_dynamic_pattern(
        {
            "title": "Autonomous Work Rate Schema",
            "keywords": ["collaborative", "pipeline"],
            "formula_type": "rate",
            "explanation": "Calculates combined time via harmonic product-to-sum ratio.",
        },
        persist=False,
    )

    # Test with new numbers: 6 and 3 -> (6*3)/(6+3) = 18/9 = 2.00
    res = learned_pattern_engine.match_and_solve("Two collaborative systems 6 and 3 process the pipeline together.")
    assert res is not None
    assert "2.00" in res["summary"]
    assert "dynamic_learned" == res["domain"]


def test_intent_router_recognizes_reasoning_and_learned_schemas():
    """Verify intent classifier routes cognitive challenges to ProblemSolvingAgent."""
    router = IntentRouter()

    # Clock problem
    d1 = router.route("A clock shows 3:15. What is the angle between the hour hand and minute hand?")
    assert d1.intent == IntentType.REASONING

    # Sheep riddle
    d2 = router.route("A farmer has 17 sheep. All but 9 die. How many sheep are left?")
    assert d2.intent == IntentType.REASONING

    # Handshake combinatorics
    d3 = router.route("In a room of 20 people, everyone shakes hands. How many handshakes?")
    assert d3.intent == IntentType.REASONING

    # Identity and name query
    d4 = router.route("What is my name?")
    assert d4.intent == IntentType.MEMORY_QUERY

    # Name introduction
    d5 = router.route("Hi my name is Harsha")
    assert d5.intent == IntentType.MEMORY_STORE


def test_system1_arithmetic_reflex():
    """Verify that System 1 fast reflex evaluates arithmetic equations instantly."""
    r1 = cognitive_solver.solve("10 + 20")
    assert "30" in r1["summary"]
    assert "30" in r1["solution"]

    r2 = cognitive_solver.solve("addition 10+20")
    assert "30" in r2["summary"]

    r3 = cognitive_solver.solve("calculate 5 * 12 + 10")
    assert "70" in r3["summary"]


def test_system2_second_largest_unique_algorithm():
    """Verify optimal O(n) time, O(1) space algorithmic synthesis for second largest element."""
    query = "Given an array of integers, find the second-largest unique element without sorting the array. Explain the algorithm and time complexity."
    res = cognitive_solver.solve(query)
    assert res is not None
    assert "\\mathcal{O}(n)" in res["solution"] or "O(n)" in res["solution"]
    assert "\\mathcal{O}(1)" in res["solution"] or "O(1)" in res["solution"]
    assert "def find_second_largest_unique" in res["solution"]
    assert "public class SecondLargestFinder" in res["solution"]
    assert "34" in res["solution"]  # dry run result


def test_prefrontal_executive_metacognitive_depth():
    """Verify Prefrontal Executive rejects superficial templates in favor of concrete depth."""
    from core.brain.neuro_cognitive_engine import neuro_cognitive_engine
    exec_cortex = neuro_cognitive_engine.executive

    shallow_text = (
        "Analysis of: 'Sort array'. Core Theoretical Foundations: "
        "System constraints determine optimal memory-computation trade-offs. "
        "Choose algorithms with bounded worst-case time complexity."
    )
    assert not exec_cortex.evaluate_depth(shallow_text)

    deep_text = (
        "### Optimal Algorithm\n"
        "```python\ndef solve(): pass\n```\n"
        "Time Complexity: O(n), Auxiliary Space: O(1)."
    )
    assert exec_cortex.evaluate_depth(deep_text)


def test_autonomous_research_learning_and_zero_shot_generalization():
    """Verify that when research provides a formula, JARVIS learns the invariant
    and solves subsequent numeric variations 100% offline."""
    # 1. Simulate learning simple interest pattern from web research
    learned_pattern_engine.learn_from_research(
        query="what is the formula for simple interest",
        solution_text="Simple interest is calculated as (P * R * T) / 100 where P is principal, R is annual rate, and T is time in years.",
        research_snippets=[
            "Simple Interest Formula: SI = (P * R * T) / 100. Total amount = Principal + SI."
        ]
    )

    # 2. Test zero-shot offline generalization with new numbers (P=5000, R=4%, T=2 years)
    res_si = learned_pattern_engine.match_and_solve("calculate simple interest on 5000 at 4 percent for 2 years")
    assert res_si is not None
    assert "400" in res_si["solution"]  # 5000 * 4 * 2 / 100 = 400
    assert "5400" in res_si["solution"]  # Total amount = 5400

    # 3. Simulate learning sum of first N integers
    learned_pattern_engine.learn_from_research(
        query="how to find sum of first n natural numbers",
        solution_text="The formula for sum of first n numbers is n*(n+1)/2.",
        research_snippets=["Sum of first n natural numbers: Sn = n(n+1)/2."]
    )

    # 4. Test offline generalization with N=100 -> 100 * 101 / 2 = 5050
    res_sum = learned_pattern_engine.match_and_solve("what is the sum of the first 100 natural numbers?")
    assert res_sum is not None
    assert "5050" in res_sum["solution"]

    # 5. Simulate learning circle geometry
    learned_pattern_engine.learn_from_research(
        query="how to calculate area of a circle from radius",
        solution_text="Area of circle is pi*r^2 and circumference is 2*pi*r.",
        research_snippets=["Circle geometry formulas: Area = pi * r^2, Circumference = 2 * pi * r."]
    )

    # 6. Test offline generalization with radius = 10 -> Area = 314.16, Circ = 62.83
    res_circle = learned_pattern_engine.match_and_solve("find area of circle with radius 10")
    assert res_circle is not None
    assert "314.16" in res_circle["solution"]
    assert "62.83" in res_circle["solution"]


