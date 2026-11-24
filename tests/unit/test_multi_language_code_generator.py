"""Unit tests for Universal Multi-Language Code Generation Engine.

Verifies:
1. Java Prime Number code generation with O(sqrt(N)) primality and Scanner.
2. Multi-language detection (Java, Python, C++, C#, JavaScript, Go, Rust).
3. Algorithmic synthesis for sorting (Quicksort, Mergesort, Bubblesort), searching (Binary Search), and math.
4. DeductiveSolver routing to Universal Code Generator.
5. JARVIS Brain handling /solve requests without executing AST parser on English text.
"""

from __future__ import annotations

import pytest
from core.brain.jarvis_brain import JARVISBrain
from core.planner.code_generator import code_generator
from core.planner.deductive_solver import deductive_solver


def test_language_detection():
    """Verify robust language identification from various query forms."""
    assert code_generator.detect_language("give me the code for prime numbers in java") == "java"
    assert code_generator.detect_language("write a python script for binary search") == "python"
    assert code_generator.detect_language("write a c++ program for quicksort") == "cpp"
    assert code_generator.detect_language("cpp code for mergesort") == "cpp"
    assert code_generator.detect_language("c# program for fibonacci series") == "csharp"
    assert code_generator.detect_language("javascript function to check palindrome") == "javascript"
    assert code_generator.detect_language("go program for prime numbers") == "go"
    assert code_generator.detect_language("rust implementation of binary search") == "rust"


def test_code_request_detection():
    """Verify query classification as code generation requests."""
    assert code_generator.is_code_request("give me the code for prime numbers in java") is True
    assert code_generator.is_code_request("write a python program for quicksort") is True
    assert code_generator.is_code_request("create a c++ script for binary search") is True
    assert code_generator.is_code_request("show java code for factorial") is True
    assert code_generator.is_code_request("what is the capital of france") is False


def test_java_prime_number_generation():
    """Verify complete Java prime number program generation."""
    res = code_generator.generate_code("give me the code for prime numbers in java")
    assert res is not None
    assert res["language"] == "java"
    assert "public class PrimeCheck" in res["code"]
    assert "public static boolean isPrime" in res["code"]
    assert "Scanner" in res["code"]
    assert "O(\\sqrt{N})" in res["solution"] or "O(sqrt(N))" in res["solution"]


def test_cpp_quicksort_generation():
    """Verify complete C++ quicksort program generation."""
    res = code_generator.generate_code("write a c++ program for quicksort")
    assert res is not None
    assert res["language"] == "cpp"
    assert "#include <iostream>" in res["code"]
    assert "partition" in res["code"]
    assert "quickSort" in res["code"]
    assert "Quick Sort" in res["solution"]


def test_python_binary_search_generation():
    """Verify Python binary search algorithm generation."""
    res = code_generator.generate_code("python code for binary search")
    assert res is not None
    assert res["language"] == "python"
    assert "def binary_search" in res["code"]
    assert "mid = left + (right - left) // 2" in res["code"] or "mid =" in res["code"]


def test_deductive_solver_code_integration():
    """Verify DeductiveSolver delegates code generation cleanly."""
    sol = deductive_solver.solve_problem("give me the code for prime numbers in java")
    assert sol is not None
    assert sol["type"] == "code_generation"
    assert "public class" in sol["code"]
    assert "isPrime" in sol["code"]


@pytest.mark.asyncio
async def test_brain_solve_prime_numbers_in_java():
    """End-to-end verification of /solve give me the code for prime numbers in java."""
    brain = JARVISBrain()
    await brain.initialize()

    resp = await brain.think("/solve give me the code for prime numbers in java")
    content = resp.content

    # Must NOT report AST syntax errors
    assert "Syntax Errors" not in content
    assert "Programming Analysis Report" not in content

    # Must provide the full Java solution
    assert "public class" in content
    assert "public static void main" in content
    assert "isPrime" in content or "Prime" in content
    assert resp.agent_used == "DeductiveSolver"
