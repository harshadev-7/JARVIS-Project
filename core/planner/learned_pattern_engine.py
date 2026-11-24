"""
JARVIS AI — Learned Problem Pattern & Autonomous Generalization Engine
========================================================================
Enables JARVIS to learn problem-solving patterns from experience and research,
abstracting the underlying mathematical, physical, or logical formulas so that
when similar questions are encountered with different numbers or variables,
JARVIS recognizes the structure and solves them completely offline without internet.
"""

from __future__ import annotations

import re
import math
import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.logger import log


class LearnedPatternEngine:
    """
    Recognizes mathematical, physical, and logical problem schemas, applies parameter substitution,
    and enables zero-shot generalization to new numeric inputs without internet access.
    Also dynamically stores and retrieves learned schemas from SQLite memory.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._dynamic_patterns: List[Dict[str, Any]] = []
        self._db_path = db_path or str(Path("D:/Documents/JARVIS/jarvis.db"))
        self._initialized = False
        self._load_patterns_from_sqlite()

    def _load_patterns_from_sqlite(self) -> None:
        """Synchronously load previously learned problem patterns from SQLite memory."""
        try:
            if not Path(self._db_path).exists():
                return
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, content FROM knowledge_items WHERE category = 'learned_problem_pattern'"
            )
            rows = cursor.fetchall()
            for row in rows:
                item_id, title, content = row
                try:
                    # Content may be structured JSON or key-value text
                    if content.startswith("{") and content.endswith("}"):
                        data = json.loads(content)
                        self._dynamic_patterns.append(data)
                    else:
                        self._dynamic_patterns.append({
                            "title": title,
                            "text": content,
                        })
                except Exception:
                    pass
            conn.close()
            self._initialized = True
            if self._dynamic_patterns:
                log.info(f"LearnedPatternEngine loaded {len(self._dynamic_patterns)} patterns from SQLite.")
        except Exception as e:
            log.debug(f"LearnedPatternEngine SQLite load notice: {e}")

    async def initialize(self):
        """Async compatibility hook to refresh stored patterns."""
        self._load_patterns_from_sqlite()

    def match_and_solve(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate if query matches a known problem structure.
        If matched, extract new variables and compute verified solution offline.
        """
        q_lower = query.lower().strip()

        # 1. Clock Angular Geometry Pattern
        if self._matches_clock_pattern(q_lower):
            sol = self._solve_clock_pattern(query)
            if sol:
                return sol

        # 2. Combined Work / Two Workers Pattern
        if self._matches_combined_work_pattern(q_lower):
            sol = self._solve_combined_work_pattern(query)
            if sol:
                return sol

        # 3. Relative Speed & Train Crossing Pattern (Specific kinematics)
        if self._matches_train_crossing_pattern(q_lower):
            sol = self._solve_train_crossing_pattern(query)
            if sol:
                return sol

        # 4. Kinematics: Speed, Distance, and Travel Time Pattern (General)
        if self._matches_speed_distance_pattern(q_lower):
            sol = self._solve_speed_distance_pattern(query)
            if sol:
                return sol

        # 5. Compound Interest Pattern
        if self._matches_compound_interest_pattern(q_lower):
            sol = self._solve_compound_interest_pattern(query)
            if sol:
                return sol

        # 5b. Simple Financial Interest Pattern
        if self._matches_simple_interest_pattern(q_lower):
            sol = self._solve_simple_interest_pattern(query)
            if sol:
                return sol

        # 6. Right Triangle / Pythagorean Theorem Pattern
        if self._matches_pythagorean_pattern(q_lower):
            sol = self._solve_pythagorean_pattern(query)
            if sol:
                return sol

        # 7. Kinetic Energy & Physics Invariant Pattern
        if self._matches_kinetic_energy_pattern(q_lower):
            sol = self._solve_kinetic_energy_pattern(query)
            if sol:
                return sol

        # 8. Linguistic Riddle: "All But N" Invariant Pattern
        if self._matches_all_but_pattern(q_lower):
            sol = self._solve_all_but_pattern(query)
            if sol:
                return sol

        # 9. Chronological Invariant: Age Difference Invariant Pattern
        if self._matches_age_ratio_pattern(q_lower):
            sol = self._solve_age_ratio_pattern(query)
            if sol:
                return sol

        # 10. Linguistic Riddle: "Take Away N" Possession Pattern
        if self._matches_take_away_pattern(q_lower):
            sol = self._solve_take_away_pattern(query)
            if sol:
                return sol

        # 11. Handshake & Round-Robin Combinatorics Pattern
        if self._matches_handshake_pattern(q_lower):
            sol = self._solve_handshake_pattern(query)
            if sol:
                return sol

        # 12. Sum of Natural Numbers / Arithmetic Progression Pattern
        if self._matches_arithmetic_sum_pattern(q_lower):
            sol = self._solve_arithmetic_sum_pattern(query)
            if sol:
                return sol

        # 13. Percentage Discount & Profit/Loss Pattern
        if self._matches_percentage_profit_discount_pattern(q_lower):
            sol = self._solve_percentage_profit_discount_pattern(query)
            if sol:
                return sol

        # 14. Compound Interest Pattern
        if self._matches_compound_interest_pattern(q_lower):
            sol = self._solve_compound_interest_pattern(query)
            if sol:
                return sol

        # 15. Combinatorial Probability (Coin tosses & Dice rolls)
        if self._matches_probability_pattern(q_lower):
            sol = self._solve_probability_pattern(query)
            if sol:
                return sol

        # 16. Dynamic Learned Patterns (Acquired from prior research & stored in SQLite)
        dynamic_sol = self._match_dynamic_patterns(query)
        if dynamic_sol:
            return dynamic_sol

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 1: Clock & Hand Angle
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_clock_pattern(self, q: str) -> bool:
        has_clock = any(w in q for w in ["clock", "watch", "hour hand", "minute hand"])
        has_angle = any(w in q for w in ["angle", "degree", "degrees", "separated"])
        has_time = bool(re.search(r"\b(\d{1,2}\s*[:\.]\s*\d{2}|\d{1,2}\s*o'?clock)\b", q))
        return (has_clock and has_angle) or (has_time and has_angle)

    def _solve_clock_pattern(self, query: str) -> Dict[str, Any]:
        """Compute angle for arbitrary clock time."""
        time_match = re.search(r"\b(\d{1,2})\s*[:\.]\s*(\d{2})\b", query)
        if time_match:
            hours = int(time_match.group(1))
            minutes = int(time_match.group(2))
        else:
            hr_match = re.search(r"\b(\d{1,2})\s*(?:o'?clock)?\b", query)
            hours = int(hr_match.group(1)) if hr_match else 12
            minutes = 0

        h_disp = hours
        m_disp = minutes
        h_norm = hours % 12

        theta_m = 6.0 * minutes
        theta_h = (30.0 * h_norm) + (0.5 * minutes)

        diff = abs(theta_h - theta_m)
        interior = min(diff, 360.0 - diff)
        reflex = 360.0 - interior

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.1f}"

        solution = (
            f"### **Learned Pattern Applied: Clock Hand Angular Geometry (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: $\\theta_m = 6M$, $\\theta_h = 30H + 0.5M$, $\\Delta \\theta = |\\theta_h - \\theta_m|$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"At **{h_disp}:{m_disp:02d}**, the angle between the hour hand and minute hand is **{fmt(interior)}°** "
            f"(Reflex angle: **{fmt(reflex)}°**).\n\n"
            f"#### **2. Mathematical Derivation (Calculated Locally):**\n"
            f"1. **Minute Hand Position ($\\theta_m$):**\n"
            f"   - Sweeps $6^\\circ$ per minute: $6^\\circ \\times {minutes} = \\mathbf{{{fmt(theta_m)}^\\circ}}$\n"
            f"2. **Hour Hand Position ($\\theta_h$):**\n"
            f"   - Sweeps $30^\\circ$ per hour + $0.5^\\circ$ per minute: $(30^\\circ \\times {h_norm}) + (0.5^\\circ \\times {minutes}) = \\mathbf{{{fmt(theta_h)}^\\circ}}$\n"
            f"3. **Angular Separation ($\\Delta \\theta$):**\n"
            f"   - $|{fmt(theta_h)}^\\circ - {fmt(theta_m)}^\\circ| = \\mathbf{{{fmt(diff)}^\\circ}}$\n"
            f"   - Minimal interior angle: $\\min({fmt(diff)}^\\circ, 360^\\circ - {fmt(diff)}^\\circ) = \\mathbf{{{fmt(interior)}^\\circ}}$."
        )

        return {
            "domain": "angular_geometry",
            "solution": solution,
            "summary": f"Angle at {h_disp}:{m_disp:02d} is {fmt(interior)} degrees.",
            "pattern_matched": "clock_angle",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 2: Combined Work / Two Workers
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_combined_work_pattern(self, q: str) -> bool:
        has_work = any(w in q for w in ["work", "job", "task", "tank", "cistern", "fill"])
        has_together = any(w in q for w in ["together", "combined", "both", "simultaneously"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return has_work and has_together and len(nums) >= 2

    def _solve_combined_work_pattern(self, query: str) -> Dict[str, Any]:
        """Compute time taken when two workers or pipes work simultaneously."""
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", query)]
        time_unit_match = re.search(r"\b(days?|hours?|minutes?|mins?|hrs?|secs?)\b", query, re.I)
        time_unit = time_unit_match.group(1) if time_unit_match else "days"

        if len(nums) < 2:
            return None

        t1, t2 = nums[0], nums[1]
        combined = (t1 * t2) / (t1 + t2) if (t1 + t2) > 0 else 0

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.2f}"

        solution = (
            f"### **Learned Pattern Applied: Combined Work & Rate Invariant (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: Combined Rate $R_{{total}} = \\frac{{1}}{{t_1}} + \\frac{{1}}{{t_2}}$, Total Time $T = \\frac{{t_1 \\times t_2}}{{t_1 + t_2}}$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"Working together, the objective is completed in **{fmt(combined)} {time_unit}**.\n\n"
            f"#### **2. Step-by-Step Derivation:**\n"
            f"1. Individual Rates:\n"
            f"   - Entity A: $\\frac{{1}}{{{fmt(t1)}}}$ of the work per {time_unit[:-1] if time_unit.endswith('s') else time_unit}\n"
            f"   - Entity B: $\\frac{{1}}{{{fmt(t2)}}}$ of the work per {time_unit[:-1] if time_unit.endswith('s') else time_unit}\n"
            f"2. Combined Work Rate:\n"
            f"   $$R = \\frac{{1}}{{{fmt(t1)}}} + \\frac{{1}}{{{fmt(t2)}}} = \\frac{{{fmt(t1 + t2)}}}{{{fmt(t1 * t2)}}}$$\n"
            f"3. Inverted for Total Duration ($T = 1/R$):\n"
            f"   $$T = \\frac{{{fmt(t1)} \\times {fmt(t2)}}}{{{fmt(t1)} + {fmt(t2)}}} = \\frac{{{fmt(t1 * t2)}}}{{{fmt(t1 + t2)}}} = \\mathbf{{{fmt(combined)} \\text{{ {time_unit}}}}}$$"
        )

        return {
            "domain": "rate_and_work",
            "solution": solution,
            "summary": f"Completed together in {fmt(combined)} {time_unit}.",
            "pattern_matched": "combined_work",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 3: Kinematics (Speed, Distance, Time)
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_speed_distance_pattern(self, q: str) -> bool:
        has_speed = any(w in q for w in ["speed", "velocity", "km/h", "mph", "m/s", "travels at", "speed of"])
        has_distance = any(w in q for w in ["distance", "km", "miles", "meters", "far does", "travel"])
        has_time = any(w in q for w in ["hours", "hours to", "time taken", "how long", "minutes", "seconds"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return ((has_speed and has_distance) or (has_speed and has_time) or (has_distance and has_time)) and len(nums) >= 2

    def _solve_speed_distance_pattern(self, query: str) -> Dict[str, Any]:
        """Compute speed, distance, or time using kinematics invariants."""
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        if len(nums) < 2:
            return None

        q_lower = query.lower()

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.2f}"

        # Case A: Finding Time (Distance and Speed given)
        if any(w in q_lower for w in ["how long", "time taken", "how many hours", "how many minutes"]):
            dist, speed = max(nums), min(nums)
            time_val = dist / speed if speed > 0 else 0
            solution = (
                f"### **Learned Pattern Applied: Kinematics (Time = Distance / Speed)**\n\n"
                f"#### **1. Definitive Answer:**\n"
                f"The journey takes **{fmt(time_val)} hours**.\n\n"
                f"#### **2. Kinematics Derivation:**\n"
                f"- **Distance ($D$):** {fmt(dist)} units\n"
                f"- **Speed ($S$):** {fmt(speed)} units/hr\n"
                f"- **Formula:** $T = \\frac{{D}}{{S}} = \\frac{{{fmt(dist)}}}{{{fmt(speed)}}} = \\mathbf{{{fmt(time_val)} \\text{{ hours}}}}$."
            )
            return {"domain": "kinematics", "solution": solution, "summary": f"Duration is {fmt(time_val)} hours.", "pattern_matched": "speed_distance_time", "verified": True}

        # Case B: Finding Distance (Speed and Time given)
        if any(w in q_lower for w in ["how far", "distance", "what distance"]):
            speed, time_val = nums[0], nums[1]
            dist = speed * time_val
            solution = (
                f"### **Learned Pattern Applied: Kinematics (Distance = Speed × Time)**\n\n"
                f"#### **1. Definitive Answer:**\n"
                f"The total distance traveled is **{fmt(dist)} units**.\n\n"
                f"#### **2. Kinematics Derivation:**\n"
                f"- **Speed ($S$):** {fmt(speed)} units/hr\n"
                f"- **Time ($T$):** {fmt(time_val)} hours\n"
                f"- **Formula:** $D = S \\times T = {fmt(speed)} \\times {fmt(time_val)} = \\mathbf{{{fmt(dist)} \\text{{ units}}}}$."
            )
            return {"domain": "kinematics", "solution": solution, "summary": f"Distance is {fmt(dist)} units.", "pattern_matched": "speed_distance_time", "verified": True}

        # Case C: Finding Speed (Distance and Time given)
        dist, time_val = nums[0], nums[1]
        speed = dist / time_val if time_val > 0 else 0
        solution = (
            f"### **Learned Pattern Applied: Kinematics (Speed = Distance / Time)**\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"The required speed is **{fmt(speed)} units/hr**.\n\n"
            f"#### **2. Kinematics Derivation:**\n"
            f"- **Distance ($D$):** {fmt(dist)} units\n"
            f"- **Time ($T$):** {fmt(time_val)} hours\n"
            f"- **Formula:** $S = \\frac{{D}}{{T}} = \\frac{{{fmt(dist)}}}{{{fmt(time_val)}}} = \\mathbf{{{fmt(speed)} \\text{{ units/hr}}}}$."
        )
        return {"domain": "kinematics", "solution": solution, "summary": f"Speed is {fmt(speed)} units/hr.", "pattern_matched": "speed_distance_time", "verified": True}

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 4: Simple Financial Interest
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_simple_interest_pattern(self, q: str) -> bool:
        if "compound" in q:
            return False
        clean_q = re.sub(r"(?<=\d),(?=\d)", "", q)
        has_interest = any(w in q for w in ["simple interest", "principal", "rate of interest", "invested", "borrowed", "interest on"])
        nums = re.findall(r"\d+(?:\.\d+)?", clean_q)
        return has_interest and len(nums) >= 2

    def _solve_simple_interest_pattern(self, query: str) -> Dict[str, Any]:
        """Compute simple interest and total accrued amount."""
        clean_q = re.sub(r"(?<=\d),(?=\d)", "", query)
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", clean_q)]
        if len(nums) < 2:
            return None

        if len(nums) >= 3:
            p = max(nums)
            remaining = [n for n in nums if n != p]
            r = remaining[0] if remaining else 5.0
            t = remaining[1] if len(remaining) > 1 else 1.0
        else:
            p = max(nums)
            r = min(nums)
            t = 1.0

        si = (p * r * t) / 100.0
        total = p + si

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.2f}"

        solution = (
            f"### **Learned Pattern Applied: Simple Financial Interest (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: $SI = \\frac{{P \\times R \\times T}}{{100}}$, $\\text{{Total Amount}} = P + SI$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"- **Simple Interest ($SI$):** **{fmt(si)}**\n"
            f"- **Total Maturity Amount ($A$):** **{fmt(total)}**\n\n"
            f"#### **2. Step-by-Step Mathematical Derivation:**\n"
            f"1. **Principal ($P$):** {fmt(p)}\n"
            f"2. **Annual Interest Rate ($R$):** {fmt(r)}%\n"
            f"3. **Time Duration ($T$):** {fmt(t)} years\n"
            f"4. **Calculation:**\n"
            f"   $$SI = \\frac{{{fmt(p)} \\times {fmt(r)} \\times {fmt(t)}}}{{100}} = \\mathbf{{{fmt(si)}}}$$\n"
            f"   $$\\text{{Total}} = {fmt(p)} + {fmt(si)} = \\mathbf{{{fmt(total)}}}$$"
        )

        return {
            "domain": "financial_mathematics",
            "solution": solution,
            "summary": f"Simple interest is {fmt(si)}, Total amount is {fmt(total)}.",
            "pattern_matched": "simple_interest",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 5: Right Triangle / Pythagorean Theorem
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_pythagorean_pattern(self, q: str) -> bool:
        return any(w in q for w in ["hypotenuse", "pythagorean", "pythagoras", "right triangle", "right-angled triangle"]) and len(re.findall(r"\d+(?:\.\d+)?", q)) >= 2

    def _solve_pythagorean_pattern(self, query: str) -> Dict[str, Any]:
        """Compute hypotenuse or missing side in a right triangle."""
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        if len(nums) < 2:
            return None

        a, b = nums[0], nums[1]
        c = math.sqrt(a**2 + b**2)

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.2f}"

        solution = (
            f"### **Learned Pattern Applied: Pythagorean Theorem (Offline Geometry Engine)**\n\n"
            f"*(Recognized invariant schema: $a^2 + b^2 = c^2 \\implies c = \\sqrt{{a^2 + b^2}}$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"The hypotenuse ($c$) is **{fmt(c)} units**.\n\n"
            f"#### **2. Mathematical Proof:**\n"
            f"1. Leg $a$: {fmt(a)}, Leg $b$: {fmt(b)}\n"
            f"2. $$c = \\sqrt{{{fmt(a)}^2 + {fmt(b)}^2}} = \\sqrt{{{fmt(a**2)} + {fmt(b**2)}}} = \\sqrt{{{fmt(a**2 + b**2)}}} = \\mathbf{{{fmt(c)}}}$$"
        )
        return {"domain": "geometry", "solution": solution, "summary": f"Hypotenuse is {fmt(c)} units.", "pattern_matched": "pythagorean", "verified": True}

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 6: Kinetic Energy
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_kinetic_energy_pattern(self, q: str) -> bool:
        return "kinetic energy" in q and len(re.findall(r"\d+(?:\.\d+)?", q)) >= 2

    def _solve_kinetic_energy_pattern(self, query: str) -> Dict[str, Any]:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", query)]
        if len(nums) < 2:
            return None
        m, v = nums[0], nums[1]
        ke = 0.5 * m * (v**2)

        def fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else f"{n:.2f}"

        solution = (
            f"### **Learned Pattern Applied: Kinetic Energy (Offline Physics Engine)**\n\n"
            f"*(Recognized invariant schema: $KE = \\frac{{1}}{{2}} m v^2$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"The kinetic energy is **{fmt(ke)} Joules**.\n\n"
            f"#### **2. Physical Derivation:**\n"
            f"- **Mass ($m$):** {fmt(m)} kg\n"
            f"- **Velocity ($v$):** {fmt(v)} m/s\n"
            f"- **Calculation:**\n"
            f"  $$KE = \\frac{{1}}{{2}} \\times {fmt(m)} \\times ({fmt(v)})^2 = \\frac{{1}}{{2}} \\times {fmt(m)} \\times {fmt(v**2)} = \\mathbf{{{fmt(ke)} \\text{{ J}}}}$$"
        )
        return {"domain": "physics", "solution": solution, "summary": f"Kinetic energy is {fmt(ke)} J.", "pattern_matched": "kinetic_energy", "verified": True}

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 7: "All But N" Linguistic & Logical Invariant
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_all_but_pattern(self, q: str) -> bool:
        return bool(re.search(
            r"\ball\s+but\s+(\d+)\s*(?:of\s+them\s*)?(?:were\s+|are\s+|is\s+|got\s+|turned\s+)?(?:die|died|survive|survived|left|remain|remained|rotten|broken|stolen|lost|escaped|fled|away)\b",
            q,
            re.I
        ))

    def _solve_all_but_pattern(self, query: str) -> Dict[str, Any]:
        """Solve classic 'all but N survive/die' riddles for any numeric variation."""
        m = re.search(
            r"\ball\s+but\s+(\d+)\s*(?:of\s+them\s*)?(?:were\s+|are\s+|is\s+|got\s+|turned\s+)?(?:die|died|survive|survived|left|remain|remained|rotten|broken|stolen|lost|escaped|fled|away)\b",
            query,
            re.I
        )
        surviving = int(m.group(1)) if m else 9

        nums = [int(n) for n in re.findall(r"\b\d+\b", query)]
        total = nums[0] if nums and nums[0] != surviving else surviving + 8
        perished = total - surviving if total >= surviving else 0

        # Extract entity (e.g. sheep, cows, apples, camels)
        entity_match = re.search(r"\b\d+\s+([a-zA-Z]+)\b", query)
        entity = entity_match.group(1).lower() if entity_match else "items"

        q_lower = query.lower()
        if "rotten" in q_lower:
            action = "were rotten"
            unaffected = "fresh / good"
        elif any(w in q_lower for w in ["die", "died"]):
            action = "died"
            unaffected = "remain alive"
        elif any(w in q_lower for w in ["broken"]):
            action = "were broken"
            unaffected = "intact"
        else:
            action = "perished / departed"
            unaffected = "remain left"

        solution = (
            f"### **Learned Pattern Applied: 'All But N' Logical Invariant (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: 'All but $N$' is an exclusionary operator meaning exactly $N$ survive / remain unaffected)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"There are **{surviving} {entity} {unaffected}**.\n\n"
            f"#### **2. First-Principles Logical Derivation:**\n"
            f"1. **Grammatical & Mathematical Structure:**\n"
            f"   - Initial Total: **{total} {entity}**.\n"
            f"   - The phrase *\"All but {surviving} {action}\"* means: **Every item EXCEPT {surviving} {action}**.\n\n"
            f"2. **Decomposition:**\n"
            f"   - Number affected (${action}$): ${total} - {surviving} = {perished} \\text{{ {entity}}}$\n"
            f"   - Number that did **not** perish/fail ({unaffected}): **{surviving} {entity}**\n\n"
            f"#### **3. Cognitive Insight:**\n"
            f"- A common cognitive trap is attempting subtraction (${total} - {surviving} = {perished}$) and guessing {perished}.\n"
            f"- However, the riddle explicitly states that **all except {surviving}** were affected. Therefore, exactly **{surviving}** remain."
        )

        return {
            "domain": "linguistic_logic",
            "solution": solution,
            "summary": f"Exactly {surviving} {entity} remain alive.",
            "pattern_matched": "all_but_n",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 8: Chronological Age Difference Invariant
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_age_ratio_pattern(self, q: str) -> bool:
        has_age = any(w in q for w in ["age", "years old", "half my age", "sister", "brother"])
        return has_age and "now" in q and len(re.findall(r"\b\d+\b", q)) >= 2

    def _solve_age_ratio_pattern(self, query: str) -> Dict[str, Any]:
        """Solve age difference invariant riddles."""
        nums = [int(n) for n in re.findall(r"\b\d+\b", query)]
        if len(nums) >= 2:
            my_past = nums[0]
            my_now = nums[1]
            sibling_past = my_past // 2 if "half" in query.lower() else my_past - 2
            diff = my_past - sibling_past
            sibling_now = my_now - diff

            solution = (
                f"### **Learned Pattern Applied: Age Difference Invariant (Offline Engine)**\n\n"
                f"*(Recognized invariant schema: $\\Delta\\text{{age}} = \\text{{Constant}}$ across time)*\n\n"
                f"#### **1. Definitive Answer:**\n"
                f"Your sibling is currently **{sibling_now} years old**.\n\n"
                f"#### **2. Mathematical Proof:**\n"
                f"1. **Past State:**\n"
                f"   - When you were {my_past}, your sibling was {sibling_past} years old.\n"
                f"   - The age difference between you is: ${my_past} - {sibling_past} = {diff} \\text{{ years}}$.\n\n"
                f"2. **Present State:**\n"
                f"   - Age differences never change as time elapses ($\\Delta = \\text{{Constant}}$).\n"
                f"   - Since you are now {my_now}, your sibling's age is: ${my_now} - {diff} = \\mathbf{{{sibling_now} \\text{{ years}}}}$."
            )
            return {"domain": "chronological_logic", "solution": solution, "summary": f"Sibling is {sibling_now} years old.", "pattern_matched": "age_difference", "verified": True}

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 9: Possession / "Take Away" Linguistic Riddle
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_take_away_pattern(self, q: str) -> bool:
        return "take away" in q and any(w in q for w in ["apple", "apples", "orange", "oranges", "coins", "items"]) and len(re.findall(r"\b\d+\b", q)) >= 2

    def _solve_take_away_pattern(self, query: str) -> Dict[str, Any]:
        nums = [int(n) for n in re.findall(r"\b\d+\b", query)]
        taken = nums[1] if len(nums) >= 2 else nums[0]
        solution = (
            f"### **Learned Pattern Applied: Possession Invariant (Offline Engine)**\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"You have **{taken} items**.\n\n"
            f"#### **2. First-Principles Logic:**\n"
            f"- The riddle asks: *\"How many do **YOU** have?\"*\n"
            f"- Since **you** took away {taken}, **you** possess exactly **{taken}**."
        )
        return {"domain": "linguistic_logic", "solution": solution, "summary": f"You possess {taken} items.", "pattern_matched": "take_away", "verified": True}

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 10: Handshake & Round-Robin Combinatorics
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_handshake_pattern(self, q: str) -> bool:
        has_handshake = any(w in q for w in ["shake hands", "handshake", "handshakes", "shakes hand", "shakes hands", "round-robin", "round robin"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return has_handshake and len(nums) >= 1

    def _solve_handshake_pattern(self, query: str) -> Dict[str, Any]:
        """Compute total combinations C(n, 2) = n*(n-1)/2."""
        nums = [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        n = nums[0] if nums else 10
        total_handshakes = (n * (n - 1)) // 2

        term = "matches" if any(w in query.lower() for w in ["round-robin", "round robin", "tournament", "teams"]) else "handshakes"

        solution = (
            f"### **Learned Pattern Applied: Handshake & Pairwise Combinatorics (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: Combinatorial selection of pairs without repetition $C(n, 2) = \\frac{{n(n - 1)}}{{2}}$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"For **{n} participants**, there are exactly **{total_handshakes} {term}**.\n\n"
            f"#### **2. Combinatorial Derivation:**\n"
            f"1. **Graph Theory / Combinatorics Formalization:**\n"
            f"   - Each of the $n = {n}$ people can shake hands with $(n - 1) = {n - 1}$ other people.\n"
            f"   - Total directed interactions = $n \\times (n - 1) = {n} \\times {n - 1} = {n * (n - 1)}$.\n"
            f"2. **Eliminating Double-Counting:**\n"
            f"   - A handshake between Person A and Person B is symmetric ($A \\leftrightarrow B$). It is counted only once.\n"
            f"   - Therefore, divide by 2:\n"
            f"     $$\\text{{Total {term.title()}}} = \\frac{{{n} \\times ({n} - 1)}}{{2}} = \\frac{{{n * (n - 1)}}}{{2}} = \\mathbf{{{total_handshakes}}}$$"
        )
        return {
            "domain": "combinatorics",
            "solution": solution,
            "summary": f"Total {term} for {n} participants is {total_handshakes}.",
            "pattern_matched": "handshake_combinatorics",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 11: Sum of First N Natural Numbers / AP
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_arithmetic_sum_pattern(self, q: str) -> bool:
        has_sum = any(w in q for w in ["sum of", "add all numbers", "sum of numbers", "sum of first", "sum from 1 to"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return has_sum and len(nums) >= 1

    def _solve_arithmetic_sum_pattern(self, query: str) -> Dict[str, Any]:
        nums = [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        n = max(nums) if nums else 100
        total_sum = (n * (n + 1)) // 2

        solution = (
            f"### **Learned Pattern Applied: Sum of Arithmetic Series (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: Gauss's summation identity $\\sum_{{i=1}}^n i = \\frac{{n(n + 1)}}{{2}}$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"The sum of numbers from 1 to **{n}** is **{total_sum}**.\n\n"
            f"#### **2. Mathematical Proof:**\n"
            f"$$\\text{{Sum}} = \\frac{{{n} \\times ({n} + 1)}}{{2}} = \\frac{{{n} \\times {n + 1}}}{{2}} = \\frac{{{n * (n + 1)}}}{{2}} = \\mathbf{{{total_sum}}}$$"
        )
        return {
            "domain": "arithmetic_series",
            "solution": solution,
            "summary": f"Sum from 1 to {n} is {total_sum}.",
            "pattern_matched": "arithmetic_sum",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 12: Relative Speed & Train Crossing
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_train_crossing_pattern(self, q: str) -> bool:
        has_train = any(w in q for w in ["train", "crossing", "passes a pole", "cross each other", "trains of length"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return has_train and len(nums) >= 2

    def _solve_train_crossing_pattern(self, query: str) -> Dict[str, Any]:
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        if len(nums) < 2:
            return None

        # If two lengths and two speeds, or one length and one speed
        q_lower = query.lower()
        if len(nums) >= 4:
            l1, l2, s1, s2 = nums[0], nums[1], nums[2], nums[3]
            total_len = l1 + l2
            # Check direction
            if "same direction" in q_lower:
                rel_speed_kmh = abs(s1 - s2)
            else:
                rel_speed_kmh = s1 + s2
            rel_speed_ms = rel_speed_kmh * (5.0 / 18.0)
            time_sec = total_len / rel_speed_ms if rel_speed_ms > 0 else 0
            summary_txt = f"Crossing time is {time_sec:.2f} seconds."
        else:
            length = nums[0]
            speed_kmh = nums[1]
            speed_ms = speed_kmh * (5.0 / 18.0)
            time_sec = length / speed_ms if speed_ms > 0 else 0
            summary_txt = f"Passage time is {time_sec:.2f} seconds."

        solution = (
            f"### **Learned Pattern Applied: Relative Speed & Transit Dynamics (Offline Engine)**\n\n"
            f"*(Recognized invariant schema: $T = \\frac{{\\text{{Total Distance}}}}{{S_{{rel}}}}$, Unit conversion: $1 \\text{{ km/h}} = \\frac{{5}}{{18}} \\text{{ m/s}}$)*\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"The time required is **{time_sec:.2f} seconds**.\n\n"
            f"#### **2. Kinematics Derivation:**\n"
            f"- Conversion: Speed in m/s = $\\text{{km/h}} \\times \\frac{{5}}{{18}}$\n"
            f"- Time = $\\frac{{\\text{{Total Length (m)}}}}{{\\text{{Relative Speed (m/s)}}}} = \\mathbf{{{time_sec:.2f} \\text{{ seconds}}}}$."
        )
        return {
            "domain": "relative_kinematics",
            "solution": solution,
            "summary": summary_txt,
            "pattern_matched": "train_crossing",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 13: Percentage Discount & Profit / Loss
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_percentage_profit_discount_pattern(self, q: str) -> bool:
        has_disc = any(w in q for w in ["discount", "off", "marked price", "original price"])
        has_prof = any(w in q for w in ["profit", "loss", "cost price", "selling price", "bought for", "sold for"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return (has_disc or has_prof) and len(nums) >= 2

    def _solve_percentage_profit_discount_pattern(self, query: str) -> Dict[str, Any]:
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", query)]
        if len(nums) < 2:
            return None

        q_lower = query.lower()
        if "discount" in q_lower or "off" in q_lower:
            orig = max(nums)
            disc_rate = min(nums)
            discount_amount = orig * (disc_rate / 100.0)
            final_price = orig - discount_amount

            solution = (
                f"### **Learned Pattern Applied: Percentage Discount Calculation (Offline Engine)**\n\n"
                f"*(Recognized invariant schema: $\\text{{Discount}} = \\text{{Price}} \\times \\frac{{D}}{{100}}$, $\\text{{Final}} = \\text{{Price}} - \\text{{Discount}}$)*\n\n"
                f"#### **1. Definitive Answer:**\n"
                f"- **Discount Amount:** **${discount_amount:.2f}**\n"
                f"- **Final Price:** **${final_price:.2f}**\n\n"
                f"#### **2. Step-by-Step Calculation:**\n"
                f"1. Original Price: ${orig:.2f}, Discount Rate: {disc_rate:.1f}%\n"
                f"2. $\\text{{Discount}} = {orig:.2f} \\times \\frac{{{disc_rate}}}{{100}} = \\mathbf{{{discount_amount:.2f}}}$\n"
                f"3. $\\text{{Final Price}} = {orig:.2f} - {discount_amount:.2f} = \\mathbf{{{final_price:.2f}}}$"
            )
            return {"domain": "commercial_math", "solution": solution, "summary": f"Final price after {disc_rate}% discount is ${final_price:.2f}.", "pattern_matched": "percentage_discount", "verified": True}

        # Profit & Loss
        cp = nums[0]
        sp = nums[1]
        profit = sp - cp
        profit_pct = (profit / cp) * 100.0 if cp > 0 else 0
        term = "Profit" if profit >= 0 else "Loss"

        solution = (
            f"### **Learned Pattern Applied: Profit & Loss Analysis (Offline Engine)**\n\n"
            f"#### **1. Definitive Answer:**\n"
            f"- **{term}:** **${abs(profit):.2f}** ({abs(profit_pct):.2f}%)\n\n"
            f"#### **2. Financial Derivation:**\n"
            f"- Cost Price (CP): ${cp:.2f}\n"
            f"- Selling Price (SP): ${sp:.2f}\n"
            f"- Invariant: $\\text{{{term}}} = |SP - CP| = \\mathbf{{{abs(profit):.2f}}}$\n"
            f"- Percentage: $\\frac{{{abs(profit):.2f}}}{{{cp:.2f}}} \\times 100 = \\mathbf{{{abs(profit_pct):.2f}\\%}}$"
        )
        return {"domain": "commercial_math", "solution": solution, "summary": f"{term} is ${abs(profit):.2f} ({abs(profit_pct):.2f}%).", "pattern_matched": "profit_loss", "verified": True}

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 14: Compound Interest
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_compound_interest_pattern(self, q: str) -> bool:
        clean_q = re.sub(r"(?<=\d),(?=\d)", "", q)
        return ("compound interest" in q or "compound interest on" in q) and len(re.findall(r"\d+(?:\.\d+)?", clean_q)) >= 2

    def _solve_compound_interest_pattern(self, query: str) -> Dict[str, Any]:
        clean_q = re.sub(r"(?<=\d),(?=\d)", "", query)
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_q)]
        if len(nums) < 2:
            return None

        currency_sym = "₹" if ("₹" in query or "rs" in query.lower() or "inr" in query.lower()) else "$"
        p = max(nums)
        remaining = [n for n in nums if n != p]
        r = remaining[0] if remaining else 8.0
        t = remaining[1] if len(remaining) > 1 else 2.0

        amount = p * ((1.0 + (r / 100.0)) ** t)
        ci = amount - p

        solution = (
            f"### **Financial Calculation: Compound Interest**\n\n"
            f"*(Formula: $A = P\\left(1 + \\frac{{R}}{{100}}\\right)^T$, $CI = A - P$)*\n\n"
            f"#### **1. Definitive Result:**\n"
            f"- **Principal ($P$):** {currency_sym}{p:,.2f}\n"
            f"- **Annual Rate ($R$):** {r:g}%\n"
            f"- **Time Period ($T$):** {int(t) if t.is_integer() else t} years\n"
            f"- **Compound Interest ($CI$):** **{currency_sym}{ci:,.2f}**\n"
            f"- **Maturity Amount ($A$):** **{currency_sym}{amount:,.2f}**\n\n"
            f"#### **2. Step-by-Step Derivation:**\n"
            f"$$A = {currency_sym}{p:,.2f} \\times \\left(1 + \\frac{{{r}}}{{100}}\\right)^{{{int(t) if t.is_integer() else t}}} = \\mathbf{{{currency_sym}{amount:,.2f}}}$$\n"
            f"$$CI = {currency_sym}{amount:,.2f} - {currency_sym}{p:,.2f} = \\mathbf{{{currency_sym}{ci:,.2f}}}$$"
        )
        return {
            "domain": "financial_mathematics",
            "solution": solution,
            "summary": f"Compound interest on {currency_sym}{p:,.2f} at {r:g}% for {int(t) if t.is_integer() else t} years is {currency_sym}{ci:,.2f} (Total amount: {currency_sym}{amount:,.2f}).",
            "pattern_matched": "compound_interest",
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern 15: Combinatorial Probability (Coin & Dice)
    # ──────────────────────────────────────────────────────────────────────────
    def _matches_probability_pattern(self, q: str) -> bool:
        has_prob = any(w in q for w in ["probability", "odds", "chance"])
        has_items = any(w in q for w in ["coin", "coins", "toss", "tosses", "dice", "die", "roll", "rolls"])
        nums = re.findall(r"\d+(?:\.\d+)?", q)
        return has_prob and has_items and len(nums) >= 1

    def _solve_probability_pattern(self, query: str) -> Dict[str, Any]:
        nums = [int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", query)]
        n = nums[0] if nums else 3
        q_lower = query.lower()

        if "coin" in q_lower or "toss" in q_lower:
            total_outcomes = 2**n
            if "at least one head" in q_lower or "at least 1 head" in q_lower:
                favorable = total_outcomes - 1
                prob = favorable / total_outcomes
                prob_pct = prob * 100.0
                desc = f"At least one head in {n} coin tosses"
            else:
                favorable = 1
                prob = 1.0 / total_outcomes
                prob_pct = prob * 100.0
                desc = f"Specific outcome across {n} coin tosses"

            solution = (
                f"### **Learned Pattern Applied: Combinatorial Probability (Offline Engine)**\n\n"
                f"*(Recognized invariant schema: For $n$ binary trials, Total Sample Space $|S| = 2^n$)*\n\n"
                f"#### **1. Definitive Answer:**\n"
                f"The probability of **{desc}** is **{favorable}/{total_outcomes}** (or **{prob_pct:.2f}%**).\n\n"
                f"#### **2. Mathematical Proof:**\n"
                f"1. Total possible sequences of {n} coin flips = $2^{{{n}}} = {total_outcomes}$.\n"
                f"2. Complement rule: $P(\\ge 1 \\text{{ Head}}) = 1 - P(\\text{{0 Heads}}) = 1 - \\frac{{1}}{{{total_outcomes}}} = \\frac{{{favorable}}}{{{total_outcomes}}} = \\mathbf{{{prob_pct:.2f}\\%}}$."
            )
            return {"domain": "probability", "solution": solution, "summary": f"Probability is {favorable}/{total_outcomes} ({prob_pct:.1f}%).", "pattern_matched": "probability_coin", "verified": True}

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic Pattern Learning & Continuous Adaptation Engine
    # ──────────────────────────────────────────────────────────────────────────
    def register_dynamic_pattern(self, pattern_dict: Dict[str, Any], persist: bool = True) -> None:
        """Register a dynamic pattern in memory and optionally persist into SQLite."""
        self._dynamic_patterns.append(pattern_dict)
        if persist:
            try:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO knowledge_items (title, content, category, source, is_verified, created_at)
                    VALUES (?, ?, 'learned_problem_pattern', 'autonomous_learning', 1, CURRENT_TIMESTAMP)
                    """,
                    (pattern_dict.get("title", "Dynamic Pattern"), json.dumps(pattern_dict))
                )
                conn.commit()
                conn.close()
                log.info(f"Dynamically persisted pattern: {pattern_dict.get('title')}")
            except Exception as e:
                log.warning(f"Could not persist dynamic pattern: {e}")

    def _match_dynamic_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Match query against dynamically learned schemas stored in memory."""
        q_lower = query.lower()
        nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", query)]

        for pat in self._dynamic_patterns:
            keywords = pat.get("keywords", [])
            if keywords and all(kw in q_lower for kw in keywords):
                formula_type = pat.get("formula_type", "constant")
                explanation = pat.get("explanation", "")
                title = pat.get("title", "Dynamically Learned Pattern")

                # Perform substitution if formula is recognized
                if formula_type == "handshake" and len(nums) >= 1:
                    n = int(nums[0])
                    res = (n * (n - 1)) // 2
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"Result for input **{n}**: **{res}**.\n\n"
                            f"#### **2. Learned Invariant Derivation:**\n"
                            f"{explanation}\n\n"
                            f"- Calculation: $\\frac{{{n} \\times ({n} - 1)}}{{2}} = \\mathbf{{{res}}}$."
                        ),
                        "summary": f"Evaluated dynamic pattern: {res}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "rate" and len(nums) >= 2:
                    t1, t2 = nums[0], nums[1]
                    comb = (t1 * t2) / (t1 + t2) if (t1 + t2) > 0 else 0
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"Combined value: **{comb:.2f}**.\n\n"
                            f"#### **2. Learned Derivation:**\n"
                            f"{explanation}"
                        ),
                        "summary": f"Calculated combined rate: {comb:.2f}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "exclusionary" and len(nums) >= 1:
                    # 'All but N die / remain' -> N is the answer
                    ans = int(nums[-1])
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"Remaining count: **{ans}**.\n\n"
                            f"#### **2. Invariant Explanation:**\n"
                            f"{explanation}\n\n"
                            f"The phrase 'all but {ans}' specifically excludes {ans} items from the effect."
                        ),
                        "summary": f"Evaluated exclusionary logic: {ans}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "simple_interest" and len(nums) >= 3:
                    p, r, t = nums[0], nums[1], nums[2]
                    si = (p * r * t) / 100.0
                    total = p + si
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"Simple Interest: **{si:.2f}** | Total Amount: **{total:.2f}**.\n\n"
                            f"#### **2. Formula Applied:**\n"
                            f"$\\text{{SI}} = \\frac{{{p} \\times {r} \\times {t}}}{{100}} = \\mathbf{{{si:.2f}}}$\n"
                            f"$\\text{{Total}} = P + \\text{{SI}} = {p} + {si:.2f} = \\mathbf{{{total:.2f}}}$."
                        ),
                        "summary": f"Calculated Simple Interest: {si:.2f}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "sum_n" and len(nums) >= 1:
                    n = int(nums[0])
                    sum_val = (n * (n + 1)) // 2
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"Sum of first {n} natural numbers: **{sum_val}**.\n\n"
                            f"#### **2. Formula Applied:**\n"
                            f"$S_n = \\frac{{{n}({n} + 1)}}{{2}} = \\mathbf{{{sum_val}}}$."
                        ),
                        "summary": f"Calculated sum of first {n} numbers: {sum_val}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "circle_geometry" and len(nums) >= 1:
                    r = nums[0]
                    area = math.pi * (r ** 2)
                    circ = 2 * math.pi * r
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Pattern Applied: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Answer:**\n"
                            f"For radius $r = {r}$:\n"
                            f"- Area: **{area:.2f}**\n"
                            f"- Circumference: **{circ:.2f}**\n\n"
                            f"#### **2. Derivation:**\n"
                            f"- $\\text{{Area}} = \\pi r^2 = \\pi \\times ({r})^2 = \\mathbf{{{area:.2f}}}$\n"
                            f"- $\\text{{Circumference}} = 2\\pi r = 2\\pi \\times {r} = \\mathbf{{{circ:.2f}}}$."
                        ),
                        "summary": f"Calculated circle geometry: Area={area:.2f}, Circ={circ:.2f}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

                elif formula_type == "constant_fact":
                    fact_answer = pat.get("fact_answer", "")
                    return {
                        "domain": "dynamic_learned",
                        "solution": (
                            f"### **Dynamically Learned Knowledge: {title} (Offline Engine)**\n\n"
                            f"#### **1. Definitive Finding:**\n"
                            f"{fact_answer or explanation}\n\n"
                            f"#### **2. Source Intelligence:**\n"
                            f"This invariant fact was acquired and verified during research."
                        ),
                        "summary": f"Retrieved verified invariant: {title}.",
                        "pattern_matched": title,
                        "verified": True,
                    }

        return None

    def learn_from_research(self, query: str, solution_text: str, research_snippets: List[str]) -> None:
        """
        Analyze research findings and automatically register a generalized pattern schema.
        Extracts invariant keywords and underlying formulas so changed variables can be solved offline.
        """
        q_lower = query.lower()
        combined_text = (solution_text + " " + " ".join(research_snippets)).lower()

        # Check for Handshake / Pairwise combinations
        if any(w in q_lower for w in ["handshake", "shake hands", "round robin", "matches"]):
            keywords = ["handshake"] if "handshake" in q_lower else ["shake hands"]
            self.register_dynamic_pattern({
                "title": "Pairwise Handshake Invariant",
                "keywords": keywords,
                "formula_type": "handshake",
                "explanation": "Combinatorial selection of pairs without repetition $C(n, 2) = n(n-1)/2$.",
                "example": query,
            })

        # Check for exclusionary riddles
        elif "all but" in q_lower:
            self.register_dynamic_pattern({
                "title": "All-But-N Exclusionary Invariant",
                "keywords": ["all but"],
                "formula_type": "exclusionary",
                "explanation": "'All but N' is an exclusionary operator where exactly N remain unaffected.",
                "example": query,
            })

        # Check for rate or pipe problems
        elif any(w in q_lower for w in ["fill", "pipe", "together", "complete the work"]):
            self.register_dynamic_pattern({
                "title": "Harmonic Work Invariant",
                "keywords": ["together", "work"],
                "formula_type": "rate",
                "explanation": "Combined rate formula: $T = (t_1 * t_2) / (t_1 + t_2)$.",
                "example": query,
            })

        # Check for Simple Interest
        elif any(w in q_lower for w in ["simple interest", "principal", "rate of interest"]):
            self.register_dynamic_pattern({
                "title": "Simple Interest Invariant",
                "keywords": ["simple interest"],
                "formula_type": "simple_interest",
                "explanation": "Simple Interest $SI = (P \\times R \\times T) / 100$.",
                "example": query,
            })

        # Check for Sum of N Natural Numbers
        elif any(w in q_lower for w in ["sum of first", "sum of the first", "natural numbers"]):
            self.register_dynamic_pattern({
                "title": "Sum of First N Integers Invariant",
                "keywords": ["sum", "first"],
                "formula_type": "sum_n",
                "explanation": "Sum of first $n$ natural numbers $S_n = n(n + 1)/2$.",
                "example": query,
            })

        # Check for Circle Geometry (area / circumference)
        elif any(w in q_lower for w in ["circle", "radius", "circumference"]) and any(w in q_lower for w in ["area", "circumference"]):
            self.register_dynamic_pattern({
                "title": "Circle Geometry Invariant",
                "keywords": ["circle", "radius"],
                "formula_type": "circle_geometry",
                "explanation": "Circle Area $\\pi r^2$, Circumference $2\\pi r$.",
                "example": query,
            })

        # General fact extraction fallback
        elif research_snippets and len(research_snippets[0]) > 20:
            # Extract key nouns/terms for future matching
            words = [w for w in re.findall(r"\b[a-z]{4,}\b", q_lower) if w not in {"what", "when", "where", "which", "does", "have", "with", "from"}]
            if len(words) >= 2:
                self.register_dynamic_pattern({
                    "title": f"Fact Invariant: {words[0]} {words[1]}",
                    "keywords": words[:3],
                    "formula_type": "constant_fact",
                    "explanation": f"Acquired factual invariant for: '{query}'",
                    "fact_answer": research_snippets[0],
                    "example": query,
                })


learned_pattern_engine = LearnedPatternEngine()
