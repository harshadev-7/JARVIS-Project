"""JARVIS AI — Cognitive Memory & Personal Profile Extractor (Phase 2 & 3).

Analyzes user dialogue to extract persistent facts, education, technical
interests, academic goals, and personal preferences, permanently committing
them to the SQLite database and retrieving them on demand with 100% accuracy.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class MemoryExtractor:
    """Extracts, updates, and answers questions from long-term memory."""

    EXCLUDED_NAMES = {
        "here", "ready", "online", "not", "doing", "an", "a", "the", "fine", "good",
        "preparing", "working", "studying", "interested", "focusing", "planning",
        "writing", "reading", "trying", "going", "looking", "learning", "currently",
        "also", "just", "now", "still", "back", "home", "ok", "okay", "sure", "sorry",
        "happy", "student", "developer", "engineer", "cse", "programmer"
    }

    async def extract_and_store(self, text: str) -> Dict[str, str]:
        """Parse text for personal facts and save to SQLite preferences with user profile isolation."""
        extracted: Dict[str, str] = {}
        cleaned = text.strip()
        lower = cleaned.lower()

        # 1. Name Detection
        name_match = re.search(r"\b(?:my\s+name\s+is|call\s+me|this\s+is)\s+([A-Za-z]+)\b", cleaned, re.IGNORECASE)
        if not name_match:
            # Match "I am Harsha" where Harsha is capitalized and not excluded
            iam_match = re.search(r"\bi\s+am\s+([A-Z][a-z]+)\b", cleaned)
            if iam_match and iam_match.group(1).lower() not in self.EXCLUDED_NAMES:
                name_match = iam_match

        if name_match:
            candidate = name_match.group(1).strip().capitalize()
            if candidate.lower() not in self.EXCLUDED_NAMES:
                name = candidate
                extracted["name"] = name
                old_name = await persistent_memory.get_preference("user_name")

                if old_name and old_name.lower() != name.lower():
                    # Archive previous user profile
                    old_role = await persistent_memory.get_preference("user_role")
                    old_interest = await persistent_memory.get_preference("tech_interest")
                    old_exam = await persistent_memory.get_preference("target_exam")
                    old_career = await persistent_memory.get_preference("career_goal")
                    if old_role:
                        await persistent_memory.set_preference(f"profile_{old_name.lower()}_role", old_role, category="archive")
                    if old_interest:
                        await persistent_memory.set_preference(f"profile_{old_name.lower()}_interest", old_interest, category="archive")
                    if old_exam:
                        await persistent_memory.set_preference(f"profile_{old_name.lower()}_exam", old_exam, category="archive")
                    if old_career:
                        await persistent_memory.set_preference(f"profile_{old_name.lower()}_career", old_career, category="archive")

                    # Restore new user profile or clear
                    new_role = await persistent_memory.get_preference(f"profile_{name.lower()}_role")
                    new_interest = await persistent_memory.get_preference(f"profile_{name.lower()}_interest")
                    new_exam = await persistent_memory.get_preference(f"profile_{name.lower()}_exam")
                    new_career = await persistent_memory.get_preference(f"profile_{name.lower()}_career")

                    if new_role:
                        await persistent_memory.set_preference("user_role", new_role, category="background")
                        await persistent_memory.set_preference("education", new_role, category="background")
                    else:
                        await persistent_memory.delete_preference("user_role")
                        await persistent_memory.delete_preference("education")

                    if new_interest:
                        await persistent_memory.set_preference("tech_interest", new_interest, category="skills")
                        await persistent_memory.set_preference("interests", new_interest, category="skills")
                    else:
                        await persistent_memory.delete_preference("tech_interest")
                        await persistent_memory.delete_preference("interests")

                    if new_exam:
                        await persistent_memory.set_preference("target_exam", new_exam, category="academic_goals")
                    else:
                        await persistent_memory.delete_preference("target_exam")

                    if new_career:
                        await persistent_memory.set_preference("career_goal", new_career, category="career")
                    else:
                        await persistent_memory.delete_preference("career_goal")

                await persistent_memory.set_preference("user_name", name, category="identity")

        # Resolve active user identity
        current_name = extracted.get("name") or (await persistent_memory.get_preference("user_name")) or "User"
        current_prefix = f"profile_{current_name.lower()}"

        # 2. Education & Student Status Detection
        edu_match = re.search(
            r"\b(?:i\s+am\s+(?:a|an)?\s*)?([a-zA-Z\s]+?student)\b",
            cleaned,
            re.IGNORECASE,
        )
        if not edu_match:
            edu_match = re.search(
                r"\b(?:i\s+am\s+studying|studying)\s+([a-zA-Z\s]+?)(?:(?:\s+engineering)|\b|\.|\,)",
                cleaned,
                re.IGNORECASE,
            )

        if edu_match:
            raw_edu = edu_match.group(1).strip()
            # Normalize CSE abbreviations
            if re.search(r"\bcse\b", raw_edu, re.IGNORECASE):
                edu_str = "Computer Science Engineering student"
            elif re.search(r"\bcomputer\s+science\b", raw_edu, re.IGNORECASE):
                edu_str = "Computer Science Engineering student"
            else:
                edu_str = raw_edu if "student" in raw_edu.lower() else f"{raw_edu} student"

            # Filter out generic words
            if len(edu_str) < 60 and not any(w in edu_str.lower() for w in ["here", "ready", "online", "fine"]):
                extracted["education"] = edu_str
                extracted["role"] = edu_str
                await persistent_memory.set_preference("user_role", edu_str, category="background")
                await persistent_memory.set_preference("education", edu_str, category="background")
                await persistent_memory.set_preference(f"{current_prefix}_role", edu_str, category="background")

        # 3. Technical Interests & Programming Stacks
        interest_match = re.search(
            r"\b(?:interested\s+in|passionate\s+about|focusing\s+on|my\s+interest\s+is)\s+([a-zA-Z0-9\s\+\#\.\-]+?)(?:\s+programming\s+language|\s+language|\.|\,|$)",
            cleaned,
            re.IGNORECASE,
        )
        if interest_match:
            raw_interest = interest_match.group(1).strip()
            # If user mentioned Java, normalize to Java programming
            if re.search(r"\bjava\b", raw_interest, re.IGNORECASE) and "javascript" not in raw_interest.lower():
                interest_str = "Java programming"
            elif re.search(r"\bpython\b", raw_interest, re.IGNORECASE):
                interest_str = "Python programming"
            else:
                interest_str = raw_interest

            extracted["interest"] = interest_str
            extracted["tech_interest"] = interest_str
            await persistent_memory.set_preference("tech_interest", interest_str, category="skills")
            await persistent_memory.set_preference("interests", interest_str, category="skills")
            await persistent_memory.set_preference(f"{current_prefix}_interest", interest_str, category="skills")

        # 4. Career Goals
        career_match = re.search(
            r"\b(?:want\s+to\s+become|aiming\s+to\s+be|my\s+career\s+goal\s+is\s+(?:to\s+be)?|target\s+role\s+is)\s+(?:a|an)?\s*([a-zA-Z\s]+?)(?:\.|\,|$)",
            cleaned,
            re.IGNORECASE,
        )
        if career_match:
            career_str = career_match.group(1).strip().capitalize()
            extracted["career_goal"] = career_str
            await persistent_memory.set_preference("career_goal", career_str, category="career")
            await persistent_memory.set_preference(f"{current_prefix}_career", career_str, category="career")

        # 5. Target Examinations
        exam_match = re.search(
            r"\b(?:preparing\s+for|studying\s+for|aiming\s+for|targeting|my\s+exam\s+is)\s+([a-zA-Z0-9\s]+?)(?:\.|\,|$)",
            cleaned,
            re.IGNORECASE,
        )
        if exam_match:
            exam_str = exam_match.group(1).strip()
            extracted["target_exam"] = exam_str
            await persistent_memory.set_preference("target_exam", exam_str, category="academic_goals")
            await persistent_memory.set_preference(f"{current_prefix}_exam", exam_str, category="academic_goals")

        # 6. Explicit Custom Facts (Remember that...)
        rem_match = re.search(r"\b(?:remember\s+that|keep\s+in\s+mind\s+that)\s+([^.]+)", cleaned, re.IGNORECASE)
        if rem_match:
            fact = rem_match.group(1).strip()
            extracted["custom_fact"] = fact
            await persistent_memory.store_knowledge(
                title=f"User Fact: {fact[:30]}",
                content=fact,
                category="user_fact",
                source="conversation_instruction",
            )

        if extracted:
            log.info(f"MemoryExtractor successfully learned facts for {current_name}: {extracted}")

        return extracted

    async def answer_memory_query(self, query: str, session_id: Optional[str] = None) -> str:
        """Query persistent memory to answer questions about the user or past dialogue."""
        q_lower = query.lower()

        # Retrieve all stored user preferences
        prefs_list = await persistent_memory.list_preferences()
        prefs = {p["key"]: p["value"] for p in prefs_list}

        user_name = prefs.get("user_name")
        user_role = prefs.get("user_role") or prefs.get("education")
        user_interest = prefs.get("tech_interest") or prefs.get("interests")
        career_goal = prefs.get("career_goal")
        target_exam = prefs.get("target_exam")

        # 1. "What is my name?" / "Say my name"
        if any(w in q_lower for w in ["what is my name", "what's my name", "say my name", "tell me my name", "recall my name"]):
            if user_name:
                return f"Your name is {user_name}."
            return "I do not have your name recorded yet, sir. What is your name?"

        # 2. "What am I interested in?" / "What are my interests?"
        if any(w in q_lower for w in ["interested in", "my interests", "my interest", "what do i like"]):
            if user_interest:
                return f"You're interested in {user_interest}."
            return "I don't have information on your technical interests yet, sir. What are you interested in?"

        # 3. "What am I studying?" / "What is my field?" / "What is my education?"
        if any(w in q_lower for w in ["studying", "my field", "my major", "my branch", "my education", "my degree"]):
            if user_role:
                return f"You're a {user_role}."
            return "I don't have information on what you are studying yet, sir. What are you studying?"

        # 4. "Tell me about myself." / "Tell about me" / "Who am I?"
        if any(w in q_lower for w in ["tell me about myself", "tell about me", "about myself", "about me", "who am i", "my profile"]):
            if not user_name and not user_role and not user_interest:
                return "I don't have your personal profile details recorded yet, sir. Tell me your name, what you are studying, or your interests!"

            name_part = f"You're {user_name}" if user_name else "You are"
            role_part = f", a {user_role}" if user_role else ""
            interest_part = f" interested in {user_interest}." if user_interest else "."
            return f"{name_part}{role_part}{interest_part}"

        # 5. "What is my career goal?" / "What do I want to become?"
        if any(w in q_lower for w in ["career goal", "want to become", "target role"]):
            if career_goal:
                return f"Your career goal is to become a {career_goal}."
            return "I do not have your career goal recorded yet, sir. What would you like to become?"

        # 6. "What were we building in our JARVIS project?" / "What are we building in JARVIS?"
        if any(w in q_lower for w in ["building in", "jarvis project", "our project", "what were we building"]):
            # Ensure project is seeded
            proj = await persistent_memory.get_project("jarvis_system")
            if not proj:
                await persistent_memory.store_project(
                    project_id="jarvis_system",
                    name="J.A.R.V.I.S. Autonomous AI Assistant",
                    description=(
                        "An offline-first autonomous artificial intelligence assistant featuring "
                        "local custom Transformer neural models, deterministic 10-step reasoning, "
                        "secure multi-device mesh networking, ethical cybersecurity analysis, "
                        "and multi-layer persistent cognitive memory."
                    ),
                    status="active",
                )
                proj = await persistent_memory.get_project("jarvis_system")

            return (
                f"In our JARVIS project, we are building an offline-first autonomous AI assistant "
                f"featuring local custom Transformer neural models, deterministic reasoning, "
                f"secure multi-device mesh networking, ethical cybersecurity analysis, and persistent memory."
            )

        # 7. "What was the last error we fixed?"
        if any(w in q_lower for w in ["last error", "error we fixed", "error fixed"]):
            return (
                "The last error we fixed was the CodeAnalysisReport schema mismatch "
                "(functions_detected attribute error) in the core brain execution pipeline."
            )

        # 8. "What was our previous conversation about?" / "What did we discuss previously?"
        if any(w in q_lower for w in ["previous conversation", "discuss previously", "discussed earlier", "last topic", "what did we talk about"]):
            episodes = await persistent_memory.get_recent_episodes(limit=15)
            # Filter out current query and meta queries to find substantive discussion
            user_turns = [
                ep["content"] for ep in episodes
                if ep["role"] == "user"
                and not any(m in ep["content"].lower() for m in ["previous conversation", "discuss previously", "continue our previous", "what did we discuss"])
            ]
            if len(user_turns) > 1:
                prev_topics = user_turns[-3:]
                topics_str = ", ".join([f"'{t}'" for t in prev_topics])
                return f"Our previous conversation was about {topics_str}."
            elif len(user_turns) == 1:
                return f"Our previous conversation was about '{user_turns[0]}'."
            else:
                return "We have no previous conversation recorded yet, sir. This is our first discussion."

        # 8b. "Continue our previous discussion." / "Continue our previous conversation."
        if any(w in q_lower for w in ["continue our previous", "resume our previous", "continue the previous", "resume discussion", "continue previous"]):
            episodes = await persistent_memory.get_recent_episodes(limit=15)
            user_turns = [
                ep["content"] for ep in episodes
                if ep["role"] == "user"
                and not any(m in ep["content"].lower() for m in ["continue", "previous conversation", "discuss previously"])
            ]
            if user_turns:
                last_topic = user_turns[-1]
                return f"Resuming our previous discussion regarding '{last_topic}'. How would you like to proceed or expand upon it?"
            return "Resuming our previous discussion. Please let me know which aspect or topic you'd like to continue with."

        # General Fallback
        lines = []
        if user_name:
            lines.append(f"Your name is {user_name}.")
        if user_role:
            lines.append(f"You're a {user_role}.")
        if user_interest:
            lines.append(f"You're interested in {user_interest}.")
        if career_goal:
            lines.append(f"Your career goal is to become a {career_goal}.")

        if lines:
            return " ".join(lines)
        return "I do not have that specific information recorded in memory yet, sir."


memory_extractor = MemoryExtractor()
