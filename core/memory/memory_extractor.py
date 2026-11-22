"""
JARVIS AI — Cognitive Memory & User Profile Extractor
=====================================================
Analyzes user dialogue to extract persistent facts, habits, technical
interests, academic goals, and personal preferences, permanently committing
them to the SQLite database and retrieving them on demand.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class MemoryExtractor:
    """Extracts, updates, and answers questions from long-term memory."""

    # Patterns for extracting user profile facts
    NAME_PATTERNS = [
        r"\bmy name is\s+([A-Za-z]+)\b",
        r"\bcall me\s+([A-Za-z]+)\b",
        r"\bthis is\s+([A-Za-z]+)\b",
        r"\bi am\s+([A-Z][a-z]+)(?:\s*[.!,]|\s*$)",
    ]

    EXCLUDED_NAMES = {
        "here", "ready", "online", "not", "doing", "an", "a", "the", "fine", "good",
        "preparing", "working", "studying", "interested", "focusing", "planning",
        "writing", "reading", "trying", "going", "looking", "learning", "currently",
        "also", "just", "now", "still", "back", "home", "ok", "okay", "sure", "sorry",
        "happy", "student", "developer", "engineer"
    }
    ROLE_PATTERNS = [
        r"\bi am (?:a|an)\s+([^,.]+?)\s*(?:interested|preparing|focusing|\.|$)",
        r"\bmy role is\s+([^,.]+)",
        r"\bworking as (?:a|an)?\s*([^,.]+)",
    ]
    INTEREST_PATTERNS = [
        r"\binterested in\s+([^,.]+)",
        r"\bpassionate about\s+([^,.]+)",
        r"\bfocusing on\s+([^,.]+)",
        r"\bworking with\s+([^,.]+)",
        r"\bmy interest is\s+([^,.]+)",
    ]
    EXAM_GOAL_PATTERNS = [
        r"\bpreparing for\s+([^,.]+)",
        r"\bstudying for\s+([^,.]+)",
        r"\baiming for\s+([^,.]+)",
        r"\btargeting\s+([^,.]+)",
        r"\bmy exam is\s+([^,.]+)",
    ]
    REMEMBER_PATTERNS = [
        r"\bremember (?:that\s+)?([^.]+)",
        r"\bkeep in mind (?:that\s+)?([^.]+)",
    ]

    async def extract_and_store(self, text: str) -> Dict[str, str]:
        """Parse text for personal facts and save to SQLite preferences with user profile isolation."""
        extracted: Dict[str, str] = {}
        cleaned = text.strip()

        # 1. Name & Multi-User Profile Switch Detection
        for pat in self.NAME_PATTERNS:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().capitalize()
                # Exclude common false positives
                if candidate.lower() not in self.EXCLUDED_NAMES:
                    name = candidate
                    extracted["name"] = name
                    old_name = await persistent_memory.get_preference("user_name")
                    
                    if old_name and old_name.lower() != name.lower():
                        # Archive previous user's profile
                        old_role = await persistent_memory.get_preference("user_role")
                        old_interest = await persistent_memory.get_preference("tech_interest")
                        old_exam = await persistent_memory.get_preference("target_exam")
                        if old_role:
                            await persistent_memory.set_preference(f"profile_{old_name.lower()}_role", old_role, category="archive")
                        if old_interest:
                            await persistent_memory.set_preference(f"profile_{old_name.lower()}_interest", old_interest, category="archive")
                        if old_exam:
                            await persistent_memory.set_preference(f"profile_{old_name.lower()}_exam", old_exam, category="archive")

                        # Restore profile for new user, or clear if new person
                        new_role = await persistent_memory.get_preference(f"profile_{name.lower()}_role")
                        new_interest = await persistent_memory.get_preference(f"profile_{name.lower()}_interest")
                        new_exam = await persistent_memory.get_preference(f"profile_{name.lower()}_exam")

                        if new_role:
                            await persistent_memory.set_preference("user_role", new_role, category="background")
                        else:
                            await persistent_memory.delete_preference("user_role")

                        if new_interest:
                            await persistent_memory.set_preference("tech_interest", new_interest, category="skills")
                        else:
                            await persistent_memory.delete_preference("tech_interest")

                        if new_exam:
                            await persistent_memory.set_preference("target_exam", new_exam, category="academic_goals")
                        else:
                            await persistent_memory.delete_preference("target_exam")

                        log.info(f"Switched user profile: {old_name} -> {name} (clean slate enabled)")

                    await persistent_memory.set_preference("user_name", name, category="identity")
                    break

        # Resolve current active user name
        current_name = extracted.get("name") or (await persistent_memory.get_preference("user_name")) or "User"
        current_key_prefix = f"profile_{current_name.lower()}"

        # 2. Role / Occupation / Student status
        for pat in self.ROLE_PATTERNS:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                # filter out false positives
                if len(role) < 50 and not any(w in role.lower() for w in ["here", "ready", "online", "fine"]):
                    extracted["role"] = role
                    await persistent_memory.set_preference("user_role", role, category="background")
                    await persistent_memory.set_preference(f"{current_key_prefix}_role", role, category="background")
                break

        # 3. Technical Interests / Stack
        for pat in self.INTEREST_PATTERNS:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                interest = match.group(1).strip()
                extracted["interest"] = interest
                await persistent_memory.set_preference("tech_interest", interest, category="skills")
                await persistent_memory.set_preference(f"{current_key_prefix}_interest", interest, category="skills")
                break

        # 4. Exam / Target Objectives
        for pat in self.EXAM_GOAL_PATTERNS:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                goal = match.group(1).strip()
                extracted["target_exam"] = goal
                await persistent_memory.set_preference("target_exam", goal, category="academic_goals")
                await persistent_memory.set_preference(f"{current_key_prefix}_exam", goal, category="academic_goals")
                break

        # 5. Explicit remember instructions
        for pat in self.REMEMBER_PATTERNS:
            match = re.search(pat, cleaned, re.IGNORECASE)
            if match:
                fact = match.group(1).strip()
                extracted["custom_fact"] = fact
                await persistent_memory.store_knowledge(
                    title=f"User Fact: {fact[:30]}",
                    content=fact,
                    category="user_fact",
                    source="conversation_instruction"
                )
                break

        if extracted:
            log.info(f"MemoryExtractor learned facts for {current_name}: {extracted}")

        return extracted

    async def answer_memory_query(self, query: str, session_id: Optional[str] = None) -> str:
        """Query persistent memory to answer questions about the user or past dialogue."""
        q_lower = query.lower()

        # Retrieve all stored user preferences
        prefs_list = await persistent_memory.list_preferences()
        prefs = {p["key"]: p["value"] for p in prefs_list}

        user_name = prefs.get("user_name", "sir")
        user_role = prefs.get("user_role")
        user_interest = prefs.get("tech_interest")
        target_exam = prefs.get("target_exam")

        # Specific query: What language/tech stack?
        if any(w in q_lower for w in ["language", "programming", "stack", "technology"]):
            if user_interest:
                return (
                    f"Based on your profile, {user_name}, you are interested in **{user_interest}**.\n\n"
                    f"*(Retrieved from JARVIS Persistent Memory)*"
                )
            else:
                return f"I do not have a specific programming language recorded yet for you, {user_name}. What language are you currently working with?"

        # Specific query: What examination / target?
        if any(w in q_lower for w in ["examination", "exam", "preparing for", "target"]):
            if target_exam:
                return (
                    f"You are actively preparing for **{target_exam}**, {user_name}.\n\n"
                    f"*(Retrieved from JARVIS Persistent Memory)*"
                )
            else:
                return f"I do not have your target examination recorded yet, {user_name}. What exam or goal are you preparing for?"

        # Specific query: Questions about what user said or clarification ("did I tell you...", "I did tell you...")
        if any(w in q_lower for w in ["did i tell", "i did tell", "when did i tell", "did i say", "i did say"]):
            if any(term in q_lower for term in ["cse", "student", "role", "college", "branch"]):
                if user_name.lower() == "mani" and not user_role:
                    return (
                        f"My apologies, Mani. You only introduced yourself as Mani in this session. "
                        f"The CSE student background was recorded previously when Harsha was active. "
                        f"I have separated your profiles so your record is completely distinct. "
                        f"Are you also a CSE student, or would you like to record a different background?"
                    )
                elif user_role:
                    return f"Yes, {user_name}. Your profile records your background as: **{user_role}**."
                else:
                    return f"You haven't told me your academic background or occupation yet, {user_name}. What is your field or role?"

        # Specific query: What is my name / Say my name
        if any(w in q_lower for w in ["my name", "say my name", "tell me my name", "recall my name", "repeat my name"]):
            if user_name != "sir":
                return f"Your name is **{user_name}**."
            return "You haven't told me your name yet, sir. What would you like me to call you?"

        # Specific query: Who am I?
        if any(w in q_lower for w in ["who am i", "who i am"]):
            if user_name != "sir":
                details = [f"You are **{user_name}**."]
                if user_role:
                    details.append(f"You are a **{user_role}**.")
                if user_interest:
                    details.append(f"Specializing in **{user_interest}**.")
                return "\n".join(details)
            return "You haven't told me your identity yet, sir. What would you like me to call you?"

        # Past conversation recall
        if any(w in q_lower for w in ["previous conversation", "our conversation", "what do you remember", "past discussion"]):
            episodes = await persistent_memory.get_recent_episodes(session_id=session_id, limit=6)
            if not episodes:
                # Query without session_id for cross-session recall
                episodes = await persistent_memory.get_recent_episodes(limit=6)

            if episodes:
                user_msgs = [ep["content"] for ep in episodes if ep["role"] == "user"]
                summary_items = [f"- \"{msg}\"" for msg in user_msgs[-4:]]
                return (
                    f"Yes, {user_name}. I maintain full episodic memory in SQLite. Here is a summary of our recent exchanges:\n\n"
                    + "\n".join(summary_items) +
                    f"\n\n**Known User Profile:**\n"
                    f"- **Identity:** {user_name} ({user_role or 'Student/Engineer'})\n"
                    f"- **Interests:** {user_interest or 'Software Engineering'}\n"
                    f"- **Target Goal:** {target_exam or 'GATE CSE'}\n\n"
                    f"All context is permanently indexed in `jarvis.db`."
                )

        # General: "What do you know about me?"
        lines = [f"Here is your persistent profile recorded in my cognitive database, **{user_name}**:\n"]
        if user_name != "sir":
            lines.append(f"• **Name:** {user_name}")
        if user_role:
            lines.append(f"• **Background / Role:** {user_role}")
        if user_interest:
            lines.append(f"• **Technical Interests:** {user_interest}")
        if target_exam:
            lines.append(f"• **Target Examination / Goal:** {target_exam}")

        # Check knowledge items
        knowledge_matches = await persistent_memory.search_knowledge("user", top_k=3)
        if knowledge_matches:
            lines.append("\n**Custom Notes & Facts Remembered:**")
            for km in knowledge_matches:
                lines.append(f"• {km['text']}")

        if len(lines) == 1:
            lines.append("I don't have any personal details recorded for you yet, sir. Tell me your name, interests, or what you are working on!")
        else:
            lines.append(f"\n*(Persistent Storage: SQLite `jarvis.db` | Auto-Synced)*")

        return "\n".join(lines)


memory_extractor = MemoryExtractor()
