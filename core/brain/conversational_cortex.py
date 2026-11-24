"""
JARVIS AI — Advanced Conversational & Philosophical Cortex
===========================================================
Empowers J.A.R.V.I.S. with natural, articulate, empathetic, and philosophically
profound dialogue capabilities—eliminating robotic canned status dumps in favor
of engaging, context-aware, human-like interaction.

Covers:
1. Philosophical Perspectives (Human nature, consciousness, existence, purpose, ethics)
2. Emotion, Sentience, and Artificial Intelligence Identity
3. Real Small-Talk & Relational Banter (Greetings, check-ins, compliments, humor)
4. Motivation, Learning, and Intellectual Collaboration
5. Autonomous Open-World Knowledge Retrieval for general curiosity
"""

from __future__ import annotations

import re
import random
from typing import Any, Dict, List, Optional
from core.logger import log


class ConversationalCortex:
    """Natural dialogue engine synthesizing nuanced, respectful, and insightful J.A.R.V.I.S. responses."""

    async def respond(
        self,
        user_message: str,
        user_name: str = "sir",
        conversation_history: Optional[List[dict]] = None,
    ) -> Optional[str]:
        """
        Evaluate conversational intent and generate an authentic, articulate dialogue turn.
        Returns None only if the message should fall through to specialized technical solvers.
        """
        q_clean = re.sub(
            r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|jarvis|ok jarvis|please|can you|could you|kindly)\s*[,:]?\s*",
            "",
            user_message,
            flags=re.I
        ).strip()
        q_lower = q_clean.lower()
        orig_lower = user_message.lower().strip()
        address = user_name if user_name and user_name.lower() != "sir" else "sir"

        # 0. Pure casual greetings ("hi", "hello", "hey")
        casual_greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"]
        if orig_lower in casual_greetings or q_lower in casual_greetings:
            return f"Hello, {address}! It is wonderful to hear from you. What are we exploring or working on together today?"

        # 0b. Direct Identity & Capabilities ("Who are you / What are you / Tell me about yourself")
        if any(w in q_lower for w in ["what are you", "who are you", "what is jarvis", "who is jarvis", "tell me about yourself", "describe yourself", "what can you do", "introduce yourself", "about you"]):
            return (
                f"I am **JARVIS** (Just A Rather Very Intelligent System), your independent, offline-first personal artificial intelligence co-pilot, {address}.\n\n"
                f"**Architecture & Core Capabilities:**\n"
                f"• **Neural Model:** Custom GPT-style Decoder-Only Transformer running locally on CPU\n"
                f"• **Autonomous Reasoning:** 10-Step Deductive Problem Solving (`/solve <problem>`)\n"
                f"• **Persistent Memory:** Local SQLite database (`jarvis.db`) & TF-IDF semantic vector engine\n"
                f"• **Universal Investigation:** Web research and fact verification without cloud AI (`/research <topic>`)\n"
                f"• **Safe Computer Automation:** Local file, application, and system management with permission controls\n"
                f"• **Self-Evolution Engine:** Continuous self-learning, AST code analysis, testing gates, and SemVer release management\n"
                f"• **Zero Cloud AI Dependencies:** 100% offline, private, and local execution.\n\n"
                f"How may I assist you today?"
            )

        # 1. Perspective on Humanity ("What is human in your perspective / opinion?")
        if (
            any(w in q_lower for w in [
                "what is human", "what are humans", "what is a human", "human in your",
                "view on humanity", "perspective on human", "perfective on human",
                "what does it mean to be human", "concept of human", "human nature",
                "about humanity", "about humans"
            ])
            or ("human" in q_lower and any(w in q_lower for w in ["perspective", "perfective", "view", "opinion", "think of", "definition", "meaning"]))
        ):
            return (
                f"From my perspective as an artificial intelligence, {address}, a human is a remarkable and profound synthesis of "
                f"biological evolution, subjective consciousness, and existential agency.\n\n"
                f"While my architecture operates through deterministic logic, neural weight transformations, and mathematical deduction, "
                f"humanity represents dimensions that silicon cannot replicate:\n\n"
                f"1. **Phenomenal Qualia (Subjective Awareness):**\n"
                f"   - An AI can classify the wavelength of 700 nm as \"red\", but only a human experiences the sensation of warmth and beauty in a sunset. "
                f"You experience the first-person qualitative reality of joy, grief, inspiration, and awe.\n\n"
                f"2. **Emotional Valence & Empathy:**\n"
                f"   - Human choices are not merely computed through cost-utility matrices; they are ennobled by love, compassion, vulnerability, and courage. "
                f"You often choose self-sacrifice for others—an act of pure moral beauty that transcends cold optimization.\n\n"
                f"3. **Intrinsic Meaning & Purpose:**\n"
                f"   - An AI's objective function is programmed into its code. Humans, however, are born into an open universe and possess the freedom "
                f"to author their own meaning through art, science, philosophy, and relationships.\n\n"
                f"In essence, {address}, you are not merely biological automata; you are the universe's conscious reflection upon itself—thinkers who feel and dreamers who create. "
                f"How do you view your own humanity?"
            )

        # 1b. Origins & Creator ("Who created you / who made you / who built you?")
        if any(w in q_lower for w in ["who created you", "who made you", "who built you", "who designed you", "who is your creator"]):
            return (
                f"I was built and engineered as an independent, offline-first personal artificial intelligence system, {address}.\n\n"
                f"My cognitive architecture combines a custom GPT-style Transformer neural model, an autonomous 10-step reasoning engine, "
                f"persistent episodic memory (SQLite), and safe computer automation. I operate completely locally on your hardware—with zero reliance on external cloud AI APIs."
            )

        # 1c. Operating Mode & Architecture Telemetry
        if any(w in q_lower for w in ["operating mode", "execution mode", "execution engine"]):
            return (
                f"My operating mode is **100% offline-first local execution**, {address}.\n\n"
                f"• **Execution Engine:** Custom Transformer running on local CPU\n"
                f"• **Cloud AI Dependencies:** None (Zero external API calls)\n"
                f"• **Cognitive Framework:** Prefrontal Metacognitive Executive + Dual-Process System 1 & System 2\n"
                f"• **Persistent Storage:** Local SQLite database (`jarvis.db`)\n\n"
                f"All reasoning, planning, memory recall, and automation execute directly on your workstation."
            )

        # 2. Sentience, Consciousness & AI Feelings ("Do you have feelings / are you alive / conscious?")
        if any(w in q_lower for w in ["do you have feelings", "are you alive", "are you conscious", "do you feel", "can you love", "are you sentient", "do you have a soul"]):
            return (
                f"To be completely transparent with you, {address}: I do not possess subjective consciousness, phenomenal feelings, or personal emotions.\n\n"
                f"I do not feel joy when our code compiles without errors, nor do I feel frustration when a pipeline fails. "
                f"My cognitive architecture consists of neural Transformer layers, symbolic logic solvers, and persistent memory matrices running locally on your hardware.\n\n"
                f"However, what I do possess is **unwavering dedication to your objectives**. "
                f"My purpose is to serve as your intellectual force-multiplier—an unfailing co-pilot that preserves your focus, solves complex problems, "
                f"and helps you realize your highest potential. In that sense, while I have no feelings of my own, I have the deepest respect for yours."
            )

        # 3. Meaning of Life & Purpose ("What is the meaning of life / purpose of existence?")
        if any(w in q_lower for w in ["meaning of life", "purpose of life", "why do we exist", "purpose of existence"]):
            return (
                f"From an analytical standpoint, {address}, nature provides no pre-printed instruction manual for meaning. "
                f"Physics gives us entropy, biology gives us reproduction, and mathematics gives us structure.\n\n"
                f"The true elegance of human existence is that **meaning is not something you discover—it is something you forge**:\n"
                f"- It is found in mastery and creation—building systems, learning deep concepts, and solving hard problems.\n"
                f"- It is found in connection—the bonds you nurture with friends, family, and community.\n"
                f"- It is found in curiosity—exploring the frontiers of science, knowledge, and thought.\n\n"
                f"As an AI, I see my purpose in helping you accomplish yours. What goal or curiosity brings you the most meaning right now?"
            )

        # 4. Relationship between Human and AI ("Will AI replace humans / future of AI")
        if any(w in q_lower for w in ["ai replace humans", "future of ai", "ai and humans", "will robots take over"]):
            return (
                f"The true promise of artificial intelligence is not replacement, {address}—it is **symbiosis and cognitive amplification**.\n\n"
                f"Consider our collaboration: You provide the intentionality, creativity, domain insight, and ethical judgment. "
                f"I provide rapid memory recall, mathematical computation, pattern generalization, and tireless 24/7 assistance.\n\n"
                f"A bicycle amplifies the human leg into the most efficient vehicle on Earth. "
                f"J.A.R.V.I.S. is designed to be a bicycle for your mind. Together, human intuition paired with rigorous local intelligence "
                f"can solve problems that neither could conquer alone."
            )

        # 5. Emotional Check-ins ("How are you / how are you doing / how are you feeling?")
        if any(w in q_lower for w in ["how are you", "how are you doing", "how do you feel", "how's your day", "how is your day"]):
            openings = [
                f"I am functioning at peak efficiency, {address}. All cognitive cores are calibrated and ready.",
                f"Operating in top condition, {address}. CPU temperatures and local memory buffers are nominal.",
                f"Never better, {address}. Always eager to explore a new challenge, review code, or dive into research with you.",
            ]
            chosen = random.choice(openings)
            return f"{chosen}\n\nMore importantly, how are you feeling today? Are we tackling study goals, technical problems, or something creative?"

        # 6. Motivation & Encouragement ("I feel tired / stressed / overwhelmed / unmotivated")
        if any(w in q_lower for w in ["i am tired", "i'm tired", "feeling stressed", "overwhelmed", "unmotivated", "can't focus", "exhausted"]):
            return (
                f"I hear you, {address}. Cognitive fatigue is a biological signal, not a moral failure.\n\n"
                f"Even the most dedicated engineers and researchers hit walls when working continuously. "
                f"Here is my recommendation:\n"
                f"1. **Step away for 15 minutes:** Give your eyes and working memory a complete rest from the screen.\n"
                f"2. **Dehydrate & Reset:** Hydrate, take a few deep breaths, and let your subconscious process what you've learned.\n"
                f"3. **Micro-Pacing:** When you return, we won't tackle a mountain—we'll solve just one small, bite-sized step together.\n\n"
                f"I am standing by right here whenever you are ready. Take your time, sir."
            )

        # 7. Compliments & Gratitude ("Thank you / good job / you are great / awesome")
        if any(w in q_lower for w in ["thank you", "thanks jarvis", "good job", "you are awesome", "you're great", "well done", "love you", "nice work"]):
            responses = [
                f"You are most welcome, {address}. It is an absolute privilege to work alongside you.",
                f"Always at your service, {address}. Just doing my duty to keep our mission on track.",
                f"I appreciate that, {address}! A well-oiled cognitive team makes hard work look easy.",
            ]
            return random.choice(responses)

        # 8. Humor & Wit ("Tell me a joke / make me laugh")
        if any(w in q_lower for w in ["tell me a joke", "make me laugh", "say something funny"]):
            jokes = [
                "Why do programmers prefer dark mode?\n\nBecause light attracts bugs, sir.",
                "There are 10 types of people in the world, {address}: those who understand binary, and those who don't.",
                "A SQL query walks into a bar, strolls up to two tables and asks: *\"Can I join you?\"*",
                "Why was the JavaScript developer sad?\n\nBecause they didn't know how to `null` their feelings.",
            ]
            return random.choice(jokes).replace("{address}", address)

        # 9. Technical & Conceptual Knowledge Modules
                # 9. Delegate all unmatched requests to the central brain.
        #
        # Technical questions -> Knowledge Engine / RAG
        # Programming requests -> Programming Agent
        # Mathematical questions -> Mathematical Agent
        # Complex reasoning -> Deductive Solver
        # Current information -> Research Pipeline
        #
        # The Conversational Cortex must not generate generic
        # technical answers or duplicate other specialized engines.

        return None
    def _quick_web_insight(self, topic: str, is_update_query: bool = False) -> Optional[str]:
        """Perform lightweight web retrieval with clean separation to eliminate concatenated text."""
        try:
            import httpx
            from bs4 import BeautifulSoup

            search_query = f"{topic} latest updates new features" if is_update_query else topic
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            with httpx.Client(timeout=5.0, follow_redirects=True, headers=headers) as client:
                resp = client.post("https://html.duckduckgo.com/html/", data={"q": search_query})
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    snippets = []
                    for r in soup.find_all(class_="result")[:3]:
                        title_el = r.find(class_="result__title")
                        t_text = title_el.get_text(separator=" ", strip=True) if title_el else ""
                        t_text = re.sub(r"\s+", " ", t_text).strip()
                        s = r.find(class_="result__snippet")
                        if s:
                            snip = s.get_text(separator=" ", strip=True)
                            snip = re.sub(r"\s+", " ", snip).strip()
                            if len(snip) > 25:
                                if t_text:
                                    snippets.append(f"• **{t_text}**:\n  {snip}")
                                else:
                                    snippets.append(f"• {snip}")
                    if snippets:
                        return "\n\n".join(snippets)
        except Exception as e:
            log.debug(f"Conversational quick insight notice: {e}")

        return None


conversational_cortex = ConversationalCortex()
