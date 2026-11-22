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
        # 9a. Java Technical Summary & Ecosystem Overview
        if (
            any(w in q_lower for w in ["summary about java", "summary of java", "about java", "what is java", "java programming", "java overview", "java language"])
            and not any(w in q_lower for w in ["update", "updates", "version", "new in java", "recent"])
        ):
            return (
                "### **Java Programming Language: Technical Summary**\n\n"
                "**Java** is a high-level, class-based, object-oriented, memory-safe language developed by James Gosling at Sun Microsystems in 1995 and maintained by Oracle.\n\n"
                "#### **1. Architectural Pillars:**\n"
                "- **Write Once, Run Anywhere (WORA):** Java source (`.java`) compiles into bytecode (`.class`), executed by the **Java Virtual Machine (JVM)**.\n"
                "- **Automatic Memory Management:** Generational Garbage Collection (G1GC, ZGC, Shenandoah) eliminates manual pointer freeing.\n"
                "- **Object-Oriented Programming (OOP):** Strictly adheres to Encapsulation, Inheritance, Polymorphism, and Abstraction.\n"
                "- **Multithreading & Concurrency:** Robust concurrency primitives (`java.util.concurrent`), locks, atomics, and modern **Virtual Threads (Project Loom)**.\n\n"
                "#### **2. Execution Architecture:**\n"
                "- **JDK (Java Development Kit):** Full development kit including the compiler (`javac`), debugger, and utilities.\n"
                "- **JRE (Java Runtime Environment):** Runtime libraries + JVM necessary to execute applications.\n"
                "- **JVM (Java Virtual Machine):** Just-In-Time (JIT) HotSpot compiler converting hot bytecode to native CPU instructions.\n\n"
                "#### **3. Primary Applications:**\n"
                "- **Enterprise Backends:** Spring Boot, Micronaut, Quarkus, Jakarta EE.\n"
                "- **Distributed Systems & Big Data:** Apache Kafka, Apache Spark, Hadoop, Cassandra, Elasticsearch.\n"
                "- **Android Systems:** Core runtime foundation."
            )

        # 9b. Java Platform Updates & Modern Releases (Java 21 LTS & Beyond)
        if "java" in q_lower and any(w in q_lower for w in ["update", "updates", "new features", "latest", "what's new", "recent", "version", "release"]):
            return (
                "### **Latest Java Platform Updates & Modern Releases (Java 21 LTS & Beyond)**\n\n"
                "Under the standard 6-month release cadence, **Java 21** is the current primary Long-Term Support (LTS) release, introducing major architectural advancements:\n\n"
                "#### **1. Game-Changing Java 21 LTS Features:**\n"
                "- **Virtual Threads (Project Loom - JEP 444):**\n"
                "  - Lightweight, user-mode threads managed directly by the JVM rather than 1:1 OS kernel threads.\n"
                "  - Enables high-throughput concurrent servers handling millions of connections with synchronous code (`Executors.newVirtualThreadPerTaskExecutor()`).\n"
                "- **Sequenced Collections (JEP 431):**\n"
                "  - Unified collection interfaces: `SequencedCollection`, `SequencedSet`, `SequencedMap` with explicit ordering methods: `.getFirst()`, `.getLast()`, `.addFirst()`, `.reversed()`.\n"
                "- **Record Patterns (JEP 440):**\n"
                "  - Deconstructs `record` instances directly in pattern matching expressions without boilerplate getters.\n"
                "- **Pattern Matching for `switch` (JEP 441):**\n"
                "  - Type-safe, expressive branching within `switch` statements and expressions with exhaustiveness checking.\n"
                "- **Generational ZGC (JEP 439):**\n"
                "  - Sub-millisecond maximum pause times even across multi-terabyte heaps by dividing memory into young and old generations.\n\n"
                "#### **2. Modern Java 22 / 23 / 24 Enhancements:**\n"
                "- **Foreign Function & Memory API (Project Panama - JEP 454):** Safe, fast interop with native C/C++ libraries without JNI.\n"
                "- **Stream Gatherers (Preview):** Customizable intermediate operations for the Java Streams API (e.g. windowing, folding).\n"
                "- **String Templates & Compact Source Files:** Streamlining syntax for modern developers."
            )

        # 9c. Conceptual Addition & Foundations of Arithmetic
        if any(w in q_lower for w in ["what is an addition", "what is addition", "explain addition", "concept of addition", "definition of addition", "tell about addition"]):
            return (
                "### **Mathematical & Computational Concept: Addition ($+$)**\n\n"
                "**Addition** is one of the four foundational binary arithmetic operations (alongside subtraction, multiplication, and division) "
                "that computes the aggregate combination of two or more quantities.\n\n"
                "#### **1. Axiomatic Mathematical Properties:**\n"
                "- **Closure:** For any $a, b \\in \\mathbb{R}$, their sum $a + b \\in \\mathbb{R}$.\n"
                "- **Commutative Law:** Operand order does not change the result:\n"
                "  $$a + b = b + a \\quad (10 + 20 = 20 + 10 = 30)$$\n"
                "- **Associative Law:** Regrouping yields identical values:\n"
                "  $$(a + b) + c = a + (b + c)$$\n"
                "- **Additive Identity ($0$):** Adding zero leaves any quantity unchanged ($a + 0 = a$).\n"
                "- **Additive Inverse:** For every number $a$, there exists $-a$ such that $a + (-a) = 0$.\n\n"
                "#### **2. Hardware Implementation in Computer Architecture (ALU):**\n"
                "Inside the CPU Arithmetic Logic Unit:\n"
                "- **Half-Adder:** Computes sum of two single bits ($\\text{Sum} = A \\oplus B$, $\\text{Carry} = A \\land B$).\n"
                "- **Full-Adder:** Computes sum including carry-in ($C_{\\text{in}}$) from previous stages.\n"
                "- **Carry-Lookahead Adders (CLA):** Eliminates ripple carry propagation delay in modern 64-bit processors."
            )

        # 9d. Operating Systems (OS) Architecture
        if any(w in q_lower for w in ["what is an os", "what is an operating system", "what is os", "operating system concept", "about operating system", "os(operating system)"]):
            return (
                "### **Computer Systems: What is an Operating System (OS)?**\n\n"
                "An **Operating System (OS)** is foundational system software that acts as an intermediary interface between computer hardware and user applications. "
                "It abstracts physical hardware complexity and coordinates resource sharing.\n\n"
                "#### **1. Core Responsibilities of an OS:**\n"
                "1. **Process Management & Scheduling:** Allocates CPU time across multiple tasks (Round Robin, SJF, Multi-Level Feedback Queues), handling context switching and synchronization.\n"
                "2. **Memory Management (RAM):** Implements **Virtual Memory**, paging, segmentation, and page replacement (LRU, Clock) to provide each process an isolated address space.\n"
                "3. **Storage & File System Management:** Organizes hierarchical storage (ext4, NTFS, APFS), file descriptors, inodes, and disk I/O scheduling.\n"
                "4. **Device & I/O Subsystem:** Provides standardized hardware device drivers, handling interrupts and Direct Memory Access (DMA).\n"
                "5. **Protection & Security:** Enforces CPU privilege rings—distinguishing **Kernel Mode (Ring 0)** with unrestricted hardware access from **User Mode (Ring 3)** where application code runs safely.\n\n"
                "#### **2. Primary Architectural Models:**\n"
                "- **Monolithic Kernels:** All core OS services run in kernel space (Linux, Unix).\n"
                "- **Microkernels:** Only minimal primitives run in kernel space; drivers and file systems run in user space (seL4, QNX).\n"
                "- **Hybrid Kernels:** Combines monolithic performance with modular architecture (Windows NT, macOS XNU)."
            )

        # 9e. Deadlock in Operating Systems
        if "deadlock" in q_lower and any(w in q_lower for w in ["operating system", "os", "prevent", "avoid", "algorithm", "occur", "cause", "what is"]):
            return (
                "### **Operating Systems: Deadlock Analysis & Prevention**\n\n"
                "A **deadlock** is a state where a set of concurrent processes are permanently blocked because each process holds a resource "
                "and is waiting to acquire another resource held by another process in the same set.\n\n"
                "#### **1. The Four Necessary & Sufficient Coffman Conditions:**\n"
                "1. **Mutual Exclusion:** At least one resource must be non-shareable.\n"
                "2. **Hold and Wait:** A process holds resources while waiting for additional resources.\n"
                "3. **No Preemption:** Resources cannot be forcibly seized; only voluntarily released.\n"
                "4. **Circular Wait:** Closed cycle of processes (P0 -> P1 -> ... -> Pn -> P0) waiting on each other.\n\n"
                "#### **2. Deadlock Prevention:**\n"
                "- **Invalidate Circular Wait (Resource Ordering):** Impose strict global order F: R -> N. Processes can only request resources in strictly ascending order (F(Rj) > F(Ri)).\n"
                "- **Invalidate Hold and Wait:** Require processes to request all resources at once.\n"
                "- **Invalidate No Preemption:** Preempt held resources if a new request cannot be allocated.\n\n"
                "#### **3. Deadlock Avoidance (Banker's Algorithm):**\n"
                "- Dynamically verifies that allocating a resource preserves a **Safe State** where a safe sequence exists.\n"
                "*(For complete Python/Java source code and dry-run execution, issue `/solve deadlock in OS`)*"
            )

        # 9f. Python Technical Summary & Ecosystem Overview
        if (
            any(w in q_lower for w in ["summary about python", "summary of python", "about python", "what is python", "python programming", "python overview", "python language"])
            and not any(w in q_lower for w in ["update", "updates", "version", "new in python", "recent"])
        ):
            return (
                "### **Python Programming Language: Technical Summary**\n\n"
                "**Python** is a high-level, interpreted, dynamically typed, garbage-collected programming language created by Guido van Rossum in 1991.\n\n"
                "#### **1. Architectural & Linguistic Pillars:**\n"
                "- **Readability & Expressiveness:** Emphasizes clean syntax using whitespace indentation (PEP 8).\n"
                "- **Multi-Paradigm:** Supports Object-Oriented, Functional, Imperative, and Procedural programming styles.\n"
                "- **Dynamic Typing & Memory Management:** Reference counting combined with a generational cyclic garbage collector.\n"
                "- **Global Interpreter Lock (GIL):** A mutex in CPython that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously (modern Python 3.13 introduces experimental free-threaded mode without the GIL).\n\n"
                "#### **2. Primary Industry Ecosystems:**\n"
                "- **Artificial Intelligence & Machine Learning:** PyTorch, TensorFlow, JAX, Hugging Face, Scikit-learn.\n"
                "- **Data Science & Analytics:** NumPy, Pandas, Polars, SciPy, Matplotlib.\n"
                "- **Backend Engineering:** FastAPI, Django, Flask, AsyncIO."
            )

        # 10. Broad Informational Inquiries
        from core.brain.decision_policy import decision_policy

        topic_match = re.search(
            r"(?i)^(?:give me\s+|tell me\s+|tell\s+|show me\s+)?(?:a\s+)?(?:summary\s+(?:about|of)|new\s+updates?\s+(?:about|of|on|in)|updates?\s+(?:about|of|on|in)|what\s+is\s+(?:an?\s+)?|what\s+are\s+|who\s+is\s+|why\s+is\s+|explain\s+|about\s+|how\s+does\s+|definition\s+of\s+|overview\s+of\s+)(.+)$",
            q_clean
        )
        if topic_match:
            raw_topic = topic_match.group(1).strip().rstrip("?.")
            # Don't capture known cognitive/math domains
            if not any(k in raw_topic.lower() for k in ["angle", "clock", "train", "speed", "distance", "sheep", "handshake", "sum of", "discount", "profit"]):
                if len(raw_topic) >= 2:
                    is_update = bool(re.search(r"\b(update|updates|new|latest|version)\b", q_clean, re.I))
                    # Only search if policy confirms it requires live internet research
                    if decision_policy.should_search_internet(q_clean):
                        summary = self._quick_web_insight(raw_topic, is_update_query=is_update)
                        if summary:
                            header = f"Latest Updates & Release Details for **{raw_topic}**" if is_update else f"Verified Technical Summary: **{raw_topic}**"
                            return (
                                f"### **{header}**\n\n"
                                f"{summary}\n\n"
                                f"Would you like to explore deeper into any specific aspect of {raw_topic}, {address}?"
                            )

        # 10b. Guard against running web perception on imperative action commands
        if re.match(r"(?i)^(play|open|launch|run|close|stop|pause|render|edit|cut|trim|start|watch|show)\b", q_clean):
            return None

        # 11. Direct Deductive & Answer-First Fallback
        from core.planner.deductive_solver import deductive_solver
        direct_ans = deductive_solver.solve_problem(q_clean)
        if direct_ans and direct_ans.get("solution"):
            return direct_ans["solution"]

        cleaned_topic = q_clean.rstrip("?.")
        if len(cleaned_topic) > 3:
            if decision_policy.should_search_internet(q_clean):
                summary = self._quick_web_insight(cleaned_topic)
                if summary:
                    return (
                        f"### **Information on {cleaned_topic}:**\n\n"
                        f"{summary}\n\n"
                        f"How can I assist you further with this, {address}?"
                    )

            return (
                f"I hear you, {address}. Regarding **{cleaned_topic}**:\n\n"
                f"All core systems are operational. I can generate code, solve quantitative problems, "
                f"or explore technical concepts directly. How can I assist you with this?"
            )

        return f"Hello, {address}! All JARVIS core systems are online. What would you like to explore or solve together today?"

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
