"""J.A.R.V.I.S. Knowledge Answering Engine.

Provides deep, structured, first-principles explanations for:
- Core computer science concepts (AI, Machine Learning, Operating Systems, Databases, Algorithms, Networks)
- Programming languages (Java, Python, C++, etc.)
- Comparative concept analysis (e.g. AI vs Machine Learning)
- Multi-level explanations (Beginner, Intermediate, Advanced) with concrete real-world examples
- Context-aware simplification ("Explain the previous answer in simpler words")
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class KnowledgeEngine:
    """Offline-first knowledge synthesis engine for conceptual questions."""

    KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
        "jarvis": {
            "title": "J.A.R.V.I.S. (Just A Rather Very Intelligent System)",
            "aliases": [
                "jarvis", "what are you", "who are you", "what is jarvis", "who is jarvis",
                "tell me about yourself", "what can you do", "introduce yourself",
                "about you", "what are your capabilities", "your name", "about jarvis"
            ],
            "overview": (
                "**J.A.R.V.I.S.** (Just A Rather Very Intelligent System) is an autonomous, 100% offline-first "
                "multi-agent artificial intelligence assistant. Designed to run completely locally on your machine's CPU "
                "without external cloud AI APIs (zero dependencies on OpenAI, Gemini, or Ollama), JARVIS unites a custom "
                "PyTorch Transformer, deterministic reasoning, persistent memory, and safe multi-device automation."
            ),
            "how_it_works": (
                "JARVIS operates through an integrated, multi-tier cognitive architecture:\n"
                "1. **Neural Transformer Core:** A custom GPT-style Decoder-Only Transformer running locally.\n"
                "2. **Deductive Problem Solving:** 10-step first-principles reasoning for deterministic math, logic, and code.\n"
                "3. **Persistent Memory:** SQLite storage (`jarvis.db`) paired with TF-IDF semantic vector recall.\n"
                "4. **Autonomous Agent Swarm:** 15 domain-specialized agents (Coding, Research, Security, Automation, Vision, etc.).\n"
                "5. **Self-Evolution Engine:** Continuous self-learning, AST code analysis, testing gates, and SemVer release management."
            ),
            "examples": [
                "**10-Step Deductive Problem Solving:** Run `/solve <problem>` for step-by-step verified problem solving.",
                "**AST Code Analysis & Refactoring:** Static inspection, bug linting, and sandboxed patching via `/analyze-source`.",
                "**Media & Computer Vision:** Highlight extraction, vertical reel creation, and OCR via AutoClip.",
                "**Authorized Multi-Device Mesh:** Local encrypted sync and companion device pairing without cloud brokers."
            ],
            "simplified": (
                "In simple words, I am **JARVIS**—your private, personal artificial intelligence co-pilot running directly "
                "on your computer. I help you write code, solve problems, research facts, automate PC tasks, and edit videos "
                "with complete privacy and zero cloud dependencies."
            )
        },
        "ai": {
            "title": "Artificial Intelligence (AI)",
            "aliases": ["ai", "artificial intelligence", "what is an ai", "what is ai", "define ai"],
            "overview": (
                "**Artificial Intelligence (AI)** is a broad branch of computer science dedicated to building "
                "machines and software capable of performing tasks that typically require human intelligence. "
                "These cognitive tasks include visual perception, speech recognition, decision-making, reasoning, "
                "and natural language understanding."
            ),
            "how_it_works": (
                "Modern AI systems operate primarily through **data-driven computational models**, "
                "particularly deep neural networks inspired by biological neural architectures. Instead of relying "
                "solely on hardcoded if-else rules, AI systems optimize mathematical loss functions over large datasets "
                "to recognize patterns, generalize to unseen inputs, and execute goal-directed actions."
            ),
            "examples": [
                "**Autonomous Vehicles:** Self-driving cars using computer vision and sensor fusion to navigate traffic.",
                "**Natural Language Assistants:** Large Language Models (like JARVIS) understanding and generating coherent text.",
                "**Medical Diagnostics:** AI algorithms analyzing radiological scans to detect anomalies earlier than human baseline.",
                "**Game AI:** Deep reinforcement learning agents mastering chess, Go, and complex real-time strategy games."
            ],
            "simplified": (
                "In simple words, **AI** is like teaching a computer how to think, learn, and solve problems on its own, "
                "just like a human does. Instead of a programmer writing instructions for every single possibility, "
                "the AI learns from examples and experience."
            )
        },
        "machine_learning": {
            "title": "Machine Learning (ML)",
            "aliases": ["machine learning", "ml", "what is machine learning", "define machine learning"],
            "overview": (
                "**Machine Learning (ML)** is a core subfield of Artificial Intelligence focused on developing algorithms "
                "that enable computer systems to automatically learn and improve from experience without being explicitly "
                "programmed for every edge case."
            ),
            "how_it_works": (
                "ML algorithms ingest training data, identify underlying statistical distributions and correlations, "
                "and adjust internal parameters (weights and biases). The primary paradigms are:\n"
                "1. **Supervised Learning:** Learning input-to-output mappings from labeled examples (e.g. linear regression, classification).\n"
                "2. **Unsupervised Learning:** Discovering hidden structures or clusters in unlabeled data (e.g. K-Means, PCA).\n"
                "3. **Reinforcement Learning:** Agents learning optimal decision policies through reward-and-punishment feedback."
            ),
            "examples": [
                "**Email Spam Filtering:** An ML classifier trained on millions of emails that automatically identifies spam patterns.",
                "**Recommendation Systems:** Streaming platforms (Netflix, Spotify) suggesting content based on your past behavior.",
                "**Credit Card Fraud Detection:** Financial algorithms detecting abnormal spending patterns in milliseconds."
            ],
            "simplified": (
                "In simple words, **Machine Learning** is like showing a child thousands of pictures of cats and dogs until the child "
                "can recognize the difference on their own. The computer looks at data, finds patterns, and makes predictions."
            )
        },
        "aiml": {
            "title": "AIML (Artificial Intelligence & Machine Learning / AI Markup Language)",
            "aliases": [
                "aiml", "what is aiml", "define aiml", "ai ml", "ai/ml",
                "what is ai/ml", "artificial intelligence markup language",
                "ai and machine learning", "ai & ml", "explain aiml",
                "tell me about aiml", "about aiml",
            ],
            "overview": (
                "In computer science and software engineering, **AIML** refers to two primary concepts:\n\n"
                "1. **AI & Machine Learning (AI/ML):** The unified discipline combining **Artificial Intelligence** "
                "(systems exhibiting autonomous perception, reasoning, and decision-making) with **Machine Learning** "
                "(statistical algorithms that generalize patterns from empirical training data without being explicitly hardcoded).\n\n"
                "2. **Artificial Intelligence Markup Language (AIML):** An XML-compliant declarative markup language "
                "created in 1995 by Dr. Richard Wallace for programming rule-based conversational agents and chatbots, "
                "such as the historical A.L.I.C.E. chatterbot."
            ),
            "how_it_works": (
                "**1. Core Pillars of AI & Machine Learning (AI/ML):**\n"
                "• **Data Pipeline & Representation:** Ingesting structured or unstructured data, feature engineering, and high-dimensional vector embeddings.\n"
                "• **Mathematical Optimization:** Minimizing objective loss functions using gradient descent algorithms (e.g. AdamW, SGD) across deep neural architectures.\n"
                "• **Inference & Decision Policy:** The trained model predicts classifications, generates text/code, or drives autonomous actions.\n\n"
                "**2. AIML Chatbot Syntax (XML-based):**\n"
                "• `<category>`: The fundamental atomic knowledge unit.\n"
                "• `<pattern>`: The user prompt match condition (with wildcard support).\n"
                "• `<template>`: The generated response.\n"
                "• `<srai>`: Symbolic reduction used to resolve synonyms and conversational redirects."
            ),
            "examples": [
                "**AI/ML Real-World Applications:** Natural language processing, computer vision, recommendation systems, and autonomous robotics.",
                "**AIML Code Snippet:**\n"
                "  `<category>`\n"
                "    `<pattern>WHAT IS YOUR NAME</pattern>`\n"
                "    `<template>My name is JARVIS.</template>`\n"
                "  `</category>`",
                "**JARVIS Offline Architecture:** A living example of modern AI/ML—featuring a local CPU Transformer, 10-step deductive solver, and persistent memory."
            ],
            "simplified": (
                "In simple words, today **AIML** most often refers to **AI and Machine Learning**—the field of teaching computers "
                "to recognize patterns, make predictions, and solve problems from data. Historically, it is also the name of a simple "
                "XML-based language created in the 1990s to build rule-based chatbots."
            )
        },
        "java": {
            "title": "Java Programming Language",
            "aliases": ["java", "what is java", "define java", "java programming language"],
            "overview": (
                "**Java** is a high-level, class-based, object-oriented programming language designed to have as few "
                "implementation dependencies as possible. Developed by Sun Microsystems (now Oracle) in 1995, its defining philosophy "
                "is **\"Write Once, Run Anywhere\" (WORA)**."
            ),
            "how_it_works": (
                "Java source code (`.java`) is compiled by the Java compiler (`javac`) into portable intermediate **Bytecode** (`.class`). "
                "This bytecode is executed on the **Java Virtual Machine (JVM)**, which interprets or dynamically compiles (JIT compilation) "
                "the bytecode into native machine instructions for the host operating system. Java also features automatic memory management "
                "via an internal Garbage Collector (GC)."
            ),
            "examples": [
                "**Enterprise Backend Systems:** Banking platforms, transaction processors, and cloud microservices (Spring Boot).",
                "**Android App Development:** The original foundational language and runtime architecture for Android applications.",
                "**Big Data Processing:** Apache Hadoop, Apache Spark, and Apache Kafka are built primarily in Java and Scala."
            ],
            "simplified": (
                "In simple words, **Java** is a popular coding language built so that a program written on a Windows PC can run "
                "on a Mac, Linux server, or smartphone without rewriting the code, thanks to a translator called the JVM."
            )
        },
                "jvm": {
            "title": "Java Virtual Machine (JVM)",
            "aliases": [
                "jvm",
                "java virtual machine",
                "what is jvm",
                "what is the jvm",
            ],
            "overview": (
                "The **Java Virtual Machine (JVM)** is an abstract runtime "
                "machine that executes Java bytecode. It allows Java "
                "programs to run on different operating systems."
            ),
            "how_it_works": (
                "1. The Java compiler converts source code into bytecode.\n"
                "2. The JVM loads and verifies the bytecode.\n"
                "3. The execution engine interprets bytecode or uses "
                "Just-In-Time (JIT) compilation.\n"
                "4. The JVM manages memory and garbage collection."
            ),
            "examples": [
                "**Platform Independence:** The same Java bytecode can run "
                "on compatible JVM implementations across operating systems.",
                "**Memory Management:** JVM automatically manages heap "
                "memory and garbage collection.",
                "**Execution:** The JVM executes compiled Java applications."
            ],
            "simplified": (
                "The JVM is like a machine-specific translator. It takes "
                "Java bytecode and executes it on your computer."
            )
        },

        "jre": {
            "title": "Java Runtime Environment (JRE)",
            "aliases": [
                "jre",
                "java runtime environment",
                "what is jre",
                "what is the jre",
            ],
            "overview": (
                "The **Java Runtime Environment (JRE)** provides the "
                "runtime components required to execute Java applications. "
                "It includes the JVM and supporting libraries."
            ),
            "how_it_works": (
                "1. The JRE provides the JVM.\n"
                "2. It supplies Java class libraries and supporting files.\n"
                "3. The JVM executes Java bytecode using these resources.\n"
                "4. Applications can run without requiring development tools."
            ),
            "examples": [
                "**Running Applications:** Executing an existing Java desktop application.",
                "**Enterprise Applications:** Providing runtime components "
                "for Java-based server applications.",
                "**Libraries:** Supplying standard Java runtime libraries."
            ],
            "simplified": (
                "The JRE is the environment needed to run Java programs. "
                "It contains the JVM and the libraries required by applications."
            )
        },

        "jdk": {
            "title": "Java Development Kit (JDK)",
            "aliases": [
                "jdk",
                "java development kit",
                "what is jdk",
                "what is the jdk",
            ],
            "overview": (
                "The **Java Development Kit (JDK)** is a software "
                "development package used to create, compile, debug, "
                "and run Java applications."
            ),
            "how_it_works": (
                "The JDK provides development tools such as:\n"
                "1. **javac:** Compiles Java source code into bytecode.\n"
                "2. **java:** Launches Java applications.\n"
                "3. **javadoc:** Generates API documentation.\n"
                "4. **jdb:** Provides debugging capabilities.\n"
                "5. Runtime components required to execute Java programs."
            ),
            "examples": [
                "**Development:** Writing Java programs in Eclipse or IntelliJ IDEA.",
                "**Compilation:** Using javac to convert .java files into .class files.",
                "**Debugging:** Finding and fixing errors in Java applications."
            ],
            "simplified": (
                "The JDK is the complete toolkit developers use to create, "
                "compile, debug, and run Java programs."
            )
        },
        "python": {
            "title": "Python Programming Language",
            "aliases": ["python", "what is python", "define python"],
            "overview": (
                "**Python** is an interpreted, high-level, general-purpose programming language known for its emphasis on "
                "code readability and clean syntax. Created by Guido van Rossum and released in 1991, Python supports multiple "
                "programming paradigms including procedural, object-oriented, and functional programming."
            ),
            "how_it_works": (
                "Python code is dynamically typed and garbage-collected. The standard CPython implementation parses source code, "
                "compiles it into an internal bytecode representation, and executes it sequentially using the Python Virtual Machine (PVM)."
            ),
            "examples": [
                "**AI & Data Science:** PyTorch, TensorFlow, NumPy, Pandas, Scikit-Learn.",
                "**Web Development:** Django and FastAPI powering scalable web APIs.",
                "**Automation & Scripting:** System administration, web scraping, and workflow orchestration."
            ],
            "simplified": (
                "In simple words, **Python** is a coding language designed to read almost like plain English, making it one of the easiest "
                "and most powerful languages for beginners, data scientists, and AI engineers."
            )
        },
        "operating_system": {
            "title": "Operating System (OS)",
            "aliases": ["operating system", "what is an operating system", "os", "what is os"],
            "overview": (
                "An **Operating System (OS)** is the fundamental system software that manages a computer's hardware resources "
                "(CPU, memory, disk storage, and peripheral devices) and provides common services for application programs."
            ),
            "how_it_works": (
                "The core of an OS is the **Kernel**, which operates in a privileged CPU state (Kernel Mode). Key responsibilities include:\n"
                "1. **Process Management:** Scheduling threads on CPU cores and managing multitasking.\n"
                "2. **Memory Management:** Allocating virtual memory spaces and handling paging to prevent process conflicts.\n"
                "3. **File System Management:** Organizing data into hierarchical directories and handling persistent storage I/O.\n"
                "4. **Device Drivers:** Translating standard OS requests into hardware-specific peripheral signals."
            ),
            "examples": [
                "**Desktop/Workstation:** Microsoft Windows, macOS, Linux (Ubuntu, Debian, Fedora).",
                "**Mobile:** Android (Linux kernel-based), iOS (Unix/Darwin-based).",
                "**Server/Enterprise:** Red Hat Enterprise Linux, Ubuntu Server."
            ],
            "simplified": (
                "In simple words, an **Operating System** is the manager of the computer. It makes sure your screen, mouse, memory, "
                "and processor all work together so you can run games, browsers, and programs smoothly."
            )
        },
        "operating_system_process": {
            "title": "Operating System Processes",
            "aliases": [
                "operating system processes",
                "operating system process",
                "os process",
                "os processes",
                "process in os",
                "process in operating system",
                "explain operating system processes",
                "what is an operating system process",
                "process management",
            ],
            "overview": (
                "An **Operating System Process** is an active, executing instance of a computer program. "
                "While a program is passive code stored statically on disk, a process is dynamic—possessing an active "
                "Program Counter (PC), dedicated address space, CPU registers, stack, heap, and assigned OS resources."
            ),
            "how_it_works": (
                "1. **Process Control Block (PCB):** The OS kernel maintains a PCB for each process containing its PID (Process ID), "
                "execution state, CPU scheduling priority, registers, memory pointers, and I/O status.\n"
                "2. **Process State Life Cycle:** A process transitions through standard states: "
                "`New → Ready → Running → Waiting/Blocked → Terminated`.\n"
                "3. **Context Switching:** When multitasking, the OS scheduler saves the CPU state of the running process into its PCB "
                "and loads the state of the next ready process in microseconds.\n"
                "4. **Memory Layout:** Each process has an isolated virtual address space divided into: "
                "Text Section (machine instructions), Data Section (global/static variables), Heap (dynamically allocated memory), and Stack (local variables and call frames)."
            ),
            "examples": [
                "**Web Browser Tabs:** In multi-process architectures like Google Chrome, each browser tab and extension runs in its own isolated OS process for stability and security.",
                "**Background Services & Daemons:** Autonomous background services (e.g. JARVIS Sentinel Daemon or print spoolers) running without a visible GUI window.",
                "**Inter-Process Communication (IPC):** Separate processes cooperating and exchanging data through OS pipes, shared memory, or sockets."
            ],
            "simplified": (
                "In simple words, a **Process** is a computer program in action. If a program is a printed recipe in a book, "
                "a process is a chef actively in the kitchen following that recipe with ingredients, pots, and stove heat."
            )
        },
        "database": {
            "title": "Database (DBMS)",
            "aliases": ["database", "what is a database", "what is database", "dbms"],
            "overview": (
                "A **Database** is an organized collection of structured or semi-structured data stored electronically in a computer system. "
                "It is controlled and accessed through a **Database Management System (DBMS)**."
            ),
            "how_it_works": (
                "Databases provide ACID guarantees (Atomicity, Consistency, Isolation, Durability) to ensure data integrity during concurrent transactions.\n"
                "• **Relational Databases (SQL):** Store data in structured tables with rows, columns, and foreign key relations (PostgreSQL, MySQL, SQLite).\n"
                "• **NoSQL Databases:** Store unstructured or flexible documents, key-value pairs, or graphs (MongoDB, Redis, Cassandra)."
            ),
            "examples": [
                "**Banking Systems:** Storing account balances and recording immutable financial transactions.",
                "**E-Commerce:** Managing user shopping carts, product catalogs, and inventory levels.",
                "**Local Storage:** SQLite (`jarvis.db`) powering persistent storage directly on local disk."
            ],
            "simplified": (
                "In simple words, a **Database** is a super-fast, highly organized digital filing cabinet that lets computer programs "
                "safely save, search, and update millions of records in a split second."
            )
        },
        "algorithm": {
            "title": "Algorithm",
            "aliases": ["algorithm", "what is an algorithm", "what is algorithm"],
            "overview": (
                "An **Algorithm** is an unambiguous, step-by-step mathematical or logical procedure designed to solve a specific class of problems "
                "or perform a computation. It transforms specified inputs into desired outputs within a finite number of steps."
            ),
            "how_it_works": (
                "Algorithms are evaluated on computational efficiency using **Asymptotic Complexity (Big-O Notation)**:\n"
                "• **Time Complexity:** How the execution time scales as input size $n$ increases (e.g. $O(1)$, $O(n)$, $O(n \\log n)$, $O(n^2)$).\n"
                "• **Space Complexity:** How much auxiliary memory is required during execution."
            ),
            "examples": [
                "**Search Algorithms:** Binary Search finding a word in a sorted dictionary in $O(\\log n)$ time.",
                "**Sorting Algorithms:** QuickSort and MergeSort ordering lists of millions of items.",
                "**Graph Navigation:** Dijkstra's algorithm calculating the fastest driving route on Google Maps."
            ],
            "simplified": (
                "In simple words, an **Algorithm** is like a precise recipe. If you follow the exact steps in order, you are guaranteed "
                "to get the correct result every single time."
            )
        },
        "computer_network": {
            "title": "Computer Network",
            "aliases": ["computer network", "what is a computer network", "networking"],
            "overview": (
                "A **Computer Network** is a set of interconnected computing devices (nodes) that communicate with each other "
                "to share resources, data, and services using standardized communication protocols."
            ),
            "how_it_works": (
                "Networks operate based on layered protocol models, primarily the **TCP/IP Model** and **OSI 7-Layer Model**:\n"
                "1. **Physical & Data Link:** Ethernet, Wi-Fi, MAC addresses.\n"
                "2. **Network Layer:** IP addressing, routing packets across routers.\n"
                "3. **Transport Layer:** TCP (reliable, connection-oriented) and UDP (fast, connectionless).\n"
                "4. **Application Layer:** HTTP/HTTPS, DNS, SSH, WebSockets."
            ),
            "examples": [
                "**The Internet:** The global network of interconnected private and public networks.",
                "**Local Area Network (LAN):** Devices connected to a home Wi-Fi router sharing local files and printers.",
                "**Peer-to-Peer Mesh:** JARVIS authorized device integration connecting mobile phones to desktop workstations."
            ],
            "simplified": (
                "In simple words, a **Computer Network** is like a digital highway system connecting multiple computers together "
                "so they can talk, share files, and browse websites."
            )
        },
    }

    def answer_question(self, question: str, previous_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Identify question intent and synthesize an authoritative direct answer."""
        q_lower = question.lower().strip().rstrip("?.")

                # 1. Context-aware simplification
        if any(
            phrase in q_lower
            for phrase in [
                "simpler words",
                "explain in simple",
                "simplify",
                "easier words",
                "like i'm five",
                "explain simply",
                "in simple language",
            ]
        ):
            if previous_context:
                matched_topic = None

                # Prefer matching the complete topic title.
                for data in self.KNOWLEDGE_BASE.values():
                    if data["title"].lower() in previous_context.lower():
                        matched_topic = data
                        break

                # If the title is unavailable, check longer aliases.
                if matched_topic is None:
                    for data in self.KNOWLEDGE_BASE.values():
                        aliases = sorted(
                            data["aliases"],
                            key=len,
                            reverse=True,
                        )

                        for alias in aliases:
                            if len(alias) < 4:
                                continue

                            if re.search(
                                rf"\b{re.escape(alias.lower())}\b",
                                previous_context.lower(),
                            ):
                                matched_topic = data
                                break

                        if matched_topic is not None:
                            break

                if matched_topic:
                    return {
                        "type": "simplified_explanation",
                        "topic": matched_topic["title"],
                        "answer": (
                            f"### {matched_topic['title']} — Simple Explanation\n\n"
                            f"{matched_topic['simplified']}\n\n"
                            f"**Real-world example:**\n"
                            f"{matched_topic['examples'][0]}"
                        ),
                        "verified": False,
                        "source_type": "built_in_knowledge",
                    }

            return {
                "type": "simplified_explanation",
                "topic": "General Concept",
                "answer": (
                    "I can simplify the previous explanation, "
                    "but I need its original topic or answer to "
                    "avoid explaining the wrong concept."
                ),
                "verified": False,
                "source_type": "general_fallback",
            }

        # 2. Check for JDK, JRE, and JVM Multi-Concept Architecture Comparison
        java_components = ["jdk", "jre", "jvm"]
        if all(component in q_lower for component in java_components):
            return self._synthesize_jdk_jre_jvm_comparison()

        # 2b. Check for Comparative Analysis across products, hardware & technologies
        from core.planner.comparative_engine import comparative_engine
        if comparative_engine.is_comparative_query(question):
            if ("ai" in q_lower or "artificial intelligence" in q_lower) and ("machine learning" in q_lower or "ml" in q_lower):
                return self._synthesize_ai_vs_ml_comparison()
            if "java" in q_lower and "python" in q_lower:
                return self._synthesize_java_vs_python_comparison()

            # Universal Comparative Engine (iPhone vs Samsung, Mac vs Windows, OLED vs QLED, etc.)
            target_a, target_b = comparative_engine.extract_targets(question)
            if target_a and target_b:
                comp_result = comparative_engine.compare(target_a, target_b)
                if comp_result:
                    return comp_result

        # 2b. Specific Specialized Concepts (Inheritance, RAM)
        if "inheritance" in q_lower:
            from core.planner.deductive_solver import deductive_solver
            sol = deductive_solver._solve_code_request("explain java inheritance")
            if sol:
                return {
                    "type": "concept_explanation",
                    "topic": "Inheritance",
                    "answer": sol["solution"],
                    "verified": True,
                }

        if "ram" in q_lower and not any(w in q_lower for w in ["program", "diagram"]):
            from core.planner.deductive_solver import deductive_solver
            sol = deductive_solver._solve_factual_knowledge("what is ram")
            if sol:
                return {
                    "type": "concept_explanation",
                    "topic": "RAM",
                    "answer": sol["solution"],
                    "verified": True,
                }

        # 3. Direct Concept Lookup in Knowledge Base (skip if code generation is requested)
        from core.planner.code_generator import code_generator
        if code_generator.is_code_request(question):
            return None

        for key, data in self.KNOWLEDGE_BASE.items():
            for alias in data["aliases"]:
                # Exact or substring match on alias
                if re.search(rf"\b{re.escape(alias)}\b", q_lower) or q_lower == alias:
                    return self._format_knowledge_response(data)

        return None

    def _format_knowledge_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a rich, structured direct knowledge explanation."""
        examples_text = "\n".join([f"• {ex}" for ex in data["examples"]])
        formatted = (
            f"### **{data['title']}**\n\n"
            f"#### **1. Core Definition:**\n"
            f"{data['overview']}\n\n"
            f"#### **2. How It Works & Architecture:**\n"
            f"{data['how_it_works']}\n\n"
            f"#### **3. Real-World Applications & Examples:**\n"
            f"{examples_text}\n\n"
            f"#### **4. Simple Takeaway:**\n"
            f"*{data['simplified']}*"
        )
        return {
            "type": "knowledge_explanation",
            "topic": data["title"],
            "answer": formatted,
            "verified": True,
        }

    def _synthesize_ai_vs_ml_comparison(self) -> Dict[str, Any]:
        """Synthesizes a first-principles comparison between AI and Machine Learning."""
        formatted = (
            "### **Comparison: Artificial Intelligence (AI) vs. Machine Learning (ML)**\n\n"
            "The relationship between AI and Machine Learning is hierarchical: **Machine Learning is a specialized subset of Artificial Intelligence**.\n\n"
            "```text\n"
            "┌───────────────────────────────────────────────────────────┐\n"
            "│ ARTIFICIAL INTELLIGENCE (The Overarching Vision)          │\n"
            "│ Any technique enabling machines to mimic human cognition. │\n"
            "│ ┌───────────────────────────────────────────────────────┐ │\n"
            "│ │ MACHINE LEARNING (The Data-Driven Subset)             │ │\n"
            "│ │ Systems that learn and improve automatically from data│ │\n"
            "│ │ ┌───────────────────────────────────────────────────┐ │ │\n"
            "│ │ │ DEEP LEARNING (Multi-layer Neural Networks)       │ │ │\n"
            "│ │ └───────────────────────────────────────────────────┘ │ │\n"
            "│ └───────────────────────────────────────────────────────┘ │\n"
            "└───────────────────────────────────────────────────────────┘\n"
            "```\n\n"
            "#### **Key Structural Differences:**\n\n"
            "| Feature | Artificial Intelligence (AI) | Machine Learning (ML) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Definition** | The broad science of creating smart machines capable of human-like tasks. | A specific method of realizing AI by training models on data. |\n"
            "| **Scope** | Wide: includes expert systems, symbolic logic, robotics, and ML. | Focused: statistical pattern recognition, neural networks, optimization. |\n"
            "| **Operation** | Can use rule-based heuristics, search trees, or learned models. | Operates purely by optimizing algorithms on input training datasets. |\n"
            "| **Goal** | Simulate human decision-making and cognitive problem solving. | Learn from data to maximize prediction accuracy on unseen examples. |\n\n"
            "#### **Real-World Analogy:**\n"
            "• **AI** is like the concept of **Transportation** (includes planes, trains, bicycles, and cars).\n"
            "• **Machine Learning** is like an **Electric Car** (a specific, highly effective modern vehicle within transportation)."
        )
        return {
            "type": "comparative_analysis",
            "topic": "AI vs Machine Learning",
            "answer": formatted,
            "verified": True,
        }

    def _synthesize_java_vs_python_comparison(self) -> Dict[str, Any]:
        """Synthesizes a comparison between Java and Python."""
        formatted = (
            "### **Comparison: Java vs. Python**\n\n"
            "Both Java and Python are industry-standard programming languages, but they prioritize different architectural tradeoffs:\n\n"
            "| Dimension | Java | Python |\n"
            "| :--- | :--- | :--- |\n"
            "| **Typing System** | Statically Typed (types checked at compile time) | Dynamically Typed (types checked at runtime) |\n"
            "| **Execution** | Compiled to Bytecode, executed on JVM | Interpreted via bytecode on CPython / PVM |\n"
            "| **Syntax Style** | Verbose, structured, explicit class hierarchies | Concise, readable, whitespace-delimited |\n"
            "| **Performance** | High raw execution speed via JIT compilation | Moderate execution speed (relies on C extensions for speed) |\n"
            "| **Primary Domains** | Enterprise backend, Android, High-throughput banking | AI/Machine Learning, Data Science, Rapid prototyping, Scripting |\n\n"
            "**Summary Recommendation:** Use **Java** for massive, high-concurrency enterprise architectures. Use **Python** for machine learning, data engineering, and agile experimentation."
        )
        return {
            "type": "comparative_analysis",
            "topic": "Java vs Python",
            "answer": formatted,
            "verified": True,
        }

    def _synthesize_jdk_jre_jvm_comparison(self) -> Dict[str, Any]:
        """Synthesizes a first-principles architectural comparison between JDK, JRE, and JVM."""
        formatted = (
            "### **Architectural Comparison: JDK vs. JRE vs. JVM**\n\n"
            "The relationship between the Java Development Kit (JDK), Java Runtime Environment (JRE), "
            "and Java Virtual Machine (JVM) is hierarchical and nested:\n\n"
            "```text\n"
            "┌────────────────────────────────────────────────────────────────────────┐\n"
            "│ JDK (Java Development Kit) - For Developers                            │\n"
            "│ Development Tools: javac (compiler), jdb (debugger), javadoc, jar, etc. │\n"
            "│ ┌────────────────────────────────────────────────────────────────────┐ │\n"
            "│ │ JRE (Java Runtime Environment) - For Running Applications          │ │\n"
            "│ │ Class Libraries (rt.jar, core packages) & Configuration Files      │ │\n"
            "│ │ ┌────────────────────────────────────────────────────────────────┐ │ │\n"
            "│ │ │ JVM (Java Virtual Machine) - The Execution Engine              │ │ │\n"
            "│ │ │ ClassLoader, Bytecode Verifier, Execution Engine (JIT + Interp)│ │ │\n"
            "│ │ │ Memory Areas (Method Area, Heap, Stack, PC Registers)          │ │ │\n"
            "│ │ └────────────────────────────────────────────────────────────────┘ │ │\n"
            "│ └────────────────────────────────────────────────────────────────────┘ │\n"
            "└────────────────────────────────────────────────────────────────────────┘\n"
            "```\n\n"
            "#### **Key Structural Differences:**\n\n"
            "| Component | Full Name | Primary Purpose | Contains |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **JVM** | Java Virtual Machine | Executes compiled bytecode on host CPU | ClassLoader, Memory Manager (Garbage Collector), JIT Compiler |\n"
            "| **JRE** | Java Runtime Environment | Provides the runtime environment to execute Java apps | JVM + Core Class Libraries + Supporting Files |\n"
            "| **JDK** | Java Development Kit | Complete development package to write, compile, and debug | JRE + Development Tools (`javac`, `jdb`, `javadoc`, `jar`) |\n\n"
            "#### **Architectural Formula:**\n"
            "• **JRE** = JVM + Class Libraries (`java.lang`, `java.util`, etc.)\n"
            "• **JDK** = JRE + Development & Debugging Tools (`javac`, `jdb`, `jar`)\n\n"
            "#### **Platform Independence:**\n"
            "• The **Java language and bytecode** are platform-independent (\"Write Once, Run Anywhere\").\n"
            "• The **JVM and JDK/JRE implementations** are platform-dependent (different native binaries for Windows, Linux, and macOS)."
        )
        return {
            "type": "comparative_analysis",
            "topic": "JDK vs JRE vs JVM",
            "answer": formatted,
            "verified": True,
        }


knowledge_engine = KnowledgeEngine()
__all__ = ["KnowledgeEngine", "knowledge_engine"]
