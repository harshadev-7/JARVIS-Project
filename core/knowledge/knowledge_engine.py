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

        # 1. Check for Context Simplification ("Explain the previous answer in simpler words")
        if any(w in q_lower for w in ["simpler words", "explain in simple", "simplify", "easier words", "like i'm five"]):
            if previous_context:
                # Find matching topic in previous context
                for key, data in self.KNOWLEDGE_BASE.items():
                    if data["title"].lower() in previous_context.lower() or key in previous_context.lower():
                        return {
                            "type": "simplified_explanation",
                            "topic": data["title"],
                            "answer": (
                                f"### **{data['title']} — Explained in Simple Words**\n\n"
                                f"{data['simplified']}\n\n"
                                f"**Quick Analogy:**\n"
                                f"• {data['examples'][0]}\n\n"
                                f"Does this clearer perspective help, sir?"
                            ),
                            "verified": True,
                        }
            # Fallback general simplification
            return {
                "type": "simplified_explanation",
                "topic": "General Concept",
                "answer": (
                    "### **Simplified Explanation**\n\n"
                    "Breaking it down to core essentials:\n\n"
                    "Think of technology as tools designed to handle repetitive mental work. "
                    "Instead of a person manually calculating numbers or routing messages, we create systems "
                    "that do it automatically using clear, step-by-step rules."
                ),
                "verified": True,
            }

        # 2. Check for Comparative Analysis (e.g. "difference between AI and machine learning")
        if "difference between" in q_lower or "compare" in q_lower or "vs" in q_lower:
            if ("ai" in q_lower or "artificial intelligence" in q_lower) and ("machine learning" in q_lower or "ml" in q_lower):
                return self._synthesize_ai_vs_ml_comparison()
            if "java" in q_lower and "python" in q_lower:
                return self._synthesize_java_vs_python_comparison()

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

        # 3. Direct Concept Lookup in Knowledge Base
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


knowledge_engine = KnowledgeEngine()
__all__ = ["KnowledgeEngine", "knowledge_engine"]
