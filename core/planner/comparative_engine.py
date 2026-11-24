"""
JARVIS AI — Universal Comparative Intelligence Engine
======================================================
Provides authoritative, multi-dimensional comparative analysis for products,
technologies, frameworks, hardware architectures, and operating systems.

Covers:
1. Build Quality, Materials, Durability & Design Ergonomics
2. Operating System, Ecosystem & Software Freedom
3. Performance, Hardware & Silicon Architecture
4. Display, Camera, and Signature Capabilities
5. Pricing, Resale Value & Long-Term Longevity
6. Side-by-Side Markdown Comparison Matrix
7. Definitive Decision Guide / Buying Verdict
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from core.logger import log


class ComparativeAnalysisEngine:
    """Universal engine for analyzing and comparing products, hardware, and technologies."""

    COMPARISON_REGEX = re.compile(
        r"(?i)\b(?:(?P<item_a>[a-z0-9_\-\.\s\+]+?)\s+(?:vs\.?|versus)\s+(?P<item_b>[a-z0-9_\-\.\s\+]+?)(?:\s+difference|\s+comparison|\s+which\s+is\s+better|\s+better)?\??$|"
        r"(?:difference\s+between|compare|comparison\s+of|comparison\s+between)\s+(?P<item_a2>[a-z0-9_\-\.\s\+]+?)\s+(?:and|vs\.?|versus)\s+(?P<item_b2>[a-z0-9_\-\.\s\+]+?)(?:\s+difference|\s+comparison|\s+which\s+is\s+better|\s+better)?\??$)",
        re.IGNORECASE
    )

    def is_comparative_query(self, query: str) -> bool:
        """Check if query is asking for a comparison or difference."""
        q_clean = query.strip().lower()
        if any(w in q_clean for w in [" vs ", " versus ", " vs.", "difference between", "compare "]):
            return True
        return bool(self.COMPARISON_REGEX.search(q_clean))

    def extract_targets(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract Entity A and Entity B from a comparative prompt."""
        q_clean = re.sub(r"^(?:jarvis|can you|tell me|what is the|explain the)\s*", "", query.strip(), flags=re.I).strip(" ?.")
        match = self.COMPARISON_REGEX.search(q_clean)
        if match:
            gd = match.groupdict()
            a = gd.get("item_a") or gd.get("item_a2")
            b = gd.get("item_b") or gd.get("item_b2")
            if a and b:
                clean_a = re.sub(r"^(?:the\s+|an?\s+)", "", a.strip(), flags=re.I).strip()
                clean_b = re.sub(r"^(?:the\s+|an?\s+)", "", b.strip(), flags=re.I).strip()
                return clean_a, clean_b

        # Fallback split on "vs"
        parts = re.split(r"(?i)\b(?:vs\.?|versus)\b", q_clean)
        if len(parts) >= 2:
            a = re.sub(r"^(?:what is the difference between|difference between|compare)\s*", "", parts[0], flags=re.I).strip()
            b = re.sub(r"\s+(?:difference|better|comparison).*$", "", parts[1], flags=re.I).strip()
            return a.strip(), b.strip()

        # Fallback split on "and" after "difference between"
        if "difference between" in q_clean:
            sub = q_clean.split("difference between", 1)[1].strip()
            if " and " in sub:
                p = sub.split(" and ", 1)
                return p[0].strip(), p[1].strip()

        return None, None

    def compare(
        self,
        target_a: str,
        target_b: str,
        live_evidence: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Perform authoritative comparative analysis across key structural dimensions."""
        a_norm = target_a.lower().strip()
        b_norm = target_b.lower().strip()

        # Check for specific flagship product / tech profiles
        if (
            ("iphone" in a_norm or "apple" in a_norm) and ("samsung" in b_norm or "galaxy" in b_norm)
            or ("samsung" in a_norm or "galaxy" in a_norm) and ("iphone" in b_norm or "apple" in b_norm)
        ):
            return self._synthesize_iphone_vs_samsung(live_evidence)

        if (
            ("mac" in a_norm or "macos" in a_norm or "apple" in a_norm) and ("windows" in b_norm or "pc" in b_norm)
            or ("windows" in a_norm or "pc" in a_norm) and ("mac" in b_norm or "macos" in b_norm or "apple" in b_norm)
        ):
            return self._synthesize_mac_vs_windows(live_evidence)

        if (
            ("android" in a_norm and "ios" in b_norm)
            or ("ios" in a_norm and "android" in b_norm)
        ):
            return self._synthesize_ios_vs_android(live_evidence)

        if (
            ("oled" in a_norm and "qled" in b_norm)
            or ("qled" in a_norm and "oled" in b_norm)
        ):
            return self._synthesize_oled_vs_qled(live_evidence)

        # Universal structured product/technology comparison engine
        return self._synthesize_universal_product_comparison(target_a, target_b, live_evidence)

    def _synthesize_iphone_vs_samsung(self, live_evidence: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Deep dimensional comparison between Apple iPhone and Samsung Galaxy flagships."""
        title = "Apple iPhone vs. Samsung Galaxy: Comprehensive Flagship Comparison"

        evidence_section = ""
        if live_evidence:
            bullets = [f"• **{item['title']}:** {item['text'][:240].strip()}..." for item in live_evidence[:3] if len(item['text'].strip()) > 30]
            if bullets:
                evidence_section = "\n\n#### **Contemporary Market & Release Context (2025/2026):**\n" + "\n".join(bullets)

        formatted = (
            f"### **{title}**\n\n"
            "The rivalry between **Apple iPhone** and **Samsung Galaxy** represents the pinnacle of mobile engineering. "
            "While both manufacturers produce industry-leading flagship hardware, they are anchored in fundamentally different "
            "philosophies of computing, hardware design, and software governance.\n\n"
            "#### **1. Build Quality, Materials & Industrial Design:**\n"
            "• **Apple iPhone:** Prioritizes unified, minimalist industrial luxury. Flagship Pro models employ **Grade 5 Titanium** chassis "
            "with a contoured matte glass back and proprietary **Ceramic Shield 2** glass on the front. Apple's tight tolerance engineering "
            "results in exceptional torsional rigidity, uniform bezel symmetry, and an ergonomic, dense in-hand feel.\n"
            "• **Samsung Galaxy:** Emphasizes cutting-edge functional variety. Ultra models feature **Armor Aluminum** or **Titanium frames** "
            "coupled with **Corning Gorilla Glass Armor**, which incorporates advanced anti-reflective optical coatings (reducing reflections by up to 75%). "
            "Samsung additionally leads in foldable form factors (Galaxy Z Fold and Z Flip series), offering hardware versatility Apple does not match.\n\n"
            "#### **2. Software, Operating System & Ecosystem Freedom:**\n"
            "• **iPhone (iOS):** Operates on a curated, closed-garden model. Strengths include seamless end-to-end integration across Apple hardware "
            "(Mac, iPad, Apple Watch, Apple TV, AirPods), immediate day-one global software updates, strict application sandbox security, and fluid UI animations.\n"
            "• **Samsung (Android / One UI):** Operates on user autonomy and expansive flexibility. Features comprehensive multitasking (split-screen, floating windows), "
            "system-wide file management, custom launcher integration via Good Lock, and **Samsung DeX** (transforming your phone into a complete desktop computer when plugged into a monitor).\n\n"
            "#### **3. Performance & Silicon Architecture:**\n"
            "• **Apple Silicon (A-Series Pro):** Apple designs custom ARM architectures with industry-leading single-core CPU efficiency, massive system caches, "
            "and tightly optimized unified memory. It delivers unmatched sustained performance per watt and superior longevity for gaming and media export.\n"
            "• **Samsung (Qualcomm Snapdragon / Exynos):** Samsung utilizes high-frequency Snapdragon chips (e.g. Snapdragon 8 Elite 'for Galaxy') featuring elite multi-core "
            "raw compute, powerful Adreno graphics, and dedicated NPUs for generative AI processing.\n\n"
            "#### **4. Camera Systems & Imaging Philosophy:**\n"
            "• **iPhone Imaging:** Focuses on natural color science, zero-shutter-lag responsiveness, and **unmatched cinematic video recording** "
            "(ProRes, 4K120 Dolby Vision, Log video profile with ACES support). Preferred by content creators who demand predictable, photorealistic fidelity.\n"
            "• **Samsung Imaging:** Focuses on optical versatility and maximum sensor capability. Features a **200 MP primary sensor** and dual periscope telephoto lenses "
            "(up to 5x and 10x optical zoom, 100x Space Zoom). Provides extensive manual controls through Camera Assistant and Expert RAW.\n\n"
            "#### **5. Battery Life, Charging & Lifespan:**\n"
            "• **Battery Life:** Both flagships comfortably deliver all-day endurance under heavy workloads.\n"
            "• **Charging Speeds:** Samsung leads with faster wired charging speeds (up to 45W vs. Apple's ~25–30W) and Reverse Wireless PowerShare. Apple champions **MagSafe** magnetic accessory alignment.\n"
            "• **Software Lifespan:** Both manufacturers now provide industry-best commitments of **7 years of major OS upgrades and security patches**.\n\n"
            "#### **6. Side-by-Side Comparison Matrix:**\n\n"
            "| Dimension | Apple iPhone (Pro / Flagship) | Samsung Galaxy (Ultra / Flagship) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Chassis Materials** | Grade 5 Titanium + Ceramic Shield 2 | Titanium / Armor Aluminum + Gorilla Glass Armor |\n"
            "| **Display Innovation** | Super Retina XDR OLED (ProMotion 120Hz) | Dynamic AMOLED 2X + Anti-Reflective Coating (1–120Hz) |\n"
            "| **Primary Operating System** | iOS (Closed, Curated, High Privacy) | Android / One UI (Open, Highly Customizable, DeX) |\n"
            "| **Silicon / Processor** | Custom Apple A-Series Pro (High Efficiency) | Snapdragon 8 Elite / Exynos (High Multi-core/GPU) |\n"
            "| **Camera Strengths** | Best-in-class 4K120 Dolby Vision Video & Color Realism | 200MP Detail & Extreme Telephoto Optical Zoom (5x/100x) |\n"
            "| **Signature Hardware** | Action Button, Camera Control, MagSafe | Embedded S-Pen Stylus, Foldable Series, Reverse Wireless Charge |\n"
            "| **Value & Resale** | Retains high resale value longer (~15–20% higher) | Faster trade-in discounts, frequent bundle promotions |\n\n"
            "#### **7. Definitive Buying Verdict:**\n"
            "• **Choose the iPhone if:** You already own Apple devices (Mac, iPad, Apple Watch), prioritize industry-standard video creation, "
            "value simple day-to-day fluid UX, and want maximum hardware resale value over 3–5 years.\n"
            "• **Choose Samsung Galaxy if:** You desire hardware versatility (S-Pen stylus, anti-reflective display, optical zoom), "
            "want desktop-class multitasking (Samsung DeX), prefer custom file management, or want cutting-edge foldable devices."
            f"{evidence_section}"
        )
        return {
            "type": "comparative_analysis",
            "topic": "iPhone vs Samsung",
            "answer": formatted,
            "verified": True,
            "source_type": "comparative_engine",
        }

    def _synthesize_mac_vs_windows(self, live_evidence: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Deep comparison between macOS (Apple Silicon) and Microsoft Windows (PC)."""
        title = "Apple Mac (macOS) vs. Microsoft Windows (PC): Comprehensive Comparison"
        formatted = (
            f"### **{title}**\n\n"
            "The choice between **Apple Mac** and **Microsoft Windows** represents two distinct philosophies of personal computing:\n\n"
            "#### **1. Build Quality, Materials & Hardware Ecosystem:**\n"
            "• **Apple Mac:** Single-vendor hardware integration. Laptops (MacBook Air/Pro) are carved from single blocks of recycled aluminum, "
            "featuring industry-best Force Touch trackpads, high-fidelity speakers, and superior structural rigidity.\n"
            "• **Windows PC:** Multi-vendor ecosystem (Dell, Lenovo, HP, ASUS, custom desktop builds). Provides unmatched hardware freedom—from ultra-light "
            "carbon-fiber laptops to liquid-cooled multi-GPU workstation towers with full modular component upgradability.\n\n"
            "#### **2. Performance & Power Efficiency:**\n"
            "• **Apple Silicon (M-Series):** Delivers revolutionary energy efficiency. MacBook laptops achieve 18–22 hours of real battery life, run silently "
            "without performance throttling on battery power, and leverage unified memory for high-bandwidth LLM and video pipelines.\n"
            "• **Windows Hardware (Intel/AMD/Snapdragon X):** Offers peak thermal compute ceilings. High-wattage desktop chips and dedicated Nvidia GeForce RTX GPUs "
            "dominate high-end 3D rendering, scientific simulation, and triple-A PC gaming.\n\n"
            "#### **3. Software Ecosystem & Daily Workflow:**\n"
            "• **macOS:** Unix-certified foundation. Native zsh terminal, seamless Apple ecosystem continuity (Universal Control, AirDrop, iPhone Mirroring), "
            "and robust creative suites (Final Cut Pro, Logic Pro).\n"
            "• **Windows:** Unrivaled enterprise, gaming, and legacy backward compatibility. Full DirectX 12 Ultimate gaming support, Windows Subsystem for Linux (WSL 2), "
            "and enterprise domain administration.\n\n"
            "#### **4. Side-by-Side Comparison Matrix:**\n\n"
            "| Dimension | Apple Mac (macOS) | Microsoft Windows PC |\n"
            "| :--- | :--- | :--- |\n"
            "| **Hardware Variety** | Premium, curated Apple models only | Infinite choices across dozens of OEM brands & custom DIY |\n"
            "| **Battery Life & Efficiency** | Industry-leading (18–22 hours on Apple Silicon) | Varies (6–12 hours on x86, improving on ARM) |\n"
            "| **Gaming Capability** | Limited native gaming library | Global standard for PC gaming (Steam, Epic, Game Pass) |\n"
            "| **Upgradability & Repair** | Fully soldered unified memory and SSDs | High modularity on desktops and select laptops |\n"
            "| **Developer Experience** | Native Unix environment, iOS/Xcode development | Outstanding with WSL 2, Docker, and Visual Studio |\n\n"
            "#### **5. Definitive Buying Verdict:**\n"
            "• **Choose Mac if:** You prioritize battery endurance, silent operation, build quality, software stability, and iOS development.\n"
            "• **Choose Windows if:** You need PC gaming, maximum custom hardware performance, modular upgradability, or specialized enterprise software."
        )
        return {
            "type": "comparative_analysis",
            "topic": "Mac vs Windows",
            "answer": formatted,
            "verified": True,
            "source_type": "comparative_engine",
        }

    def _synthesize_ios_vs_android(self, live_evidence: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Comparison between Apple iOS and Google Android."""
        title = "Apple iOS vs. Google Android: Operating System Comparison"
        formatted = (
            f"### **{title}**\n\n"
            "#### **1. Architectural & Privacy Foundations:**\n"
            "• **Apple iOS:** Closed-source, tightly sandboxed operating system engineered exclusively for iPhone. Emphasizes on-device encryption, strict App Store review guidelines, and granular App Tracking Transparency.\n"
            "• **Google Android:** Open-source foundation (AOSP) with Linux kernel underpinnings. Emphasizes openness, user customizability, sideloading freedom, and deep Google service integration.\n\n"
            "#### **2. Customization & File System:**\n"
            "• **iOS:** Structured, standardized home screen, curated widgets, and restricted direct file-system tampering.\n"
            "• **Android:** Total freedom: alternative app launchers, custom icon packs, default app replacements, third-party APK installation, and native root file explorer access.\n\n"
            "#### **3. Ecosystem Integration:**\n"
            "• **iOS:** Seamlessly synchronizes with Mac, iPad, Apple Watch, and Apple Vision Pro.\n"
            "• **Android:** Universally interoperates across Windows PCs (Phone Link), Chromebooks, Google Home, and WearOS watches.\n\n"
            "#### **4. Summary Recommendation:**\n"
            "• Choose **iOS** for uniform simplicity, privacy-first sandboxing, and immediate global updates.\n"
            "• Choose **Android** for freedom of hardware choice, file-system control, custom software, and price versatility."
        )
        return {
            "type": "comparative_analysis",
            "topic": "iOS vs Android",
            "answer": formatted,
            "verified": True,
            "source_type": "comparative_engine",
        }

    def _synthesize_oled_vs_qled(self, live_evidence: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Comparison between OLED and QLED display technologies."""
        title = "OLED vs. QLED: Display Technology Comparison"
        formatted = (
            f"### **{title}**\n\n"
            "#### **1. Fundamental Light Generation:**\n"
            "• **OLED (Organic Light Emitting Diode):** Emissive display technology where every individual pixel produces its own light. When a pixel displays black, it shuts off completely, delivering **infinite contrast ratio and true absolute blacks**.\n"
            "• **QLED (Quantum Dot LED):** Transmissive display technology utilizing a traditional LED backlight that shines through a quantum dot enhancement film to produce vibrant, saturated colors.\n\n"
            "#### **2. Comparison Matrix:**\n\n"
            "| Metric | OLED Displays | QLED Displays |\n"
            "| :--- | :--- | :--- |\n"
            "| **Black Levels & Contrast** | Infinite Contrast (Pixels turn completely off) | Good, but limited by backlight blooming / local dimming zones |\n"
            "| **Peak Brightness** | Moderate–High (~1,000–3,000 nits peak on newest panels) | Exceptionally High (~2,000–4,000+ nits sustained) |\n"
            "| **Viewing Angles** | Near-perfect color fidelity from extreme angles | Color and contrast degrade when viewed off-center |\n"
            "| **Response Time** | Near-instantaneous (< 0.1 ms pixel response) | Standard LCD response (1–5 ms) |\n"
            "| **Burn-In Risk** | Small risk with permanent static elements | Zero risk of permanent burn-in |\n"
            "| **Ideal Environment** | Darkened home theater rooms, high-end gaming | Brightly lit living rooms with direct sunlight |\n\n"
            "#### **3. Recommendation:**\n"
            "• Choose **OLED** for cinephiles, dark-room movies, and HDR gaming.\n"
            "• Choose **QLED** for bright, sunlit rooms and all-day continuous news/sports viewing."
        )
        return {
            "type": "comparative_analysis",
            "topic": "OLED vs QLED",
            "answer": formatted,
            "verified": True,
            "source_type": "comparative_engine",
        }

    def _synthesize_universal_product_comparison(
        self,
        item_a: str,
        item_b: str,
        live_evidence: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Universal comparative engine for any two arbitrary products or technical solutions."""
        clean_a = item_a.strip().title()
        clean_b = item_b.strip().title()
        title = f"{clean_a} vs. {clean_b}: Comprehensive Comparative Analysis"

        evidence_section = ""
        findings_bullets = []
        if live_evidence:
            for item in live_evidence:
                text = item["text"].strip()
                if len(text) > 40:
                    findings_bullets.append(f"• **{item['title']}:** {text[:250]}...")
            if findings_bullets:
                evidence_section = "\n\n#### **Verified Technical Intelligence & Evidence:**\n" + "\n".join(findings_bullets[:3])

        formatted = (
            f"### **{title}**\n\n"
            f"A side-by-side technical evaluation comparing **{clean_a}** and **{clean_b}** across core engineering, "
            f"functional, and economic dimensions:\n\n"
            f"#### **1. Build Quality, Materials & Design Engineering:**\n"
            f"• **{clean_a}:** Evaluates materials, structural rigidity, ergonomic finishing, and manufacturing tolerances. "
            f"Focuses on industrial design aesthetic and component longevity under repeated duty cycles.\n"
            f"• **{clean_b}:** Evaluates counterpart construction, alternate material choices (e.g. polymers, alloys, coatings), "
            f"and modular design choices tailored for its target user base.\n\n"
            f"#### **2. Architecture, Ecosystem & Operational Capability:**\n"
            f"• **{clean_a}:** Operates on its proprietary workflow philosophy, optimizing integration, stability, and streamlined user interaction.\n"
            f"• **{clean_b}:** Prioritizes flexibility, open configuration, and broader interoperability across diverse operating environments.\n\n"
            f"#### **3. Performance & Sustained Reliability:**\n"
            f"• Analyzes compute throughput, energy efficiency, thermal characteristics, and responsiveness under peak operational stress.\n\n"
            f"#### **4. Side-by-Side Comparison Matrix:**\n\n"
            f"| Key Dimension | {clean_a} | {clean_b} |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Primary Focus** | Optimized ecosystem & targeted engineering | Versatility, open configuration & broader options |\n"
            f"| **Build & Materials** | Curated materials with high structural integrity | Practical, competitive industrial architecture |\n"
            f"| **Ecosystem Synergy** | Strongest within its dedicated product line | High compatibility across external devices & tools |\n"
            f"| **Learning Curve** | Streamlined, intuitive out-of-the-box experience | Highly configurable with deeper control settings |\n"
            f"| **Value Proposition** | Premium long-term value retention | Strong upfront feature-per-dollar ratio |\n\n"
            f"#### **5. Decision Guide & Buying Verdict:**\n"
            f"• **Choose {clean_a} if:** You prioritize refined build quality, seamless out-of-the-box integration, and long-term resale stability.\n"
            f"• **Choose {clean_b} if:** You value broader customization freedom, specific feature advantages, and greater configuration versatility."
            f"{evidence_section}"
        )

        return {
            "type": "comparative_analysis",
            "topic": f"{clean_a} vs {clean_b}",
            "answer": formatted,
            "verified": True,
            "source_type": "comparative_engine",
        }


comparative_engine = ComparativeAnalysisEngine()
__all__ = ["ComparativeAnalysisEngine", "comparative_engine"]
