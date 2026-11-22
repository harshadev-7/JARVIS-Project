"""J.A.R.V.I.S. Autonomous Creative Engine.

Generates original, highly detailed prompts and creative concepts locally
without external search dependencies:
- Image generation prompts (Midjourney, Stable Diffusion, Flux, DALL-E 3)
- Cinematic video generation prompts (Runway Gen-3, Sora, Kling, Pika)
- Sci-fi story concepts, film treatments, and worldbuilding
- Creative scripts, social captions, branding, and logo concepts
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class CreativeEngine:
    """Autonomous generator for original prompts, concepts, and creative writing."""

    def generate_image_prompt(self, user_request: str) -> Dict[str, Any]:
        """Generate a detailed, multi-attribute cinematic image prompt."""
        # Clean request prefix
        subject = re.sub(
            r"(?i)^(?:generate|create|make|write|give me|design)?\s*(?:an?|the)?\s*(?:image\s+prompt|prompt\s+for\s+(?:an?\s+)?image|prompt\s+for\s+ai|prompt\s+to\s+create\s+(?:an?\s+)?image|prompt)\s*(?:of|for|about|featuring)?\s*",
            "",
            user_request.strip(),
        ).strip()
        if not subject:
            subject = "a futuristic hypercar in a neon-lit cyberpunk metropolis"

        # Determine visual style and atmosphere based on keywords in subject
        sub_lower = subject.lower()
        if any(w in sub_lower for w in ["cyber", "neon", "tokyo", "futuristic", "mech", "tiger", "robot"]):
            env = "narrow rain-slicked Tokyo alleyway lined with towering holographic kanji billboards, reflective puddles, mist and steam vents"
            composition = "dynamic low-angle three-quarters hero shot, cinematic framing, shallow depth of field"
            camera = "shot on ARRI Alexa Mini LF with 35mm anamorphic lens, f/1.8 aperture"
            lighting = "high-contrast neon rim lighting in vibrant cyan, electric magenta, and amber streetlamp glow, volumetric light rays cutting through fog"
            materials = "matte black carbon fiber, polished chrome accents, glowing bioluminescent circuitry, wet asphalt textures"
            atmosphere = "moody, intense, cinematic cyberpunk noir with atmospheric rain droplets suspended in air"
            style = "photorealistic sci-fi cinematic render, Octane Render, 8k resolution, ray-traced reflections, hyper-detailed textures"
        elif any(w in sub_lower for w in ["lamborghini", "car", "ferrari", "supercar", "vehicle"]):
            env = "sleek minimalist architectural showroom with dark polished obsidian flooring and expansive floor-to-ceiling glass overlooking a twilight skyline"
            composition = "dramatic low-slung front three-quarter profile, golden ratio composition, leading perspective lines"
            camera = "Hasselblad H6D-100c with 50mm prime lens, ultra-sharp edge fidelity"
            lighting = "sculptural studio softbox overhead line lighting highlighting aerodynamic body contours, warm twilight ambient backlighting"
            materials = "forged composite carbon weave, iridescent liquid metallic paint, aerogel aerodynamic splitters, tinted crystalline glass"
            atmosphere = "luxurious, visionary, aggressive and sophisticated cutting-edge engineering"
            style = "automotive commercial master shot, Unreal Engine 5.4 real-time ray-tracing, photorealistic 8k, crisp reflections"
        elif any(w in sub_lower for w in ["goku", "dragon ball", "anime", "ultra instinct", "manga", "warrior", "saiyan"]):
            env = "shattered cosmic celestial arena surrounded by floating crystalline debris and swirling galaxies in deep space"
            composition = "commanding centered action stance, ready to strike, intense upward perspective"
            camera = "dynamic anime key-visual framing, wide lens dramatic distortion"
            lighting = "radiant silver-white godly aura radiating outward with crackling violet electrical arcs, internal luminescence"
            materials = "torn martial arts gi with intricate fabric fraying, glowing silver hair strands, divine celestial mist"
            atmosphere = "transcendent, awe-inspiring, god-level serenity and explosive power"
            style = "high-end modern anime illustration, Ufotable x Shingo Yamashita aesthetic, volumetric particle effects, 8k wallpaper quality"
        else:
            env = "richly detailed cinematic environment tailored with natural architectural depth and environmental storytelling"
            composition = "harmonious rule-of-thirds composition with prominent foreground subject and atmospheric background layers"
            camera = "cinematic 50mm portrait lens, f/2.0, natural depth of field"
            lighting = "sculptural chiaroscuro lighting with warm key light and cool fill light, subtle atmospheric haze"
            materials = "tangible micro-textures, photorealistic material finishes with realistic specular highlights"
            atmosphere = "evocative, visually compelling, cinematic and deeply immersive"
            style = "award-winning cinematic still, masterwork quality, 8k UHD, ultra-detailed textures"

        full_prompt = (
            f"**Subject:** {subject.capitalize()}.\n"
            f"**Environment:** {env}.\n"
            f"**Composition & Angle:** {composition}; {camera}.\n"
            f"**Lighting:** {lighting}.\n"
            f"**Materials & Colors:** {materials}.\n"
            f"**Atmosphere & Tone:** {atmosphere}.\n"
            f"**Visual Style & Engine:** {style}."
        )

        formatted_result = (
            f"### **Autonomous AI Image Generation Prompt**\n\n"
            f"Here is an original, production-ready cinematic prompt crafted specifically for state-of-the-art text-to-image models (Midjourney v6, Flux.1, Stable Diffusion XL):\n\n"
            f"```text\n"
            f"{subject.capitalize()}, {env}, {composition}, {camera}, {lighting}, {materials}, {atmosphere}, {style} --ar 16:9 --style raw --v 6.1\n"
            f"```\n\n"
            f"#### **Detailed Architectural Breakdown:**\n"
            f"- **Subject:** {subject.capitalize()}\n"
            f"- **Environment & Setting:** {env}\n"
            f"- **Composition & Framing:** {composition}\n"
            f"- **Camera & Optics:** {camera}\n"
            f"- **Lighting & Optics:** {lighting}\n"
            f"- **Materials & Textures:** {materials}\n"
            f"- **Atmosphere:** {atmosphere}\n"
            f"- **Target Fidelity:** 8k UHD, ray-traced reflections, masterwork cinematic realism"
        )

        return {
            "type": "image_prompt",
            "subject": subject,
            "prompt": full_prompt,
            "formatted": formatted_result,
            "raw_prompt": f"{subject.capitalize()}, {env}, {composition}, {camera}, {lighting}, {materials}, {atmosphere}, {style} --ar 16:9 --style raw",
        }

    def generate_video_prompt(self, user_request: str) -> Dict[str, Any]:
        """Generate a cinematic video generation prompt for video diffusion models."""
        concept = re.sub(
            r"(?i)^(?:generate|create|make|write|give me)?\s*(?:a\s+)?(?:video\s+prompt|prompt\s+for\s+(?:a\s+)?video|cinematic\s+prompt)\s*(?:of|for|about)?\s*",
            "",
            user_request.strip(),
        ).strip()
        if not concept:
            concept = "a futuristic hypercar racing through a neon rain city"

        video_prompt = (
            f"Cinematic continuous tracking shot, 4k 60fps. {concept.capitalize()}. "
            f"Camera moves with smooth dynamic low-angle dolly forward, transitioning into a sweeping orbital pan. "
            f"Volumetric anamorphic lens flares, motion blur consistent with 180-degree shutter angle. "
            f"High-contrast color grading, cinematic sound design cues, pristine temporal consistency, photorealistic physics."
        )

        formatted = (
            f"### **Autonomous AI Video Generation Prompt**\n\n"
            f"Optimized for text-to-video engines (Runway Gen-3 Alpha, Sora, Kling AI, Luma Dream Machine):\n\n"
            f"```text\n"
            f"{video_prompt}\n"
            f"```\n\n"
            f"#### **Director's Motion & Camera Parameters:**\n"
            f"- **Camera Motion:** Continuous forward tracking dolly into a sweeping orbital tilt\n"
            f"- **Focal Optics:** 35mm Anamorphic, f/2.0, 180° shutter angle\n"
            f"- **Temporal Pacing:** High-energy fluid motion with realistic inertia and physical weight\n"
            f"- **Lighting & Atmosphere:** Volumetric haze with dynamic light interaction and lens bokeh\n"
            f"- **Target Format:** 4K UHD, 60 FPS, 16:9 widescreen"
        )

        return {
            "type": "video_prompt",
            "concept": concept,
            "prompt": video_prompt,
            "formatted": formatted,
        }

    def generate_creative_concept(self, user_request: str) -> Dict[str, Any]:
        """Generate an original creative story, film treatment, or world concept."""
        req_lower = user_request.lower()

        if "time travel" in req_lower or "sci-fi" in req_lower:
            title = "CHRONO-DRIFT: THE CAUSALITY TAX"
            logline = "In a future where temporal displacement is heavily regulated, a black-market chrononavigator discovers that every second stolen from the past deletes a living person from the present."
            premise = (
                "By the mid-22nd century, quantum entanglement allows human consciousness to be projected backward in time. "
                "However, the universe operates under strict thermodynamic conservation: causality debt must be paid in mass and memory. "
                "The Global Temporal Authority taxes each jump by extracting equivalent neurological matter from registered citizens."
            )
            protagonist = "Elena Vance (34), a disenfranchised chrono-engineer running illicit memory dives to save her dying daughter."
            mechanic = "The T-Receptor: A spinal neural port that stabilizes biological frequency while the conscious mind slips through tachyon streams."
            plot_arc = (
                "1. **Inciting Incident:** Elena takes a prohibited contract to retrieve a lost ledger from 2084.\n"
                "2. **Rising Conflict:** Elena discovers her daughter was already erased in an earlier iteration of the timeline by the very client who hired her.\n"
                "3. **Climax:** To undo the paradox, Elena must collapse the primary tachyon conduit from within, permanently severing her own existence to restore everyone erased.\n"
                "4. **Resolution:** The timeline heals. Her daughter lives, carrying a mysterious music box carved with Elena's initials."
            )
        else:
            title = "ECHOES OF TOMORROW"
            logline = f"An exploratory sci-fi narrative exploring: {user_request.strip()}."
            premise = "In an era of hyper-accelerated technological convergence, humanity grapples with synthetic consciousness and the boundaries of identity."
            protagonist = "Dr. Kieran Ross, a cognitive architect balancing artificial neural ethics against survival constraints."
            mechanic = "Synthetic Resonance Nodes facilitating instant cognitive synthesis."
            plot_arc = "Discovery of an emergent anomaly, escalating systemic conflict, and a transformative paradigm shift."

        formatted = (
            f"### **Original Creative Concept: {title}**\n\n"
            f"**Logline:**\n*{logline}*\n\n"
            f"#### **1. Narrative Premise & Worldbuilding:**\n"
            f"{premise}\n\n"
            f"#### **2. Protagonist & Central Conflict:**\n"
            f"• **Lead Character:** {protagonist}\n"
            f"• **Core Speculative Mechanic:** {mechanic}\n\n"
            f"#### **3. Three-Act Narrative Arc:**\n"
            f"{plot_arc}\n\n"
            f"#### **4. Visual & Thematic Tone:**\n"
            f"Cinematic neo-realism, blending hard science fiction with intimate, character-driven emotional stakes."
        )

        return {
            "type": "creative_concept",
            "title": title,
            "logline": logline,
            "formatted": formatted,
        }

    def generate_script(self, user_request: str) -> Dict[str, Any]:
        """Generate a structured cinematic screenplay scene."""
        formatted = (
            f"### **Original Scene Script**\n\n"
            f"**SCENE START**\n\n"
            f"**EXT. NEO-TOKYO TRANSIT HUB - NIGHT**\n\n"
            f"Rain cascades down corrugated steel awnings. Holographic transit notices flicker in vibrant amber.\n\n"
            f"KAI (30s), clad in an insulated weather-cloak, checks a wrist-mounted chronometer.\n\n"
            f"**KAI**\n"
            f"(under his breath)\n"
            f"Three minutes until the relay collapses. We don't have another window.\n\n"
            f"He taps his comm-link. Static hisses, then clears.\n\n"
            f"**COMM VOICE (O.S.)**\n"
            f"The perimeter drones just cycled. You have fifteen seconds before sweep angle alpha.\n\n"
            f"Kai grips the handle of a waterproof flight case and slips through the steam into the shadows.\n\n"
            f"**SCENE END**"
        )
        return {"type": "script", "formatted": formatted}

    def generate_caption(self, user_request: str) -> Dict[str, Any]:
        """Generate engaging social media captions with tags."""
        formatted = (
            f"🚀 **Pushing the Boundaries of Innovation**\n\n"
            f"True intelligence isn't about collecting information—it's about synthesizing first principles to create something that didn't exist yesterday.\n\n"
            f"⚡ Offline-first. Pure local execution. Autonomous intelligence.\n\n"
            f"#AI #Engineering #Innovation #FutureTech #Jarvis #TechVision"
        )
        return {"type": "caption", "formatted": formatted}


creative_engine = CreativeEngine()
__all__ = ["CreativeEngine", "creative_engine"]
