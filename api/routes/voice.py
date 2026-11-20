"""
JARVIS AI — Voice Assistant API Routes
=======================================
Endpoints for Neural Text-To-Speech (TTS), Speech-To-Text (STT),
Wake-Word status, and voice loop orchestration.
"""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from services.voice.tts_engine import tts_engine
from services.voice.wake_word import wake_word_detector, VoiceState
from services.voice.voice_orchestrator import voice_orchestrator

router = APIRouter(prefix="/voice", tags=["Voice Assistant"])


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text for JARVIS to speak")
    voice: str = Field("en-GB-RyanNeural", description="Neural voice identifier")


class VoiceTurnRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Spoken transcript or command")


@router.post("/synthesize", summary="Synthesize natural voice MP3 audio")
async def synthesize_speech(request: SynthesizeRequest):
    """Generate and stream neural voice audio."""
    try:
        audio_bytes = await tts_engine.synthesize_bytes(request.text, voice=request.voice)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", summary="Process a voice conversation turn")
async def process_turn(request: VoiceTurnRequest):
    """Handle wake-word evaluation, AI reasoning, and TTS reply."""
    result = await voice_orchestrator.process_voice_turn(simulated_transcript=request.transcript)
    return result


@router.get("/status", summary="Get voice engine state")
async def get_voice_status():
    return {
        "state": wake_word_detector.state,
        "wake_word": "Hey Jarvis",
        "default_voice": tts_engine.DEFAULT_VOICE,
    }


@router.post("/interrupt", summary="Trigger voice interruption")
async def interrupt_speech():
    wake_word_detector.trigger_interruption()
    return {"message": "Voice playback interrupted", "state": wake_word_detector.state}
