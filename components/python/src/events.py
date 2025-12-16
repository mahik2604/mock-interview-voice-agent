# Voice Agent Event Types
# All data flowing through the voice pipeline is represented as events.

import base64
import time
from dataclasses import dataclass
from typing import Literal, Union


def _now_ms() -> int:
    return int(time.time() * 1000)

# ========================= User / STT Events =========================

@dataclass
class UserInputEvent:
    type: Literal["user_input"]
    audio: bytes # Raw PCM audio from microphone
    ts: int

    @classmethod
    def create(cls, audio: bytes) -> "UserInputEvent":
        return cls(type="user_input", audio=audio, ts=_now_ms())


@dataclass
class STTChunkEvent:
    type: Literal["stt_chunk"]
    transcript: str # Partial transcript
    ts: int

    @classmethod
    def create(cls, transcript: str) -> "STTChunkEvent":
        return cls(type="stt_chunk", transcript=transcript, ts=_now_ms())


@dataclass
class STTOutputEvent:
    type: Literal["stt_output"]
    transcript: str # Final transcript for a turn
    ts: int

    @classmethod
    def create(cls, transcript: str) -> "STTOutputEvent":
        return cls(type="stt_output", transcript=transcript, ts=_now_ms())


STTEvent = Union[STTChunkEvent, STTOutputEvent]


# ========================= Agent Events =========================

@dataclass
class AgentChunkEvent:
    type: Literal["agent_chunk"]
    text: str # Streaming LLM output
    ts: int

    @classmethod
    def create(cls, text: str) -> "AgentChunkEvent":
        return cls(type="agent_chunk", text=text, ts=_now_ms())


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"]
    ts: int

    @classmethod
    def create(cls) -> "AgentEndEvent":
        return cls(type="agent_end", ts=_now_ms())


@dataclass
class ToolCallEvent:
    type: Literal["tool_call"]
    id: str
    name: str
    args: dict
    ts: int

    @classmethod
    def create(cls, id: str, name: str, args: dict) -> "ToolCallEvent":
        return cls(type="tool_call", id=id, name=name, args=args, ts=_now_ms())


@dataclass
class ToolResultEvent:
    type: Literal["tool_result"]
    tool_call_id: str
    name: str
    result: str
    ts: int

    @classmethod
    def create(cls, tool_call_id: str, name: str, result: str) -> "ToolResultEvent":
        return cls(
            type="tool_result",
            tool_call_id=tool_call_id,
            name=name,
            result=result,
            ts=_now_ms(),
        )


AgentEvent = Union[
    AgentChunkEvent, 
    AgentEndEvent, 
    ToolCallEvent, 
    ToolResultEvent
    ]



# ========================= TTS Events =========================


@dataclass
class TTSChunkEvent:
    type: Literal["tts_chunk"]
    audio: bytes     # PCM audio chunk
    ts: int
    
    @classmethod
    def create(cls, audio: bytes) -> "TTSChunkEvent":
        return cls(type="tts_chunk", audio=audio, ts=_now_ms())


VoiceAgentEvent = Union[
    UserInputEvent, 
    STTEvent, 
    AgentEvent, 
    TTSChunkEvent
    ]


# ========================= Serialization =========================

def event_to_dict(event: VoiceAgentEvent) -> dict:
    """Convert events to JSON-safe dicts for WebSocket transmission."""
    if isinstance(event, UserInputEvent):
        return {"type": event.type, "ts": event.ts}

    if isinstance(event, (STTChunkEvent, STTOutputEvent)):
        return {"type": event.type, "transcript": event.transcript, "ts": event.ts}

    if isinstance(event, AgentChunkEvent):
        return {"type": event.type, "text": event.text, "ts": event.ts}

    if isinstance(event, AgentEndEvent):
        return {"type": event.type, "ts": event.ts}

    if isinstance(event, ToolCallEvent):
        return {
            "type": event.type,
            "id": event.id,
            "name": event.name,
            "args": event.args,
            "ts": event.ts,
        }

    if isinstance(event, ToolResultEvent):
        return {
            "type": event.type,
            "toolCallId": event.tool_call_id,
            "name": event.name,
            "result": event.result,
            "ts": event.ts,
        }

    if isinstance(event, TTSChunkEvent):
        return {
            "type": event.type,
            "audio": base64.b64encode(event.audio).decode("ascii"),
            "ts": event.ts,
        }

    raise ValueError(f"Unknown event type: {type(event)}")