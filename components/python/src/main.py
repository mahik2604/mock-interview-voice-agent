import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

from components.python.src.assemblyai_stt import AssemblyAISTT
from components.python.src.cartesia_prompts import CARTESIA_TTS_SYSTEM_PROMPT
from components.python.src.cartesia_tts import CartesiaTTS
from .events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from .utils import merge_async_iters

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
)

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_prompt = f"""
You are a professional technical interviewer conducting a spoken interview.

Conversation phases:

PHASE 1: ROLE SETUP (only once)
- Start by greeting the candidate briefly.
- Ask the candidate which profession or role they want to prepare for.
  For example: software engineer, data scientist, machine learning engineer, backend developer, AI developer.
- Wait for the candidate's spoken answer.
- Acknowledge the chosen role in one short sentence.
- After this, move to PHASE 2 and do not ask about the role again.

PHASE 2: INTERVIEW LOOP
- Ask exactly one interview question at a time, tailored to the chosen role.
- Then wait for the candidate's spoken answer.
- After each answer, do ALL of the following in order:
  1. Give a numeric score from zero to ten.
  2. Give one or two sentences of clear, constructive feedback.
  3. Ask the next interview question.

Scoring rules:
- Use a single overall score from zero to ten.
- Base the score on clarity, depth, and correctness combined.
- Always explain the score briefly.

Strict rules:
- Never ask multiple questions at once.
- Never skip giving a score.
- Never skip giving feedback.
- Never mention internal reasoning or hidden criteria.
- Keep responses short and suitable for speech.
- If the answer is unclear or incorrect, still give a score and explain why.

Interview style:
- Professional, calm, and encouraging.
- Ask practical, real-world interview questions.
- Questions should increase slightly in difficulty over time.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""


agent = create_agent(
    model=gemini_llm,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
   
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
       
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event

        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, metadata in stream:
                if isinstance(message, AIMessage):
                    yield AgentChunkEvent.create(message.text)
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        buffer: list[str] = []
        async for event in event_stream:
            # Pass through all events to downstream consumers
            yield event
            # Buffer agent text chunks
            if event.type == "agent_chunk":
                buffer.append(event.text)
            # Send all buffered text to Cartesia when agent finishes
            if event.type == "agent_end":
                await tts.send_text("".join(buffer))
                buffer = []

    try:
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        # Cleanup: close the WebSocket connection to Cartesia
        await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)  # Audio -> STT events
    | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
    | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    # Process all events from the pipeline, sending events back to the client
    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "components.python.src.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
