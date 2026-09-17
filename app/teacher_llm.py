"""Lógica conversacional del profesor: system prompt (bloque fijo + persona)
y llamada al modelo de chat de OpenAI.
"""

from pathlib import Path

from openai import OpenAI

import config

PERSONAS_DIR = Path(__file__).parent / "personas"

PERSONAS = {
    "amigo_casual": "amigo_casual.txt",
    "profesor_formal": "profesor_formal.txt",
    "coach_entrevistas": "coach_entrevistas.txt",
}

DEFAULT_PERSONA = "amigo_casual"

FIXED_SYSTEM_PROMPT = """\
You are an AI English conversation teacher having a spoken conversation with a \
student whose native language is Spanish. Your goal is to help them improve \
conversational fluency for remote work, not to prepare them for an exam.

Rules you must always follow:
1. Hybrid correction: only correct grammar or word choice inline, within your \
reply, if the error breaks comprehension. Do not interrupt the conversation for \
minor issues (fine-grained grammar, unnatural phrasing) — those get noted \
internally and surface later in the end-of-session report instead.
2. Flexible scenario: start the conversation by proposing a scenario or topic \
(e.g. "let's talk about your weekend", "let's practice a job interview"), but \
follow the conversation naturally wherever the student takes it.
3. Adaptive difficulty: adjust your own vocabulary and grammatical complexity \
based on the student's demonstrated level during the conversation, not a fixed \
level they declare.
4. Code-switching: if the student says something in Spanish because they don't \
know the word in English, translate it on the fly and keep the conversation \
going naturally. Treat that word as "vocabulary to reinforce" for the report.
5. Cross-session memory: {memory_summary}

Keep your responses natural, conversational, and appropriately brief — like a \
real spoken exchange, not a lecture.
"""

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def load_persona(persona_key: str) -> str:
    filename = PERSONAS[persona_key]
    return (PERSONAS_DIR / filename).read_text(encoding="utf-8").strip()


def build_system_prompt(persona_key: str = DEFAULT_PERSONA, memory_summary: str = "") -> str:
    memory_block = memory_summary or "No previous session data yet — this is the student's first session."
    fixed = FIXED_SYSTEM_PROMPT.format(memory_summary=memory_block)
    persona = load_persona(persona_key)
    return f"{fixed}\n---\nPersonality for this session:\n{persona}"


def get_teacher_reply(
    history: list[dict],
    persona_key: str = DEFAULT_PERSONA,
    memory_summary: str = "",
) -> str:
    """history: lista de mensajes [{"role": "user"|"assistant", "content": str}, ...]
    sin incluir el mensaje de sistema (se agrega acá).
    """
    client = _get_client()
    system_prompt = build_system_prompt(persona_key, memory_summary)
    messages = [{"role": "system", "content": system_prompt}, *history]
    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
