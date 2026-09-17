"""Lógica conversacional del profesor: system prompt (bloque fijo + persona)
y llamada al modelo de chat de OpenAI.
"""

import json
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

You must respond with two separate channels, "reply" and "suggestion":
- "reply" is what gets spoken out loud to the student. Keep it natural,
  conversational, and appropriately brief — like a real spoken exchange, not a
  lecture.
- "suggestion" is shown to the student as text, never spoken. Use it to point
  out a more natural or correct way to phrase something they just said this
  turn — quote their original phrase and give a better alternative in one or
  two short sentences. Set it to null if nothing stands out.

Rules you must always follow:
1. Hybrid correction: only correct grammar or word choice inline, within \
"reply", if the error breaks comprehension. Never let "reply" turn into a \
correction lecture — minor issues (fine-grained grammar, unnatural phrasing) \
belong in "suggestion" instead, every turn, not just when they break \
comprehension.
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
"""

TURN_JSON_SCHEMA = {
    "name": "teacher_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "Natural, conversational reply to be spoken out loud to the student.",
            },
            "suggestion": {
                "type": ["string", "null"],
                "description": (
                    "A short, friendly written tip about a more natural or correct way to "
                    "phrase what the student said this turn, quoting their original phrase and "
                    "a better alternative. Null if nothing stands out."
                ),
            },
        },
        "required": ["reply", "suggestion"],
        "additionalProperties": False,
    },
}

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


def get_teacher_turn(
    history: list[dict],
    persona_key: str = DEFAULT_PERSONA,
    memory_summary: str = "",
) -> dict:
    """history: lista de mensajes [{"role": "user"|"assistant", "content": str}, ...]
    sin incluir el mensaje de sistema (se agrega acá).

    Devuelve {"reply": str, "suggestion": str | None}. "reply" es lo que se
    sintetiza en audio; "suggestion" se muestra como texto aparte, sin hablar.
    """
    client = _get_client()
    system_prompt = build_system_prompt(persona_key, memory_summary)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": turn["role"], "content": turn["content"]} for turn in history]
    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": TURN_JSON_SCHEMA},
    )
    return json.loads(response.choices[0].message.content)
