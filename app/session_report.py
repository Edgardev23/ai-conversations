"""Generación del reporte final de sesión: el LLM sintetiza la conversación
y se combina con la tendencia de puntajes de pronunciación/fluidez de Azure.
"""

import json

import anthropic
from openai import OpenAI

import config

# Modelos "inteligentes" para la retroalimentación final (no para la
# conversación turno a turno, que usa el modelo económico de teacher_llm).
REPORT_MODELS = {
    "OpenAI — gpt-5.6-sol": ("openai", "gpt-5.6-sol"),
    "Claude — Opus 5": ("anthropic", "claude-opus-5"),
}

DEFAULT_REPORT_MODEL = "OpenAI — gpt-5.6-sol"

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript_summary": {
            "type": "string",
            "description": "2-4 sentence summary of what was discussed this session.",
        },
        "grammar_errors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Grammar or phrasing issues observed this session, one short description each.",
        },
        "vocab_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "English words/phrases the student didn't know and said in Spanish instead.",
        },
        "recommendations": {
            "type": "string",
            "description": "2-4 concrete, actionable recommendations of what to practice next.",
        },
    },
    "required": ["transcript_summary", "grammar_errors", "vocab_gaps", "recommendations"],
    "additionalProperties": False,
}

PRONUNCIATION_METRICS = ["accuracy", "fluency", "completeness", "prosody", "pronunciation"]

_openai_client = None
_anthropic_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _format_transcript(history: list[dict]) -> str:
    lines = []
    for turn in history:
        speaker = "Student" if turn["role"] == "user" else "Teacher"
        lines.append(f"{speaker}: {turn['content']}")
        if turn.get("suggestion"):
            lines.append(f"(Note: a more natural way to say the student's line above: {turn['suggestion']})")
    return "\n".join(lines)


def _aggregate_pronunciation(pronunciation_scores: list[dict]) -> dict:
    """pronunciation_scores: salida de assess_pronunciation() por turno del usuario."""
    aggregated = {}
    for metric in PRONUNCIATION_METRICS:
        values = [turn[metric] for turn in pronunciation_scores if turn.get(metric) is not None]
        if values:
            aggregated[metric] = {
                "average": round(sum(values) / len(values), 1),
                "first": round(values[0], 1),
                "last": round(values[-1], 1),
            }
    return aggregated


def _generate_report_text_openai(model_id: str, prompt: str) -> dict:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "session_report", "strict": True, "schema": REPORT_SCHEMA},
        },
    )
    return json.loads(response.choices[0].message.content)


def _generate_report_text_anthropic(model_id: str, prompt: str) -> dict:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model_id,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": REPORT_SCHEMA}},
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def generate_report(
    history: list[dict],
    pronunciation_scores: list[dict],
    model_key: str = DEFAULT_REPORT_MODEL,
) -> dict:
    """history: turnos de la conversación (los del profesor pueden traer "suggestion").
    pronunciation_scores: lista de dicts de assess_pronunciation(), uno por turno del
    usuario evaluado con éxito (puede estar vacía).
    """
    provider, model_id = REPORT_MODELS[model_key]
    transcript = _format_transcript(history)
    prompt = (
        "Below is the transcript of an English conversation practice session between "
        "an AI teacher and a Spanish-speaking student. Write a session report for the "
        f"student.\n\nTranscript:\n{transcript}"
    )

    if provider == "openai":
        report = _generate_report_text_openai(model_id, prompt)
    elif provider == "anthropic":
        report = _generate_report_text_anthropic(model_id, prompt)
    else:
        raise ValueError(f"Proveedor de modelo desconocido: {provider}")

    report["pronunciation_scores"] = _aggregate_pronunciation(pronunciation_scores)
    # Sin timestamps locales (gpt-4o-transcribe no los soporta, ver stt.py), la fluidez
    # se mide solo con Azure -> ya está en pronunciation_scores["fluency"].
    report["fluency_metrics"] = {}
    return report
