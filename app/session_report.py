"""Generación del reporte final de sesión: el LLM sintetiza la conversación
y se combina con la tendencia de puntajes de pronunciación/fluidez de Azure.
"""

import json

from openai import OpenAI

import config

REPORT_JSON_SCHEMA = {
    "name": "session_report",
    "strict": True,
    "schema": {
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
    },
}

PRONUNCIATION_METRICS = ["accuracy", "fluency", "completeness", "prosody", "pronunciation"]

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


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


def generate_report(history: list[dict], pronunciation_scores: list[dict]) -> dict:
    """history: turnos de la conversación (los del profesor pueden traer "suggestion").
    pronunciation_scores: lista de dicts de assess_pronunciation(), uno por turno del
    usuario evaluado con éxito (puede estar vacía).
    """
    client = _get_client()
    transcript = _format_transcript(history)
    prompt = (
        "Below is the transcript of an English conversation practice session between "
        "an AI teacher and a Spanish-speaking student. Write a session report for the "
        f"student.\n\nTranscript:\n{transcript}"
    )

    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": REPORT_JSON_SCHEMA},
    )
    report = json.loads(response.choices[0].message.content)
    report["pronunciation_scores"] = _aggregate_pronunciation(pronunciation_scores)
    # Sin timestamps locales (gpt-4o-transcribe no los soporta, ver stt.py), la fluidez
    # se mide solo con Azure -> ya está en pronunciation_scores["fluency"].
    report["fluency_metrics"] = {}
    return report
