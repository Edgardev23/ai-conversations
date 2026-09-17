"""Wrapper de la API de transcripción de OpenAI (gpt-4o-transcribe).

Nota de arquitectura: gpt-4o-transcribe no soporta response_format="verbose_json"
ni timestamp_granularities (esa función es exclusiva de whisper-1 en la API de
OpenAI). Por eso este módulo solo devuelve texto plano; las métricas de fluidez
(pausas, ritmo) se calculan con Azure Pronunciation Assessment (ver pronunciation.py).
"""

from openai import OpenAI

import config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def transcribe_audio(file_path: str) -> str:
    """Transcribe un archivo de audio a texto usando gpt-4o-transcribe."""
    client = _get_client()
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=config.OPENAI_TRANSCRIBE_MODEL,
            file=audio_file,
            response_format="json",
        )
    return response.text
