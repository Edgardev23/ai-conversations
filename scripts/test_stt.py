"""Prueba manual del módulo STT (Fase 2): graba unos segundos desde el
micrófono por defecto y muestra la transcripción de gpt-4o-transcribe.

Uso (desde la raíz del proyecto, con el venv activo):
    python -m scripts.test_stt [segundos]
"""

import subprocess
import sys
import tempfile

from app.stt import transcribe_audio


def record(path: str, seconds: int) -> None:
    print(f"Grabando {seconds}s desde el micrófono por defecto... habla ahora.")
    subprocess.run(
        ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(seconds), path],
        check=True,
    )


def main() -> None:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    record(path, seconds)

    print("Transcribiendo con gpt-4o-transcribe...")
    text = transcribe_audio(path)
    print(f"\nTranscripción:\n{text}")


if __name__ == "__main__":
    main()
