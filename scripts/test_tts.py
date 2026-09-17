"""Prueba manual de TTS (Fase 4): sintetiza una frase con Azure y la reproduce.

Uso (desde la raíz del proyecto, con el venv activo):
    python -m scripts.test_tts ["texto a sintetizar"]
"""

import subprocess
import sys
import tempfile

from app.tts import synthesize_speech

DEFAULT_TEXT = "Hi! I'm your English conversation teacher. Nice to meet you."


def main() -> None:
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    print(f"Sintetizando: {text!r}")
    synthesize_speech(text, path)

    print(f"Reproduciendo {path} ...")
    # paplay (en vez de aplay) respeta el sink por defecto de PipeWire/PulseAudio.
    subprocess.run(["paplay", path], check=True)


if __name__ == "__main__":
    main()
