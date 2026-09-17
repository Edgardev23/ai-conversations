"""Prueba manual de Pronunciation Assessment (Fase 6): graba unos segundos
desde el micrófono y muestra los puntajes de Azure (modo unscripted).

Uso (desde la raíz del proyecto, con el venv activo):
    python -m scripts.test_pronunciation [segundos]
"""

import subprocess
import sys
import tempfile

from app.pronunciation import assess_pronunciation


def record(path: str, seconds: int) -> None:
    print(f"Grabando {seconds}s desde el micrófono por defecto... habla en inglés.")
    subprocess.run(
        ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(seconds), path],
        check=True,
    )


def main() -> None:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    record(path, seconds)

    print("Evaluando pronunciación con Azure...")
    scores = assess_pronunciation(path)

    print(f"\nTexto reconocido: {scores['recognized_text']}")
    print(f"Accuracy:      {scores['accuracy']}")
    print(f"Fluency:       {scores['fluency']}")
    print(f"Completeness:  {scores['completeness']}")
    print(f"Prosody:       {scores['prosody']}")
    print(f"Pronunciation: {scores['pronunciation']}")


if __name__ == "__main__":
    main()
