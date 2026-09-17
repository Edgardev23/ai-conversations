"""Prueba manual del motor conversacional (Fase 3): chat por texto en la
terminal, sin voz todavía.

Uso (desde la raíz del proyecto, con el venv activo):
    python -m scripts.test_teacher [persona]

persona: amigo_casual (default) | profesor_formal | coach_entrevistas
Escribe "exit" para salir.
"""

import sys

from app.teacher_llm import DEFAULT_PERSONA, PERSONAS, get_teacher_turn


def main() -> None:
    persona_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERSONA
    if persona_key not in PERSONAS:
        print(f"Persona desconocida '{persona_key}'. Opciones: {list(PERSONAS)}")
        return

    print(f"Persona: {persona_key} — escribe 'exit' para salir.\n")

    history: list[dict] = []
    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})
        turn = get_teacher_turn(history, persona_key=persona_key)
        history.append({"role": "assistant", "content": turn["reply"], "suggestion": turn.get("suggestion")})

        print(f"Profesor: {turn['reply']}")
        if turn.get("suggestion"):
            print(f"💡 {turn['suggestion']}")
        print()


if __name__ == "__main__":
    main()
