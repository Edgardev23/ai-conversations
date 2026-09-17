"""Interfaz Gradio: une STT + LLM + TTS + Pronunciation Assessment, y
dispara el reporte final de sesión. Usa gr.State para aislar el
historial de conversación (y los puntajes de pronunciación) por
sesión/pestaña.
"""

import tempfile

import gradio as gr

from app import db
from app.pronunciation import assess_pronunciation
from app.session_report import generate_report
from app.stt import transcribe_audio
from app.teacher_llm import DEFAULT_PERSONA, get_teacher_turn
from app.tts import synthesize_speech


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        if turn["role"] == "user":
            lines.append(f"**Tú:** {turn['content']}")
            continue
        lines.append(f"**Profesor:** {turn['content']}")
        suggestion = turn.get("suggestion")
        if suggestion:
            lines.append(f"💡 {suggestion}")
    return "\n\n".join(lines)


def _format_report(report: dict) -> str:
    lines = ["## Reporte de la sesión", "", report["transcript_summary"], ""]

    if report["grammar_errors"]:
        lines.append("**Gramática a reforzar:**")
        lines += [f"- {item}" for item in report["grammar_errors"]]
        lines.append("")

    if report["vocab_gaps"]:
        lines.append("**Vocabulario pendiente:**")
        lines += [f"- {item}" for item in report["vocab_gaps"]]
        lines.append("")

    if report["pronunciation_scores"]:
        lines.append("**Pronunciación (promedio / primer turno / último turno):**")
        for metric, values in report["pronunciation_scores"].items():
            lines.append(f"- {metric}: {values['average']} / {values['first']} / {values['last']}")
        lines.append("")

    lines.append("**Recomendaciones:**")
    lines.append(report["recommendations"])

    return "\n".join(lines)


def start_session():
    user_id = db.get_or_create_default_user()
    session_id = db.create_session(user_id=user_id, persona=DEFAULT_PERSONA)
    memory_summary = db.get_memory_summary(user_id)
    return session_id, memory_summary


def handle_turn(
    audio_path: str | None,
    history: list[dict],
    pronunciation_scores: list[dict],
    memory_summary: str,
):
    history = history or []
    pronunciation_scores = pronunciation_scores or []

    if audio_path is None:
        return history, _format_history(history), None, None, pronunciation_scores

    user_text = transcribe_audio(audio_path)
    history.append({"role": "user", "content": user_text})

    try:
        pronunciation_scores.append(assess_pronunciation(audio_path))
    except RuntimeError:
        pass  # Azure no reconoció habla en el turno (silencio/ruido); no bloquea la charla

    turn = get_teacher_turn(history, persona_key=DEFAULT_PERSONA, memory_summary=memory_summary)
    reply_text = turn["reply"]
    history.append({"role": "assistant", "content": reply_text, "suggestion": turn.get("suggestion")})

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        reply_audio_path = tmp.name
    synthesize_speech(reply_text, reply_audio_path)

    # el último None limpia el input de audio para que quede listo para el siguiente turno
    return history, _format_history(history), reply_audio_path, None, pronunciation_scores


def handle_end_session(session_id: int, history: list[dict], pronunciation_scores: list[dict]):
    if not history:
        return "No hubo conversación que reportar todavía."

    report = generate_report(history, pronunciation_scores or [])

    db.end_session(session_id)
    user_id = db.get_or_create_default_user()
    db.save_session_report(session_id, user_id, report)

    return _format_report(report)


def build_app() -> gr.Blocks:
    db.init_db()

    with gr.Blocks(title="Práctica de Speaking en Inglés") as demo:
        gr.Markdown("# Práctica de Speaking en Inglés")

        session_id_state = gr.State(None)
        history_state = gr.State([])  # historial de conversación, aislado por sesión/pestaña
        pronunciation_state = gr.State([])  # puntajes de Azure por turno del usuario
        memory_summary_state = gr.State("")  # resumen de sesiones previas (Fase 8)

        conversation = gr.Markdown(label="Conversación")

        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Habla aquí")
            teacher_audio = gr.Audio(label="Respuesta del profesor", autoplay=True)

        end_session_btn = gr.Button("Terminar sesión")
        report_output = gr.Markdown(label="Reporte final")

        demo.load(start_session, outputs=[session_id_state, memory_summary_state])

        mic_input.stop_recording(
            handle_turn,
            inputs=[mic_input, history_state, pronunciation_state, memory_summary_state],
            outputs=[history_state, conversation, teacher_audio, mic_input, pronunciation_state],
        )

        end_session_btn.click(
            handle_end_session,
            inputs=[session_id_state, history_state, pronunciation_state],
            outputs=[report_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
