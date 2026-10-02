"""Interfaz Gradio: une STT + LLM + TTS + Pronunciation Assessment, y
dispara el reporte final de sesión. Usa gr.State para aislar el
historial de conversación por sesión/pestaña.
"""

import tempfile
import time

import gradio as gr

from app import db
from app.pronunciation import collect_scores, drop_assessment, submit_assessment
from app.session_report import DEFAULT_REPORT_MODEL, REPORT_MODELS, generate_report
from app.stt import transcribe_audio
from app.teacher_llm import DEFAULT_PERSONA, PERSONAS, get_teacher_turn
from app.tts import synthesize_speech

SESSION_WARNING_SECONDS = 15 * 60
DURATION_WARNING_TEXT = (
    "⏰ Llevas unos 15-20 minutos en esta sesión. Puedes seguir si quieres, "
    "pero es un buen momento para cerrarla si prefieres."
)


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


def _turn_outputs(
    history: list[dict],
    session_id: int | None,
    memory_summary: str | None,
    session_start: float | None,
    duration_notice: str,
    reply_audio_path: str | None = None,
    error: str = "",
):
    """Arma las salidas del turno en el orden que espera el evento de Gradio.

    El None del medio limpia mic_input para que quede listo para el siguiente
    turno.
    """
    return (
        history,
        _format_history(history),
        reply_audio_path,
        None,
        session_id,
        memory_summary,
        session_start,
        duration_notice,
        duration_notice,
        error,
    )


def handle_turn(
    audio_path: str | None,
    history: list[dict],
    session_id: int | None,
    memory_summary: str | None,
    session_start: float | None,
    duration_notice: str,
    persona_key: str,
):
    history = history or []

    if audio_path is None:
        return _turn_outputs(history, session_id, memory_summary, session_start, duration_notice)

    # la sesión se crea en el primer turno, ya con la persona elegida en el dropdown
    if session_id is None:
        user_id = db.get_or_create_default_user()
        session_id = db.create_session(user_id=user_id, persona=persona_key)
        memory_summary = db.get_memory_summary(user_id)
        session_start = time.time()

    # El pronunciation assessment solo depende del audio y sus puntajes no se
    # usan hasta el reporte final, pero tarda más que el resto del turno junto
    # (9-15s). Así que se encola y acá no se espera: el turno sigue con
    # STT -> LLM -> TTS y devuelve. Los puntajes se recogen al cerrar la sesión
    # (ver handle_end_session).
    assessment = submit_assessment(session_id, audio_path)

    try:
        user_text = transcribe_audio(audio_path)
    except Exception:
        drop_assessment(session_id, assessment)
        return _turn_outputs(
            history,
            session_id,
            memory_summary,
            session_start,
            duration_notice,
            error="⚠️ No pude transcribir el audio (problema de conexión con OpenAI). Intenta grabar de nuevo.",
        )

    history.append({"role": "user", "content": user_text})

    try:
        turn = get_teacher_turn(history, persona_key=persona_key, memory_summary=memory_summary)
    except Exception:
        drop_assessment(session_id, assessment)
        history.pop()  # el profesor no pudo responder; no dejamos un turno de usuario colgado
        return _turn_outputs(
            history,
            session_id,
            memory_summary,
            session_start,
            duration_notice,
            error="⚠️ El profesor no pudo responder (problema de conexión con la API). Intenta de nuevo.",
        )

    reply_text = turn["reply"]
    history.append({"role": "assistant", "content": reply_text, "suggestion": turn.get("suggestion")})

    reply_audio_path = None
    error = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            reply_audio_path = tmp.name
        synthesize_speech(reply_text, reply_audio_path)
    except Exception:
        reply_audio_path = None
        error = "⚠️ No se pudo generar el audio de la respuesta (Azure), pero el texto sí quedó arriba."

    if session_start and not duration_notice and (time.time() - session_start) >= SESSION_WARNING_SECONDS:
        duration_notice = DURATION_WARNING_TEXT

    return _turn_outputs(
        history,
        session_id,
        memory_summary,
        session_start,
        duration_notice,
        reply_audio_path=reply_audio_path,
        error=error,
    )


def handle_end_session(
    session_id: int | None,
    history: list[dict],
    model_key: str,
):
    if not history:
        return "No hubo conversación que reportar todavía."

    # Único punto donde se esperan los assessments encolados durante la sesión:
    # la charla ya terminó, así que acá la espera no interrumpe a nadie (y en
    # general ya están resueltos desde turnos anteriores).
    pronunciation_scores = collect_scores(session_id)

    try:
        report = generate_report(history, pronunciation_scores, model_key=model_key)
    except Exception:
        return "⚠️ No se pudo generar el reporte (problema de conexión con la API). Intenta de nuevo."

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
        memory_summary_state = gr.State(None)  # resumen de sesiones previas (Fase 8)
        session_start_state = gr.State(None)  # timestamp del primer turno (Fase 10)
        duration_notice_state = gr.State("")  # aviso de duración, una vez mostrado se mantiene

        persona_dropdown = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value=DEFAULT_PERSONA,
            label="Personalidad del profesor",
        )

        conversation = gr.Markdown(label="Conversación")
        duration_notice_output = gr.Markdown()
        error_output = gr.Markdown()

        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Habla aquí")
            teacher_audio = gr.Audio(label="Respuesta del profesor", autoplay=True)

        with gr.Row():
            report_model_dropdown = gr.Dropdown(
                choices=list(REPORT_MODELS.keys()),
                value=DEFAULT_REPORT_MODEL,
                label="Modelo para la retroalimentación final",
            )
            end_session_btn = gr.Button("Terminar sesión")

        report_output = gr.Markdown(label="Reporte final")

        mic_input.stop_recording(
            handle_turn,
            inputs=[
                mic_input,
                history_state,
                session_id_state,
                memory_summary_state,
                session_start_state,
                duration_notice_state,
                persona_dropdown,
            ],
            outputs=[
                history_state,
                conversation,
                teacher_audio,
                mic_input,
                session_id_state,
                memory_summary_state,
                session_start_state,
                duration_notice_state,
                duration_notice_output,
                error_output,
            ],
        )

        end_session_btn.click(
            handle_end_session,
            inputs=[session_id_state, history_state, report_model_dropdown],
            outputs=[report_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(css=".gradio-container { padding-top: 32px; }")
