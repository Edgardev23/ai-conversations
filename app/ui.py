"""Interfaz Gradio: une STT + LLM + TTS + Pronunciation Assessment, y
dispara el reporte final de sesión. Usa gr.State para aislar el
historial de conversación (y los puntajes de pronunciación) por
sesión/pestaña.
"""

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

from app import db
from app.pronunciation import assess_pronunciation
from app.session_report import DEFAULT_REPORT_MODEL, REPORT_MODELS, generate_report
from app.stt import transcribe_audio
from app.teacher_llm import DEFAULT_PERSONA, PERSONAS, get_teacher_turn
from app.tts import synthesize_speech

SESSION_WARNING_SECONDS = 15 * 60
DURATION_WARNING_TEXT = (
    "⏰ Llevas unos 15-20 minutos en esta sesión. Puedes seguir si quieres, "
    "pero es un buen momento para cerrarla si prefieres."
)


def _keep_user_audio(audio_path: str) -> str:
    """Copia la grabación a un archivo propio de la app.

    El path que entrega el micrófono vive en el caché de entradas de Gradio y se
    limpia junto con el componente; con la copia el reproductor de "Tu último
    mensaje" conserva el audio igual que el del profesor.
    """
    with tempfile.NamedTemporaryFile(suffix=Path(audio_path).suffix or ".wav", delete=False) as tmp:
        kept_path = tmp.name
    shutil.copyfile(audio_path, kept_path)
    return kept_path


def _format_history(history: list[dict]) -> list[dict]:
    """Pasa el historial interno al formato de mensajes que espera gr.Chatbot."""
    messages = []
    for turn in history:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
            continue
        content = turn["content"]
        suggestion = turn.get("suggestion")
        if suggestion:
            content = f"{content}\n\n💡 {suggestion}"
        messages.append({"role": "assistant", "content": content})
    return messages


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


def handle_turn(
    audio_path: str | None,
    history: list[dict],
    pronunciation_scores: list[dict],
    session_id: int | None,
    memory_summary: str | None,
    session_start: float | None,
    duration_notice: str,
    persona_key: str,
):
    history = history or []
    pronunciation_scores = pronunciation_scores or []

    def result(user_audio=None, reply_audio=None, error=""):
        # el None de mic_input lo limpia para que quede listo para el siguiente turno
        return (
            history,
            _format_history(history),
            user_audio,
            reply_audio,
            None,
            pronunciation_scores,
            session_id,
            memory_summary,
            session_start,
            duration_notice,
            duration_notice,
            error,
        )

    if audio_path is None:
        return result()

    user_audio_path = _keep_user_audio(audio_path)

    # la sesión se crea en el primer turno, ya con la persona elegida en el dropdown
    if session_id is None:
        user_id = db.get_or_create_default_user()
        session_id = db.create_session(user_id=user_id, persona=persona_key)
        memory_summary = db.get_memory_summary(user_id)
        session_start = time.time()

    # El pronunciation assessment solo depende del audio (no del texto ni de la
    # respuesta del profesor) y sus resultados no se usan hasta el reporte
    # final, así que corre en paralelo mientras seguimos con STT -> LLM -> TTS.
    pronunciation_executor = ThreadPoolExecutor(max_workers=1)
    pronunciation_future = pronunciation_executor.submit(assess_pronunciation, audio_path)

    try:
        user_text = transcribe_audio(audio_path)
    except Exception:
        pronunciation_executor.shutdown(wait=False)
        return result(
            user_audio=user_audio_path,
            error="⚠️ No pude transcribir el audio (problema de conexión con OpenAI). Intenta grabar de nuevo.",
        )

    history.append({"role": "user", "content": user_text})

    try:
        turn = get_teacher_turn(history, persona_key=persona_key, memory_summary=memory_summary)
    except Exception:
        pronunciation_executor.shutdown(wait=False)
        history.pop()  # el profesor no pudo responder; no dejamos un turno de usuario colgado
        return result(
            user_audio=user_audio_path,
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

    try:
        pronunciation_scores.append(pronunciation_future.result())
    except RuntimeError:
        pass  # Azure no reconoció habla en el turno (silencio/ruido); no bloquea la charla
    finally:
        pronunciation_executor.shutdown(wait=False)

    if session_start and not duration_notice and (time.time() - session_start) >= SESSION_WARNING_SECONDS:
        duration_notice = DURATION_WARNING_TEXT

    return result(user_audio=user_audio_path, reply_audio=reply_audio_path, error=error)


def handle_end_session(
    session_id: int | None,
    history: list[dict],
    pronunciation_scores: list[dict],
    model_key: str,
):
    if not history:
        return "No hubo conversación que reportar todavía."

    try:
        report = generate_report(history, pronunciation_scores or [], model_key=model_key)
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
        pronunciation_state = gr.State([])  # puntajes de Azure por turno del usuario
        memory_summary_state = gr.State(None)  # resumen de sesiones previas (Fase 8)
        session_start_state = gr.State(None)  # timestamp del primer turno (Fase 10)
        duration_notice_state = gr.State("")  # aviso de duración, una vez mostrado se mantiene

        persona_dropdown = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value=DEFAULT_PERSONA,
            label="Personalidad del profesor",
        )

        # Chatbot (y no Markdown) para que el log haga autoscroll al último turno
        # en vez de volver al inicio cada vez que responde el profesor.
        conversation = gr.Chatbot(label="Conversación", height=420, autoscroll=True)
        duration_notice_output = gr.Markdown()
        error_output = gr.Markdown()

        with gr.Row():
            # editable=False quita los controles de recorte: al soltar el botón el
            # turno se envía solo, así que no hay nada que editar sobre la grabación.
            mic_input = gr.Audio(
                sources=["microphone"], type="filepath", label="Habla aquí", editable=False
            )
            # mismo reproductor que el del profesor (sin autoplay, para que no suene
            # encima de la respuesta); conserva la última grabación del usuario
            user_audio = gr.Audio(label="Tu último mensaje (escúchate)")
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
                pronunciation_state,
                session_id_state,
                memory_summary_state,
                session_start_state,
                duration_notice_state,
                persona_dropdown,
            ],
            outputs=[
                history_state,
                conversation,
                user_audio,
                teacher_audio,
                mic_input,
                pronunciation_state,
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
            inputs=[session_id_state, history_state, pronunciation_state, report_model_dropdown],
            outputs=[report_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(css=".gradio-container { padding-top: 32px; }")
