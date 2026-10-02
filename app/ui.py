"""Interfaz Gradio: une STT + LLM + TTS + Pronunciation Assessment, y
dispara el reporte final de sesión. Usa gr.State para aislar el
historial de conversación (y los puntajes de pronunciación) por
sesión/pestaña.
"""

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

from app import db
from app.pronunciation import assess_pronunciation
from app.session_report import DEFAULT_REPORT_MODEL, REPORT_MODELS, generate_report
from app.stt import transcribe_audio
from app.teacher_llm import DEFAULT_PERSONA, get_teacher_turn
from app.tts import synthesize_speech
from app.ui_html import HEADER_HTML, PERSONA_META, render_conversation, render_persona_cards

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

SESSION_WARNING_SECONDS = 15 * 60
DURATION_WARNING_TEXT = (
    "⏰ Llevas unos 15-20 minutos en esta sesión. Puedes seguir si quieres, "
    "pero es un buen momento para cerrarla si prefieres."
)


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

    no_op = (
        history,
        render_conversation(history, persona_key),
        None,
        None,
        pronunciation_scores,
        session_id,
        memory_summary,
        session_start,
        duration_notice,
        duration_notice,
        "",
    )

    if audio_path is None:
        return no_op

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
        return (*no_op[:-1], "⚠️ No pude transcribir el audio (problema de conexión con OpenAI). Intenta grabar de nuevo.")

    history.append({"role": "user", "content": user_text})

    try:
        turn = get_teacher_turn(history, persona_key=persona_key, memory_summary=memory_summary)
    except Exception:
        pronunciation_executor.shutdown(wait=False)
        history.pop()  # el profesor no pudo responder; no dejamos un turno de usuario colgado
        error = "⚠️ El profesor no pudo responder (problema de conexión con la API). Intenta de nuevo."
        return (
            history,
            render_conversation(history, persona_key),
            None,
            None,
            pronunciation_scores,
            session_id,
            memory_summary,
            session_start,
            duration_notice,
            duration_notice,
            error,
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

    # el None limpia mic_input para que quede listo para el siguiente turno
    return (
        history,
        render_conversation(history, persona_key),
        reply_audio_path,
        None,
        pronunciation_scores,
        session_id,
        memory_summary,
        session_start,
        duration_notice,
        duration_notice,
        error,
    )


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

    with gr.Blocks(title="ConversAItion") as demo:
        gr.HTML(HEADER_HTML)

        session_id_state = gr.State(None)
        history_state = gr.State([])  # historial de conversación, aislado por sesión/pestaña
        pronunciation_state = gr.State([])  # puntajes de Azure por turno del usuario
        memory_summary_state = gr.State(None)  # resumen de sesiones previas (Fase 8)
        session_start_state = gr.State(None)  # timestamp del primer turno (Fase 10)
        duration_notice_state = gr.State("")  # aviso de duración, una vez mostrado se mantiene
        persona_state = gr.State(DEFAULT_PERSONA)  # alimenta lo mismo que antes el dropdown

        duration_notice_output = gr.Markdown(elem_id="duration-notice")
        error_output = gr.Markdown(elem_id="error-notice")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column(elem_classes=["persona-panel"]):
                    persona_cards_html = gr.HTML(render_persona_cards(DEFAULT_PERSONA))
                    persona_triggers = {}
                    with gr.Row(visible=True):
                        for key in PERSONA_META:
                            persona_triggers[key] = gr.Button(
                                key, elem_id=f"trigger-{key}", elem_classes=["persona-trigger"]
                            )

                with gr.Column(elem_classes=["mic-section"]):
                    mic_input = gr.Audio(
                        sources=["microphone"], type="filepath", label="Habla aquí", elem_id="mic-recorder"
                    )
                    gr.HTML('<div class="mic-label">Habla aquí</div>')

                report_model_dropdown = gr.Dropdown(
                    choices=list(REPORT_MODELS.keys()),
                    value=DEFAULT_REPORT_MODEL,
                    label="Modelo para la retroalimentación final",
                    elem_classes=["model-select"],
                )

            with gr.Column(scale=3):
                with gr.Column(elem_classes=["chat-panel"]):
                    gr.HTML('<div class="chat-panel-title">Conversación</div>')
                    conversation = gr.HTML(render_conversation([], DEFAULT_PERSONA))
                teacher_audio = gr.Audio(label="Respuesta del profesor", autoplay=True, elem_id="teacher-audio-hidden")
                end_session_btn = gr.Button("Terminar sesión", elem_classes=["end-session-btn"])

        report_output = gr.Markdown(label="Reporte final")

        for key, trigger_btn in persona_triggers.items():
            trigger_btn.click(
                lambda k=key: (k, render_persona_cards(k)),
                outputs=[persona_state, persona_cards_html],
            )

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
                persona_state,
            ],
            outputs=[
                history_state,
                conversation,
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
    build_app().launch(css_paths=[ASSETS_DIR / "styles.css"])
