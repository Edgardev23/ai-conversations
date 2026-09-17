"""Interfaz Gradio: une STT + LLM + TTS. Usa gr.State para aislar el
historial de conversación por sesión/pestaña.
"""

import tempfile

import gradio as gr

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


def handle_turn(audio_path: str | None, history: list[dict]):
    history = history or []

    if audio_path is None:
        return history, _format_history(history), None, None

    user_text = transcribe_audio(audio_path)
    history.append({"role": "user", "content": user_text})

    turn = get_teacher_turn(history, persona_key=DEFAULT_PERSONA)
    reply_text = turn["reply"]
    history.append({"role": "assistant", "content": reply_text, "suggestion": turn.get("suggestion")})

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        reply_audio_path = tmp.name
    synthesize_speech(reply_text, reply_audio_path)

    # el último None limpia el input de audio para que quede listo para el siguiente turno
    return history, _format_history(history), reply_audio_path, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Práctica de Speaking en Inglés") as demo:
        gr.Markdown("# Práctica de Speaking en Inglés")

        state = gr.State([])  # historial de conversación, aislado por sesión/pestaña

        conversation = gr.Markdown(label="Conversación")

        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Habla aquí")
            teacher_audio = gr.Audio(label="Respuesta del profesor", autoplay=True)

        mic_input.stop_recording(
            handle_turn,
            inputs=[mic_input, state],
            outputs=[state, conversation, teacher_audio, mic_input],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
