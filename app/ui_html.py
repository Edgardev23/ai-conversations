"""Fragmentos de HTML para la capa visual de Gradio: header, tarjetas de
personalidad y panel de conversación. Generados en Python a partir de los
mismos datos que ya produce la lógica de negocio (app/teacher_llm.py,
historial de turnos) — no hay estado ni lógica nueva acá, solo presentación.
"""

import base64
import html
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ICONS_DIR = ASSETS_DIR / "icons"


@lru_cache(maxsize=None)
def _icon(name: str) -> str:
    """SVG inline (no <img src>, para no depender de rutas estáticas servidas)."""
    return (ICONS_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _image_data_uri(name: str) -> str:
    """PNG embebido como data URI (mismo motivo que _icon: nada de rutas estáticas)."""
    data = base64.b64encode((ICONS_DIR / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


PERSONA_META = {
    "amigo_casual": {"label": "Amigo casual", "icon": "face_amigo.png", "icon_sm": "face_amigo_sm.png"},
    "profesor_formal": {"label": "Profesor", "icon": "face_profesor.png", "icon_sm": "face_profesor_sm.png"},
    "coach_entrevistas": {
        "label": "Entrevistador ejecutivo",
        "icon": "face_entrevistador.png",
        "icon_sm": "face_entrevistador_sm.png",
    },
}

HEADER_HTML = f"""
<div class="app-header">
  <div class="app-title">Convers<span class="grad">AI</span>tion</div>
</div>
"""


def render_persona_cards(selected: str) -> str:
    cards = []
    for key, meta in PERSONA_META.items():
        is_selected = "selected" if key == selected else ""
        onclick = f"document.querySelector('#trigger-{key} button').click()"
        cards.append(f"""
        <div class="persona-card {is_selected}" onclick="{onclick}">
          <div class="icon-wrap"><img src="{_image_data_uri(meta["icon"])}" alt=""></div>
          <span class="card-label">{html.escape(meta["label"])}</span>
          <div class="check-badge">✓</div>
        </div>
        """)
    return f"""
    <div class="persona-panel-title">Personalidad del profesor</div>
    <div class="persona-cards">{''.join(cards)}</div>
    """


def _bubble_row(role: str, text: str, persona_avatar_html: str) -> str:
    safe_text = html.escape(text).replace("\n", "<br>")
    if role == "user":
        return f"""
        <div class="bubble-row user">
          <div class="bubble-avatar">🧑</div>
          <div class="bubble user">{safe_text}</div>
        </div>
        """
    return f"""
    <div class="bubble-row teacher">
      <div class="bubble-avatar">{persona_avatar_html}</div>
      <div class="bubble teacher">{safe_text}</div>
    </div>
    """


def _suggestion_card(suggestion: str) -> str:
    safe_text = html.escape(suggestion)
    return f"""
    <div class="suggestion-card">
      {_icon("lightbulb.svg")}
      <div class="suggestion-text">
        <span class="suggestion-label">Sugerencia</span>
        {safe_text}
      </div>
    </div>
    """


def render_conversation(history: list[dict], persona_key: str) -> str:
    if not history:
        return """
        <div class="chat-scroll">
          <div class="chat-empty">
            <span class="emoji">💬</span>
            Presiona el micrófono y empieza a hablar en inglés.
          </div>
        </div>
        """

    persona_avatar_html = f'<img src="{_image_data_uri(PERSONA_META[persona_key]["icon_sm"])}" alt="">'
    rows = []
    for turn in history:
        if turn["role"] == "user":
            rows.append(_bubble_row("user", turn["content"], persona_avatar_html))
            continue
        rows.append(_bubble_row("teacher", turn["content"], persona_avatar_html))
        suggestion = turn.get("suggestion")
        if suggestion:
            rows.append(_suggestion_card(suggestion))

    return f'<div class="chat-scroll" id="chat-scroll-anchor">{"".join(rows)}</div>'
