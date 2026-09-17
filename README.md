# Speaking App

App local para practicar speaking en inglés con un "profesor" de IA por voz: corrige pronunciación y gramática, y da un reporte al final de cada sesión.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # completar con tus API keys
```

## Ejecutar

```bash
python -m app.ui
```

Abre `http://127.0.0.1:7860` en el navegador.

> **Nota:** el componente de audio de Gradio crashea la pestaña en Brave (probado en este proyecto). Usa Chromium/Chrome o Firefox mientras no se investigue más a fondo.

Ver `prompt_maestro_app_speaking_ingles.md` (carpeta padre) para la especificación completa del proyecto.
