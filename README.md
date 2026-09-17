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

Ver `prompt_maestro_app_speaking_ingles.md` (carpeta padre) para la especificación completa del proyecto.
