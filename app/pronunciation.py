"""Azure Pronunciation Assessment en modo unscripted (sin texto de
referencia fijo, porque la conversación es libre).

La evaluación tarda entre 9 y 15 segundos y su resultado no se usa hasta el
reporte final, así que no corre dentro del turno: se encola con
`submit_assessment()` y los puntajes se recogen una sola vez al cerrar la
sesión con `collect_scores()`.
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import azure.cognitiveservices.speech as speechsdk

import config

# Varios workers para que una sesión de turnos cortos y seguidos no acumule
# cola: cada evaluación dura más que el turno que la encoló.
_MAX_WORKERS = 4

# Margen al recoger: a esa altura las evaluaciones llevan rato corriendo, el
# timeout solo evita que una llamada colgada se lleve el reporte con ella.
_COLLECT_TIMEOUT_SECONDS = 30

_executor: ThreadPoolExecutor | None = None
_pending: dict[int, list[Future]] = {}
_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="pronunciation"
        )
    return _executor


def submit_assessment(session_id: int, audio_path: str) -> Future:
    """Encola la evaluación del turno y devuelve enseguida, sin esperarla.

    El future también queda guardado por sesión para `collect_scores()`; se
    devuelve solo para poder descartarlo con `drop_assessment()` si el turno
    termina fallando.
    """
    future = _get_executor().submit(assess_pronunciation, audio_path)
    with _lock:
        _pending.setdefault(session_id, []).append(future)
    return future


def drop_assessment(session_id: int, future: Future) -> None:
    """Descarta la evaluación de un turno que no llegó a completarse.

    Si ya arrancó no se puede cancelar, pero igual sale de la lista de la
    sesión: un turno que no quedó en el historial no debe puntuar en el reporte.
    """
    future.cancel()
    with _lock:
        pending = _pending.get(session_id)
        if pending is not None and future in pending:
            pending.remove(future)


def collect_scores(session_id: int | None) -> list[dict]:
    """Espera las evaluaciones encoladas de la sesión y devuelve sus puntajes.

    Se llama una sola vez, al cerrar la sesión. El orden es el de los turnos,
    que es lo que el reporte usa para la tendencia (primero vs. último). Un
    turno que Azure no pudo evaluar (silencio/ruido) o que falló se omite: no
    vale perder el reporte completo por un turno sin puntaje.

    Nota: una sesión que el alumno abandona sin cerrar deja su entrada en
    `_pending` hasta que termina el proceso. Es intencional por simplicidad —
    son unos pocos futures ya resueltos en una app local.
    """
    with _lock:
        futures = _pending.pop(session_id, [])

    scores = []
    for future in futures:
        try:
            scores.append(future.result(timeout=_COLLECT_TIMEOUT_SECONDS))
        except Exception:
            continue
    return scores


def assess_pronunciation(audio_path: str) -> dict:
    """Evalúa un turno de audio del usuario. Devuelve accuracy, fluency,
    completeness, prosody y el puntaje global de pronunciación (0-100).
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=False,
    )
    pronunciation_config.enable_prosody_assessment()

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        details = getattr(result, "cancellation_details", None)
        raise RuntimeError(f"Azure no pudo evaluar el audio: {result.reason} {details}")

    assessment = speechsdk.PronunciationAssessmentResult(result)
    return {
        "recognized_text": result.text,
        "accuracy": assessment.accuracy_score,
        "fluency": assessment.fluency_score,
        "completeness": assessment.completeness_score,
        "prosody": assessment.prosody_score,
        "pronunciation": assessment.pronunciation_score,
    }
