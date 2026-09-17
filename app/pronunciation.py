"""Azure Pronunciation Assessment en modo unscripted (sin texto de
referencia fijo, porque la conversación es libre).
"""

import azure.cognitiveservices.speech as speechsdk

import config


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
