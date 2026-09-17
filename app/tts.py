"""Síntesis de voz con Azure TTS (voz neural en-US)."""

import azure.cognitiveservices.speech as speechsdk

import config

DEFAULT_VOICE = "en-US-AriaNeural"


def synthesize_speech(text: str, output_path: str, voice: str = DEFAULT_VOICE) -> str:
    """Sintetiza `text` a voz y lo guarda como WAV en `output_path`."""
    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return output_path

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure TTS cancelado: {details.reason} - {details.error_details}")

    raise RuntimeError(f"Azure TTS falló: {result.reason}")
