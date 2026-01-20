# -*- coding: utf-8 -*-
"""Test Google Cloud TTS integration."""
import os
import pytest
import tempfile

try:
    from google.cloud import texttospeech
except ImportError:
    pytest.skip("google-cloud-texttospeech not installed", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_google_tts_synthesize():
    key_file = os.getenv("GOOGLE_SERVICE_KEY_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_file or not os.path.exists(key_file):
        pytest.skip("No Google service key file configured")

    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text="Hello, this is a Google TTS test.")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    assert response.audio_content
    # Optionally write to temp WAV file
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as f:
        f.write(response.audio_content)
        f.flush()
        assert os.path.getsize(f.name) > 0
