"""
asr_cloud.py
------------
Wrappers for cloud ASR providers with glossary/phrase-biasing:
 - Google Cloud Speech-to-Text (speechContexts)
 - Azure Cognitive Services Speech (PhraseList)

Requirements:
 - google-cloud-speech
 - azure-cognitiveservices-speech

Environment setup:
 - Google: set GOOGLE_APPLICATION_CREDENTIALS to the JSON key file.
 - Azure: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars.

Usage examples are in the __main__ block.
"""

import os
from typing import List, Optional
import json

# Google Cloud imports
try:
    from google.cloud import speech
    from google.cloud.speech import RecognitionConfig, RecognitionAudio, SpeechContext
except Exception:
    speech = None

# Azure imports
try:
    import azure.cognitiveservices.speech as azure_speech
except Exception:
    azure_speech = None


def transcribe_google(
    audio_uri: str,
    phrases: Optional[List[str]] = None,
    language_code: str = "en-US",
    sample_rate_hertz: int = 16000,
    encoding: RecognitionConfig.AudioEncoding = RecognitionConfig.AudioEncoding.LINEAR16,
    use_long_running: bool = True,
) -> str:
    """
    Transcribe audio using Google Cloud Speech-to-Text.
    audio_uri: local path or GCS URI ("gs://bucket/path.wav"). For local files, use client.long_running_recognize with content loaded.
    phrases: list of domain phrases to bias (glossary).
    Returns concatenated transcript string.
    """

    if speech is None:
        raise ImportError("google-cloud-speech is not installed. pip install google-cloud-speech")

    client = speech.SpeechClient()

    # If local file, load content
    if audio_uri.startswith("gs://"):
        audio = RecognitionAudio(uri=audio_uri)
    else:
        with open(audio_uri, "rb") as f:
            content = f.read()
        audio = RecognitionAudio(content=content)

    speech_contexts = [SpeechContext(phrases=phrases)] if phrases else None

    config = RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
        speech_contexts=speech_contexts,
        enable_automatic_punctuation=True,
        model="default",
    )

    if use_long_running:
        operation = client.long_running_recognize(config=config, audio=audio)
        result = operation.result(timeout=600)  # adjust timeout as needed
    else:
        result = client.recognize(config=config, audio=audio)

    transcripts = []
    for res in result.results:
        transcripts.append(res.alternatives[0].transcript)
    return " ".join(transcripts).strip()


def transcribe_azure(
    audio_path: str,
    phrases: Optional[List[str]] = None,
    language: str = "en-US",
    subscription_key: Optional[str] = None,
    region: Optional[str] = None,
) -> str:
    """
    Transcribe audio using Azure Cognitive Services Speech SDK with PhraseList (biasing).
    Returns concatenated transcript.
    """
    if azure_speech is None:
        raise ImportError("azure-cognitiveservices-speech is not installed. pip install azure-cognitiveservices-speech")

    subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY")
    region = region or os.getenv("AZURE_SPEECH_REGION")
    if not subscription_key or not region:
        raise ValueError("Azure subscription key / region not set in environment (AZURE_SPEECH_KEY/AZURE_SPEECH_REGION).")

    speech_config = azure_speech.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = language

    audio_input = azure_speech.AudioConfig(filename=audio_path)
    recognizer = azure_speech.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Create phrase list grammar and add phrases
    if phrases:
        phrase_list_grammar = azure_speech.PhraseListGrammar.from_recognizer(recognizer)
        for p in phrases:
            phrase_list_grammar.addPhrase(p)

    final_texts = []
    done = False

    def _cb(evt):
        # For long audio, handle partial/final results; this simple wrapper uses final results
        pass

    # Use continuous recognition for longer files
    done_flag = {"done": False}
    def stop_cb(evt):
        done_flag["done"] = True

    recognizer.recognized.connect(lambda evt: final_texts.append(evt.result.text) if evt.result.reason == azure_speech.ResultReason.RecognizedSpeech else None)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    recognizer.start_continuous_recognition()
    import time
    # Wait until completion (naive)
    while not done_flag["done"]:
        time.sleep(0.5)
    recognizer.stop_continuous_recognition()

    return " ".join([t for t in final_texts if t]).strip()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["google", "azure"], required=True)
    p.add_argument("--audio", required=True, help="path to local audio file or gs:// URI (google)")
    p.add_argument("--glossary", default=None, help="path to glossary txt (one term per line)")
    p.add_argument("--out", default="tmp/cloud_transcript.txt")
    args = p.parse_args()

    phrases = None
    if args.glossary:
        with open(args.glossary, "r", encoding="utf-8") as f:
            phrases = [l.strip() for l in f if l.strip()]

    if args.provider == "google":
        text = transcribe_google(args.audio, phrases=phrases)
    else:
        text = transcribe_azure(args.audio, phrases=phrases)

    Path_out_dir = os.path.dirname(args.out)
    if Path_out_dir:
        os.makedirs(Path_out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        fout.write(text)
    print(f"[DONE] saved -> {args.out}")
