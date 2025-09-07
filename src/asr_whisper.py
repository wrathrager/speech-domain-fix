# Whisper wrapper
"""
asr_whisper.py
---------------
Wrapper around OpenAI Whisper for transcribing audio files.
Supports basic transcription + language selection.
"""

import whisper
import os

class WhisperASR:
    def __init__(self, model_size: str = "small"):
        """
        Initialize Whisper model.
        Args:
            model_size (str): one of ["tiny", "base", "small", "medium", "large"]
        """
        self.model_size = model_size
        print(f"[INFO] Loading Whisper model: {model_size} ...")
        self.model = whisper.load_model(model_size)
        print("[INFO] Whisper model loaded successfully!")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file into text.
        Args:
            audio_path (str): path to audio file (.wav, .mp3, .m4a, etc.)
            language (str): language code (default = English)
        Returns:
            transcript (str)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[INFO] Transcribing {audio_path} ...")
        result = self.model.transcribe(audio_path, language=language)
        transcript = result["text"].strip()

        print("[INFO] Transcription complete!")
        return transcript


if __name__ == "__main__":
    # Example usage
    asr = WhisperASR(model_size="small")
    text = asr.transcribe("data/raw_audio/sample.wav")
    print("Transcript:\n", text)
