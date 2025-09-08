"""
src/asr_utils.py
----------------
Convenience helpers for running ASR on files (whisper-based), optional chunking,
and simple CLI for quick experiments.

Dependencies:
- whisper (openai-whisper)
- ffmpeg installed on system (for chunking)
- torch (for device detection)

Usage (example):
    python src/asr_utils.py --input data/raw_audio/sample.wav --model small --out tmp/raw.txt
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import whisper
import torch


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ffmpeg_split(audio_path: str, chunk_length_s: int, out_dir: str) -> List[str]:
    """
    Split an audio file into chunks using ffmpeg segmenter.
    Returns list of chunk file paths.
    Requires ffmpeg on PATH.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "chunk_%04d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_length_s),
        "-c",
        "pcm_s16le",
        "-ar",
        "16000",
        pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = sorted([str(p) for p in out_dir.glob("chunk_*.wav")])
    return chunks


class ASRUtils:
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """
        Load whisper model.
        model_size: tiny, base, small, medium, large
        device: optional override (e.g., 'cpu' or 'cuda')
        """
        self.model_size = model_size
        self.device = device or _get_device()
        print(f"[INFO] Loading whisper model '{model_size}' on device '{self.device}' ...")
        # whisper.load_model will attempt to use GPU if available; device argument isn't passed
        # to load_model in some whisper versions, so just load and rely on torch.
        self.model = whisper.load_model(model_size)
        print("[INFO] Model loaded.")

    def transcribe_file(
        self,
        audio_path: str,
        language: str = "en",
        chunk_length_s: Optional[int] = None,
        return_segments: bool = False,
        temp_dir: Optional[str] = None,
        **whisper_kwargs,
    ) -> Any:
        """
        Transcribe an audio file. If chunk_length_s is provided, file will be split into
        pieces and transcribed chunk-by-chunk, which can help with long inputs / memory.
        whisper_kwargs passed to model.transcribe (e.g., temperature, beam_size).
        If return_segments=True, returns a list of segments (with start/end/text).
        Otherwise returns concatenated text string.
        """
        audio_path = str(audio_path)
        if chunk_length_s:
            temp_dir = temp_dir or tempfile.mkdtemp(prefix="asr_chunks_")
            chunks = _ffmpeg_split(audio_path, chunk_length_s, temp_dir)
            texts = []
            all_segments = []
            for c in chunks:
                result = self.model.transcribe(c, language=language, **whisper_kwargs)
                texts.append(result.get("text", "").strip())
                if return_segments:
                    segs = result.get("segments", [])
                    # adjust segment times by chunk offset (not implemented here for simplicity)
                    all_segments.extend(segs)
            full_text = "\n".join(texts).strip()
            return all_segments if return_segments else full_text
        else:
            result = self.model.transcribe(audio_path, language=language, **whisper_kwargs)
            if return_segments:
                return result.get("segments", [])
            return result.get("text", "").strip()

    @staticmethod
    def save_transcript(text: str, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ASR utils (Whisper wrapper)")
    p.add_argument("--input", required=True, help="input audio path")
    p.add_argument("--model", default="small", help="whisper model size")
    p.add_argument("--chunk", type=int, default=None, help="chunk length in seconds (optional)")
    p.add_argument("--out", default="tmp/raw.txt", help="output transcript path")
    p.add_argument("--language", default="en", help="language code")
    args = p.parse_args()

    utils = ASRUtils(model_size=args.model)
    txt = utils.transcribe_file(args.input, language=args.language, chunk_length_s=args.chunk)
    utils.save_transcript(txt, args.out)
    print(f"[DONE] saved transcript -> {args.out}")
