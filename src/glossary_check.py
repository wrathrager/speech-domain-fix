"""
glossary_check.py
-----------------
Utilities to analyze transcripts vs glossary:
 - compute frequency of candidate domain words in a corpus of transcripts
 - find terms present in transcripts but missing from glossary
 - fuzzy-match candidate tokens to glossary to suggest additions

Usage:
    python src/glossary_check.py --transcripts_dir data/transcripts --glossary data/glossary.txt --out reports/glossary_report.json
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Set, Dict
import re
import difflib

TOKEN_RE = re.compile(r"[A-Za-z\-]+")  # simple tokenization for domain words


def load_glossary(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def gather_transcript_tokens(transcripts_dir: str) -> Counter:
    c = Counter()
    p = Path(transcripts_dir)
    for f in p.glob("*"):
        if f.is_file() and f.suffix.lower() in [".txt", ".vtt", ".srt", ".json"]:
            text = f.read_text(encoding="utf-8", errors="ignore")
            tokens = TOKEN_RE.findall(text)
            tokens = [t.lower() for t in tokens if len(t) > 2]
            c.update(tokens)
    return c


def find_missing_terms(counter: Counter, glossary: List[str], top_k: int = 200, cutoff: float = 0.8) -> Dict:
    glossary_lower = [g.lower() for g in glossary]
    common = counter.most_common(top_k)
    suggestions = []
    for tok, freq in common:
        if tok in glossary_lower:
            continue
        # fuzzy compare to glossary (if close, skip); else suggest
        close = difflib.get_close_matches(tok, glossary_lower, n=1, cutoff=cutoff)
        if close:
            continue
        suggestions.append({"term": tok, "freq": freq})
    return {"suggestions": suggestions, "total_unique_tokens": len(counter)}


def fuzzy_map_to_glossary(tokens: List[str], glossary: List[str], cutoff: float = 0.7) -> Dict[str, str]:
    out = {}
    glossary_lower = [g.lower() for g in glossary]
    for t in tokens:
        m = difflib.get_close_matches(t.lower(), glossary_lower, n=1, cutoff=cutoff)
        out[t] = m[0] if m else None
    return out


def main(transcripts_dir: str, glossary_path: str, out_path: str):
    glossary = load_glossary(glossary_path)
    token_counts = gather_transcript_tokens(transcripts_dir)
    missing = find_missing_terms(token_counts, glossary)
    report = {
        "glossary_count": len(glossary),
        "top_tokens": token_counts.most_common(100),
        "missing_suggestions": missing["suggestions"],
        "unique_tokens": missing["total_unique_tokens"],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] saved glossary report -> {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--transcripts_dir", required=True)
    p.add_argument("--glossary", required=True)
    p.add_argument("--out", default="tmp/glossary_report.json")
    args = p.parse_args()
    main(args.transcripts_dir, args.glossary, args.out)
