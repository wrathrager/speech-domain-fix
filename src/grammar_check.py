"""
src/grammar_check.py
--------------------
Lightweight rule-based / glossary-aware corrections for ASR transcripts.
Not intended to replace LLM corrections, but provides a deterministic baseline
that is fast and offline.

Usage:
    from grammar_check import GrammarChecker
    checker = GrammarChecker(glossary_path="data/glossary.txt")
    fixed = checker.correct_text(raw_transcript)
"""

import re
import difflib
from pathlib import Path
from typing import Optional, Dict, List


DEFAULT_MISSPELLINGS: Dict[str, str] = {
    # small example map; expand this as you discover common ASR mistakes
    "maniness": "moneyness",
    "modernism": "moneyness",
    "volatillity": "volatility",
    "volitility": "volatility",
}


class GrammarChecker:
    def __init__(self, glossary_path: Optional[str] = None, misspellings: Optional[Dict[str, str]] = None):
        """
        Load glossary words and misspelling map.
        glossary_path: file with one domain word per line.
        misspellings: mapping of common incorrect -> correct spellings.
        """
        self.misspellings = DEFAULT_MISSPELLINGS.copy()
        if misspellings:
            self.misspellings.update(misspellings)

        self.glossary = []
        if glossary_path:
            p = Path(glossary_path)
            if p.exists():
                self.glossary = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            # store lowercase for matching
            self.glossary_lower = [g.lower() for g in self.glossary]
        else:
            self.glossary_lower = []

    def _replace_token(self, token: str) -> str:
        low = token.lower()
        # 1) direct misspelling mapping
        if low in self.misspellings:
            corrected = self.misspellings[low]
            return self._preserve_case(token, corrected)

        # 2) if close to a glossary word -> replace
        if self.glossary_lower:
            matches = difflib.get_close_matches(low, self.glossary_lower, n=1, cutoff=0.8)
            if matches:
                idx = self.glossary_lower.index(matches[0])
                corrected = self.glossary[idx]
                return self._preserve_case(token, corrected)

        return token

    @staticmethod
    def _preserve_case(orig: str, corrected: str) -> str:
        if orig.isupper():
            return corrected.upper()
        if orig[0].isupper():
            return corrected.capitalize()
        return corrected

    def correct_text(self, text: str, do_basic_punct: bool = True) -> str:
        """
        Correct tokens using misspelling map and glossary fuzzy matching.
        Also applies some light punctuation normalization.
        """
        # tokenization that preserves punctuation roughly
        def repl(match):
            tok = match.group(0)
            return self._replace_token(tok)

        corrected = re.sub(r"[A-Za-z0-9_]+", repl, text)

        # basic punctuation/whitespace fixes
        if do_basic_punct:
            # collapse multiple spaces
            corrected = re.sub(r"\s+", " ", corrected).strip()
            # fix space before punctuation
            corrected = re.sub(r"\s+([,.;:!?])", r"\1", corrected)
            # ensure single space after punctuation
            corrected = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", corrected)
            # naive sentence capitalization
            corrected = re.sub(r"(^|[\.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), corrected)

        return corrected


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="raw transcript txt")
    p.add_argument("--glossary", default=None, help="glossary file (one term per line)")
    p.add_argument("--out", default="tmp/grammar_fixed.txt", help="output path")
    args = p.parse_args()

    txt = Path(args.input).read_text(encoding="utf-8")
    checker = GrammarChecker(glossary_path=args.glossary)
    fixed = checker.correct_text(txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(fixed, encoding="utf-8")
    print(f"[DONE] saved grammar-fixed transcript -> {args.out}")
