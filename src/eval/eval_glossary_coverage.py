#Check glossary term recall in system outputs (how many glossary/domain terms are preserved vs. missed).
"""
eval_glossary_coverage.py
-------------------------
Compute glossary coverage in ASR outputs compared to references.

Metrics:
 - Glossary Recall: fraction of glossary terms in refs also appearing in hyps.
 - Precision: fraction of glossary terms in hyps that were in refs.
 - F1 score.
 - File-level glossary hits/misses.

Usage:
    python src/eval/eval_glossary_coverage.py --refs data/refs --hyps data/system_outputs --glossary data/glossary.txt --out reports/glossary_eval.json
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_texts(path: str) -> dict:
    texts = {}
    for f in Path(path).glob("*.txt"):
        texts[f.stem] = f.read_text(encoding="utf-8").lower()
    return texts


def load_glossary(path: str):
    return [l.strip().lower() for l in open(path, encoding="utf-8") if l.strip()]


def evaluate_glossary(refs: dict, hyps: dict, glossary: list):
    results = {"files": {}, "aggregate": {}}
    total_tp = total_fp = total_fn = 0

    for fid, ref_text in refs.items():
        hyp_text = hyps.get(fid, "").lower()
        ref_terms = {t for t in glossary if t in ref_text}
        hyp_terms = {t for t in glossary if t in hyp_text}

        tp = len(ref_terms & hyp_terms)
        fp = len(hyp_terms - ref_terms)
        fn = len(ref_terms - hyp_terms)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results["files"][fid] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # aggregate
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    results["aggregate"] = {"precision": prec, "recall": rec, "f1": f1}

    return results


def main(refs_dir, hyps_dir, glossary_path, out_path):
    refs = load_texts(refs_dir)
    hyps = load_texts(hyps_dir)
    glossary = load_glossary(glossary_path)

    results = evaluate_glossary(refs, hyps, glossary)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved glossary eval -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refs", required=True)
    p.add_argument("--hyps", required=True)
    p.add_argument("--glossary", required=True)
    p.add_argument("--out", default="reports/glossary_eval.json")
    args = p.parse_args()
    main(args.refs, args.hyps, args.glossary, args.out)
