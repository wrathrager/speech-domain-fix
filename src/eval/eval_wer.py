#Compute Word Error Rate (WER) and Character Error Rate (CER) between system transcripts and reference transcripts.
"""
eval_wer.py
-----------
Compute WER and CER for ASR outputs against ground truth transcripts.

Inputs:
 - refs_dir/: directory of reference transcripts (.txt, one per audio)
 - hyps_dir/: directory of system outputs (.txt, same filenames as refs)
Outputs:
 - JSON report with WER, CER, and file-level details.

Usage:
    python src/eval/eval_wer.py --refs data/refs --hyps data/system_outputs --out reports/wer_report.json
"""

import os
import json
import argparse
from pathlib import Path
from jiwer import wer, cer


def load_texts(path: str) -> dict:
    data = {}
    p = Path(path)
    for f in p.glob("*.txt"):
        text = f.read_text(encoding="utf-8").strip()
        data[f.stem] = text
    return data


def main(refs_dir: str, hyps_dir: str, out_path: str):
    refs = load_texts(refs_dir)
    hyps = load_texts(hyps_dir)

    results = {"files": {}, "aggregate": {}}
    wers, cers = [], []

    for fid, ref_text in refs.items():
        hyp_text = hyps.get(fid, "")
        w = wer(ref_text, hyp_text)
        c = cer(ref_text, hyp_text)
        wers.append(w)
        cers.append(c)
        results["files"][fid] = {"wer": w, "cer": c, "ref": ref_text, "hyp": hyp_text}

    results["aggregate"]["wer_mean"] = sum(wers) / len(wers) if wers else None
    results["aggregate"]["cer_mean"] = sum(cers) / len(cers) if cers else None
    results["aggregate"]["n_files"] = len(refs)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved WER report -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refs", required=True, help="directory of reference txt files")
    p.add_argument("--hyps", required=True, help="directory of hypothesis txt files")
    p.add_argument("--out", default="reports/wer_report.json")
    args = p.parse_args()
    main(args.refs, args.hyps, args.out)
