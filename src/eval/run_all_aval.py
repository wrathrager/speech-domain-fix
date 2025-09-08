"""
run_all_eval.py
---------------
Driver script to evaluate Quick / Medium / Long fix pipelines.

Runs:
  - WER + CER
  - Glossary precision/recall/F1

Outputs:
  - JSON with all results
  - Markdown summary table for easy viewing

Usage:
    python src/eval/run_all_eval.py \
        --refs data/refs \
        --glossary data/glossary.txt \
        --out reports/eval_summary.json
"""

import os
import json
import argparse
from pathlib import Path
import subprocess


def run_eval_wer(refs, hyps, out_file):
    subprocess.run([
        "python", "src/eval/eval_wer.py",
        "--refs", refs,
        "--hyps", hyps,
        "--out", out_file
    ], check=True)


def run_eval_glossary(refs, hyps, glossary, out_file):
    subprocess.run([
        "python", "src/eval/eval_glossary_coverage.py",
        "--refs", refs,
        "--hyps", hyps,
        "--glossary", glossary,
        "--out", out_file
    ], check=True)


def collect_results(paths):
    out = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
        else:
            out[key] = None
    return out


def make_summary_table(results):
    """
    Creates a Markdown table comparing Quick / Medium / Long fix.
    """
    header = "| Method | WER | CER | Glossary F1 | Glossary Recall | Glossary Precision |\n"
    header += "|--------|-----|-----|-------------|-----------------|--------------------|\n"

    rows = []
    for method, res in results.items():
        if not res: 
            rows.append(f"| {method} | N/A | N/A | N/A | N/A | N/A |")
            continue

        wer_mean = res["wer"]["aggregate"].get("wer_mean", 0)
        cer_mean = res["wer"]["aggregate"].get("cer_mean", 0)
        gloss = res["glossary"]["aggregate"]
        f1 = gloss.get("f1", 0)
        rec = gloss.get("recall", 0)
        prec = gloss.get("precision", 0)

        rows.append(
            f"| {method} | {wer_mean:.3f} | {cer_mean:.3f} | {f1:.3f} | {rec:.3f} | {prec:.3f} |"
        )
    return header + "\n".join(rows)


def main(refs, glossary, out_json, out_md):
    methods = {
        "Quick Fix": "outputs/quick_fix",
        "Medium Fix": "outputs/medium_fix",
        "Long Fix": "outputs/long_fix",
    }

    os.makedirs("reports", exist_ok=True)

    results = {}
    for name, hyp_dir in methods.items():
        wer_out = f"reports/{name.replace(' ', '_').lower()}_wer.json"
        gloss_out = f"reports/{name.replace(' ', '_').lower()}_glossary.json"

        print(f"[INFO] Evaluating {name}...")
        run_eval_wer(refs, hyp_dir, wer_out)
        run_eval_glossary(refs, hyp_dir, glossary, gloss_out)

        results[name] = {
            "wer": json.load(open(wer_out, "r", encoding="utf-8")),
            "glossary": json.load(open(gloss_out, "r", encoding="utf-8")),
        }

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown table
    md = make_summary_table(results)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[DONE] Full eval saved: {out_json}, {out_md}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refs", required=True, help="Directory of reference transcripts (.txt)")
    p.add_argument("--glossary", required=True, help="Path to glossary.txt")
    p.add_argument("--out", default="reports/eval_summary.json")
    p.add_argument("--out_md", default="reports/eval_summary.md")
    args = p.parse_args()

    main(args.refs, args.glossary, args.out, args.out_md)
