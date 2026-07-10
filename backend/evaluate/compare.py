"""
compare.py — diff two evaluation result JSONs and flag regressions.

Usage:
    python -m backend.evaluate.compare <baseline.json> <latest.json>

Exit code:
    0  — no regressions
    1  — one or more metrics regressed beyond threshold
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REGRESSION_THRESHOLD = 0.05   # score drop >= this is flagged as a regression
SCORE_FMT = ".3f"


def _metric_map(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Returns { question_input: { metric_name: score } } for each test case.
    """
    out = {}
    for case in results:
        key = case["input"]
        out[key] = {m["name"]: m["score"] for m in case.get("metrics", [])}
    return out


def compare(baseline_path: str, latest_path: str) -> int:
    baseline_raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    latest_raw   = json.loads(Path(latest_path).read_text(encoding="utf-8"))

    baseline = _metric_map(baseline_raw)
    latest   = _metric_map(latest_raw)

    regressions: list[tuple[str, str, float, float]] = []
    improvements: list[tuple[str, str, float, float]] = []

    all_questions = sorted(set(baseline) | set(latest))

    col_q   = 50
    col_m   = 28
    col_b   = 8
    col_l   = 8
    col_d   = 9

    header = (
        f"{'Question':<{col_q}}  {'Metric':<{col_m}}  "
        f"{'Baseline':>{col_b}}  {'Latest':>{col_l}}  {'Delta':>{col_d}}"
    )
    print(header)
    print("-" * len(header))

    for q in all_questions:
        b_metrics = baseline.get(q, {})
        l_metrics = latest.get(q, {})
        all_metrics = sorted(set(b_metrics) | set(l_metrics))

        for metric in all_metrics:
            b_score = b_metrics.get(metric)
            l_score = l_metrics.get(metric)

            if b_score is None or l_score is None:
                continue

            delta = l_score - b_score
            q_display = (q[:col_q - 1] + "…") if len(q) > col_q else q
            m_display = (metric[:col_m - 1] + "…") if len(metric) > col_m else metric

            flag = ""
            if delta <= -REGRESSION_THRESHOLD:
                flag = "  ← REGRESSION"
                regressions.append((q, metric, b_score, l_score))
            elif delta >= REGRESSION_THRESHOLD:
                flag = "  ↑ improved"
                improvements.append((q, metric, b_score, l_score))

            print(
                f"{q_display:<{col_q}}  {m_display:<{col_m}}  "
                f"{b_score:>{col_b}{SCORE_FMT}}  {l_score:>{col_l}{SCORE_FMT}}  "
                f"{delta:>+{col_d}{SCORE_FMT}}{flag}"
            )

    print()
    print(f"Summary: {len(improvements)} improvement(s), {len(regressions)} regression(s)")

    if regressions:
        print("\nREGRESSIONS:")
        for q, m, b, l in regressions:
            print(f"  [{m}] {q[:80]}  {b:{SCORE_FMT}} → {l:{SCORE_FMT}}  (Δ {l-b:+.3f})")
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m backend.evaluate.compare <baseline.json> <latest.json>")
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
