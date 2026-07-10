"""
DeepEval evaluation pipeline — 10 metrics, low cost, low latency.

Metrics:
  RAG quality  : Context Precision, Context Recall, Context Relevancy,
                 Faithfulness, Answer Relevancy, Correctness (GEval)
  Performance  : Latency (total), Retrieval Time, Generation Time
  Cost         : Token Usage (prompt + completion tokens per test case)

Design decisions:
  - PDF is ingested ONCE and reused across all test cases (not once per case).
  - All LLM-judged metrics use OpenAI (native DeepEval support — no custom wrapper).
  - Correctness uses GEval (flexible) instead of exact-match.
  - Latency / Retrieval Time / Generation Time are non-LLM metrics (free).
  - Token Usage is captured from the RAG graph run and stored per case.
  - max_cases parameter allows running just 1 test case for quick smoke-tests.

Event-loop isolation:
  DeepEval's evaluate() calls loop.run_until_complete() internally and installs
  a SIGINT handler (sys.exit). Running it inside asyncio.to_thread() (FastAPI)
  triggers asyncio loop conflicts and kills uvicorn on hot-reload. The API
  endpoint therefore spawns evaluate.py as a subprocess so it gets its own
  clean interpreter with no existing event loop.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Reconfigure stdout/stderr to UTF-8 and tell Rich to use plain text rendering.
# On Windows, the default console encoding is cp1252 which can't encode emoji
# (e.g. the ✨ DeepEval prints). Rich also detects Windows and forces its legacy
# Win32 console renderer which bypasses Python's encoding layer entirely.
# Setting TERM=dumb makes Rich fall back to the plain ANSI path which respects
# the stream encoding. Reconfiguring stdout/stderr ensures our print() calls also work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage

from backend.config import (
    EVAL_CONCURRENCY,
    EVAL_MODEL,
    EVAL_RESULTS_FILE,
    EVAL_THRESHOLD,
    EVAL_THRESHOLDS,
    EVAL_THROTTLE,
    GOLDENS_FILE,
    REPORTS_DIR,
)
from backend.rag.document_pipeline import ingest_pdf
from backend.rag.rag_graph import build_rag_subgraph


# ── Non-LLM metrics (free, just read from test case metadata) ────

class MeasuredValueMetric(BaseMetric):
    """
    Reads a pre-measured value from an attribute stashed on the test case by
    _run_query (timings, token counts) and passes if it is <= threshold.
    One parametrized class replaces the four former copy-pasted metric classes;
    metric names are kept identical so reports stay comparable with old baselines.
    """

    def __init__(self, name: str, attr: str, threshold: float, unit: str = "s"):
        self.name             = name
        self.attr             = attr
        self.threshold        = threshold
        self.unit             = unit          # "s" → seconds, "tokens" → integer count
        self.score            = 0.0
        self.success          = False
        self.reason           = ""
        self.evaluation_model = None
        self.strict_mode      = False
        self.async_mode       = False
        self.verbose_mode     = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        value = getattr(test_case, self.attr, None)
        if value is None:
            self.score = 0.0
            self.success = False
            self.reason = f"No data for {self.name}"
            return self.score
        self.score   = round(float(value), 3)
        self.success = value <= self.threshold
        if self.unit == "tokens":
            self.reason = f"{int(value)} tokens (threshold {int(self.threshold)})"
        else:
            self.reason = f"{value:.2f}s (threshold {self.threshold}s)"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# ── Goldens ──────────────────────────────────────────────────────

def load_goldens(path: str = GOLDENS_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── RAG runner with timing + token capture ────────────────────────

def _run_query(graph, query: str, session_id: str) -> dict:
    """
    Run a query through the compiled RAG subgraph.
    Returns answer, context chunks, and timing/token metadata.
    """
    from backend.rag.rag_graph import set_step_callback

    _token_estimate: list[int] = [0]
    _gen_start: list[float]    = [0.0]
    _gen_end:   list[float]    = [0.0]

    def _on_step(step) -> None:
        # Steps are plain strings or badge dicts (see rag_graph module docstring).
        is_llm_badge = isinstance(step, dict) and step.get("badge") == "llm"
        if is_llm_badge:
            _token_estimate[0] += int(step.get("in", 0) or 0) + int(step.get("out", 0) or 0)
        if isinstance(step, str) and "Phase 1: answering" in step:
            _gen_start[0] = time.perf_counter()
        if is_llm_badge and _gen_start[0] > 0 and _gen_end[0] == 0:
            _gen_end[0] = time.perf_counter()

    set_step_callback(_on_step, session_id)
    t_total_start = time.perf_counter()

    try:
        final = graph.invoke({
            "messages":       [HumanMessage(content=query)],
            "session_id":     session_id,
            "query":          query,
            "retrieved_docs": [],
            "rewrite_count":  0,
            "is_relevant":    None,
            "answer":         None,
        })
    finally:
        set_step_callback(None, session_id)

    t_total = time.perf_counter() - t_total_start
    gen_s   = (_gen_end[0] - _gen_start[0]) if _gen_end[0] > 0 else 0.0
    retr_s  = max(0.0, t_total - gen_s)

    answer  = final.get("answer") or ""
    # Exclude document_summary chunks — the RAG graph already filters them before
    # passing context to the LLM (generate_answer_node line ~391). Including them
    # in retrieval_context passed to DeepEval would poison Contextual Relevancy
    # (summary bullets about unrelated leave types dilute the ratio) and Contextual
    # Recall (judge can't attribute specific expected sentences to a broad summary blob).
    context = [
        d.page_content
        for d in (final.get("retrieved_docs") or [])
        if d.metadata.get("type") != "document_summary"
    ]

    return {
        "answer":       answer,
        "context":      context,
        "total_s":      round(t_total, 3),
        "retrieval_s":  round(retr_s, 3),
        "generation_s": round(gen_s, 3),
        "total_tokens": _token_estimate[0],
    }


# ── Main evaluation ───────────────────────────────────────────────

def run_evaluation(
    pdf_path: str,
    goldens_file: str = GOLDENS_FILE,
    max_cases: Optional[int] = None,
) -> dict:
    """
    Run the full evaluation pipeline on a PDF.

    Args:
        pdf_path:     Path to the PDF to evaluate.
        goldens_file: Path to the goldens JSON file.
        max_cases:    Limit to first N test cases (e.g. 1 for a quick smoke-test).
    """
    pairs = load_goldens(goldens_file)
    if max_cases:
        pairs = pairs[:max_cases]

    # Ingest once — reuse the same session_id for all test cases.
    shared_session_id = f"eval_{uuid4().hex}"
    ingest_pdf(pdf_path, shared_session_id)

    graph = build_rag_subgraph().compile()

    # OpenAI eval LLM — native DeepEval support, correct schema handling
    eval_llm = GPTModel(model=EVAL_MODEL)

    t = EVAL_THRESHOLDS
    llm_metrics = [
        ContextualPrecisionMetric( threshold=t["contextual_precision"],  model=eval_llm),
        ContextualRecallMetric(    threshold=t["contextual_recall"],     model=eval_llm),
        ContextualRelevancyMetric( threshold=t["contextual_relevancy"],  model=eval_llm),
        FaithfulnessMetric(        threshold=t["faithfulness"],          model=eval_llm),
        AnswerRelevancyMetric(     threshold=t["answer_relevancy"],      model=eval_llm),
        GEval(
            name="Correctness",
            criteria=(
                "Judge whether the actual output is factually correct compared to the "
                "expected output. Award high scores for semantically equivalent answers "
                "even if wording differs. Penalise hallucinations or contradictions."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=t["correctness"],
            model=eval_llm,
        ),
    ]

    perf_metrics = [
        MeasuredValueMetric("Latency (s)",         "_total_s",      threshold=60.0),
        MeasuredValueMetric("Retrieval Time (s)",  "_retrieval_s",  threshold=5.0),
        MeasuredValueMetric("Generation Time (s)", "_generation_s", threshold=20.0),
        MeasuredValueMetric("Token Usage",         "_total_tokens", threshold=5000, unit="tokens"),
    ]

    test_cases: list[LLMTestCase] = []
    for pair in pairs:
        run = _run_query(graph, pair["input"], shared_session_id)
        tc = LLMTestCase(
            input=pair["input"],
            actual_output=run["answer"],
            expected_output=pair["expected_output"],
            retrieval_context=run["context"],
        )
        tc._total_s      = run["total_s"]
        tc._retrieval_s  = run["retrieval_s"]
        tc._generation_s = run["generation_s"]
        tc._total_tokens = run["total_tokens"]
        test_cases.append(tc)

    results = evaluate(
        test_cases,
        llm_metrics,
        async_config=AsyncConfig(
            max_concurrent=EVAL_CONCURRENCY,
            throttle_value=EVAL_THROTTLE,
        ),
    )

    for tc in test_cases:
        for m in perf_metrics:
            m.measure(tc)

    summary = []
    for i, r in enumerate(results.test_results):
        tc = test_cases[i]
        perf_scores = [
            {"name": m.name, "score": m.score, "passed": m.is_successful(), "reason": m.reason}
            for m in perf_metrics
        ]
        summary.append({
            "input":           r.input,
            "actual_output":   r.actual_output,
            "expected_output": r.expected_output,
            "success":         r.success,
            "timing": {
                "total_s":      tc._total_s,
                "retrieval_s":  tc._retrieval_s,
                "generation_s": tc._generation_s,
                "total_tokens": tc._total_tokens,
            },
            "metrics": [
                {"name": m.name, "score": m.score, "passed": m.success, "reason": m.reason}
                for m in r.metrics_data
            ] + perf_scores,
        })

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    stem = Path(goldens_file).stem
    timestamped_path = str(Path(REPORTS_DIR) / f"{timestamp}_{stem}.json")
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    Path(timestamped_path).write_text(payload, encoding="utf-8")
    Path(EVAL_RESULTS_FILE).write_text(payload, encoding="utf-8")

    return {
        "results_file": timestamped_path,
        "latest_file":  EVAL_RESULTS_FILE,
        "test_count":   len(summary),
        "results":      summary,
    }


if __name__ == "__main__":
    pdf      = sys.argv[1] if len(sys.argv) > 1 else "documents/sample.pdf"
    goldens  = sys.argv[2] if len(sys.argv) > 2 else GOLDENS_FILE
    n        = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out = run_evaluation(pdf, goldens_file=goldens, max_cases=n)
    print(f"Saved {out['test_count']} results → {out['results_file']}")
    # Write JSON to stdout so the subprocess caller (api.py) can read it
    print("__RESULT__" + json.dumps(out, ensure_ascii=False))
