"""Evaluation endpoint — runs the DeepEval pipeline in a subprocess."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

from fastapi import APIRouter, HTTPException

from backend.models import EvaluateRequest

router = APIRouter()


@router.post("/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    from backend.config import GOLDENS_FILE

    if not os.path.exists(req.pdf_path):
        raise HTTPException(400, f"PDF not found: {req.pdf_path}")
    goldens = req.goldens_file or GOLDENS_FILE
    if not os.path.exists(goldens):
        raise HTTPException(400, f"Goldens file not found: {goldens}")

    # Run in a subprocess — DeepEval's evaluate() calls loop.run_until_complete()
    # and installs a SIGINT handler (sys.exit). Running it inside asyncio.to_thread
    # causes event-loop conflicts and kills uvicorn. A subprocess gets its own clean
    # Python interpreter with no existing event loop.
    # Use subprocess.run (via asyncio.to_thread) instead of asyncio.create_subprocess_exec
    # because Windows SelectorEventLoop does not support the latter (raises NotImplementedError).
    cmd = [sys.executable, "-m", "backend.evaluate.evaluate", req.pdf_path, goldens]
    if req.max_cases:
        cmd.append(str(req.max_cases))

    def _run_subprocess():
        return subprocess.run(
            cmd,
            capture_output=True,
            cwd=os.getcwd(),
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    completed = await asyncio.to_thread(_run_subprocess)

    if completed.returncode != 0:
        err = (completed.stderr or "")[-2000:]
        raise HTTPException(500, f"Evaluation subprocess failed:\n{err}")

    output = completed.stdout or ""
    # evaluate.py prints __RESULT__<json> on the last stdout line
    result_line = next(
        (l for l in reversed(output.splitlines()) if l.startswith("__RESULT__")),
        None,
    )
    if result_line:
        return json.loads(result_line[len("__RESULT__"):])

    raise HTTPException(500, "Evaluation completed but produced no result JSON")
