"""
Pipeline audit trail — saves every stage artifact to disk per session.

Ingestion layout  (data/runs/{session_id}/):
  00_source/original.pdf
  01_extraction/page_001_raw.txt, page_001_clean.txt, extraction_summary.json
  02_images/page_003_xref45_KEPT.png, page_003_xref45_KEPT_caption.txt
            page_001_xref12_SKIP_too_small_120x15.txt
            images_summary.json
  03_tables/page_007_table1_MARKDOWN.png, page_007_table1_MARKDOWN.md
            page_009_table2_VLM.png,      page_009_table2_VLM.txt
            tables_summary.json
  04_chunks/page_001_parent_01.txt, page_001_parent_01_child_a.txt ...
            chunks_summary.json
  05_vector_store/upsert_summary.json
  ingestion_summary.json

Query layout  (data/runs/{session_id}/queries/{ts}_{short_query}/):
  01_query.txt
  02_dense_vector.json
  03_sparse_vector.json
  04_prefetch_results.json
  05_reranked_results.json
  06_relevancy_check.json
  07_rewritten_query.txt        (only if rewrite happened)
  08_rerank_after_rewrite.json  (only if rewrite happened)
  09_context_sent.txt
  10_phase1_answer.txt
  11_vision_triggered.json      (only if [NEEDS_VISUAL] fired)
  12_vision_images/             (copies of PNGs sent to vision LLM)
  13_final_answer.txt
  query_summary.json
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

RUNS_DIR = Path("data/runs")


# ── Helpers ──────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _write_bytes(path: Path, data: bytes) -> None:
    try:
        path.write_bytes(data)
    except Exception:
        pass


# ── Ingestion audit ──────────────────────────────────────────────

class IngestionAudit:
    """Collects and saves all ingestion artifacts for one session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.root = RUNS_DIR / session_id
        self._t0  = time.perf_counter()
        self._summary: dict = {
            "session_id":       session_id,
            "mode":             None,
            "source":           None,
            "pages":            0,
            "text_pages":       0,
            "images_kept":      0,
            "images_skipped":   0,
            "tables_markdown":  0,
            "tables_vlm":       0,
            "parent_chunks":    0,
            "child_chunks":     0,
            "vectors_upserted": 0,
            "elapsed_s":        None,
            "errors":           [],
        }
        self._image_records: list[dict] = []
        self._table_records: list[dict] = []
        self._chunk_records: list[dict] = []

    def _dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── lifecycle ────────────────────────────────────────────────

    def start(self, pdf_path: str, mode: str) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._summary["source"] = Path(pdf_path).name
            self._summary["mode"]   = mode
            src_dir = self._dir("00_source")
            shutil.copy2(pdf_path, src_dir / "original.pdf")
        except Exception:
            pass

    def finish(self, error: str | None = None) -> None:
        try:
            if error:
                self._summary["errors"].append(error)
            self._summary["elapsed_s"] = round(time.perf_counter() - self._t0, 2)
            _write_json(self.root / "ingestion_summary.json", self._summary)
        except Exception:
            pass

    # ── stage 01: text extraction ────────────────────────────────

    def save_page_text(self, page_num: int, raw: str, clean: str) -> None:
        try:
            d = self._dir("01_extraction")
            _write_text(d / f"page_{page_num:03d}_raw.txt",   raw)
            _write_text(d / f"page_{page_num:03d}_clean.txt", clean)
            self._summary["pages"] = max(self._summary["pages"], page_num)
            if clean.strip():
                self._summary["text_pages"] += 1
        except Exception:
            pass

    def save_extraction_summary(self, page_count: int) -> None:
        try:
            d = self._dir("01_extraction")
            self._summary["pages"] = page_count
            _write_json(d / "extraction_summary.json", {
                "pages":      page_count,
                "text_pages": self._summary["text_pages"],
            })
        except Exception:
            pass

    # ── stage 02: images ─────────────────────────────────────────

    def save_image_kept(self, page_num: int, xref: int, png_bytes: bytes, caption: str) -> None:
        try:
            d    = self._dir("02_images")
            stem = f"page_{page_num:03d}_xref{xref}_KEPT"
            _write_bytes(d / f"{stem}.png", png_bytes)
            _write_text(d / f"{stem}_caption.txt", caption)
            self._image_records.append({"page": page_num, "xref": xref, "status": "kept"})
            self._summary["images_kept"] += 1
        except Exception:
            pass

    def save_image_skipped(self, page_num: int, xref: int, reason: str) -> None:
        try:
            d           = self._dir("02_images")
            safe_reason = reason.replace(" ", "_").replace("/", "_")[:40]
            _write_text(d / f"page_{page_num:03d}_xref{xref}_SKIP_{safe_reason}.txt", reason)
            self._image_records.append({"page": page_num, "xref": xref, "status": "skipped", "reason": reason})
            self._summary["images_skipped"] += 1
        except Exception:
            pass

    def save_images_summary(self) -> None:
        try:
            _write_json(self._dir("02_images") / "images_summary.json", {
                "kept":    self._summary["images_kept"],
                "skipped": self._summary["images_skipped"],
                "records": self._image_records,
            })
        except Exception:
            pass

    # ── stage 03: tables ─────────────────────────────────────────

    def save_table_markdown(self, page_num: int, table_idx: int, png_bytes: bytes, markdown: str) -> None:
        try:
            d    = self._dir("03_tables")
            stem = f"page_{page_num:03d}_table{table_idx}_MARKDOWN"
            _write_bytes(d / f"{stem}.png", png_bytes)
            _write_text(d / f"{stem}.md",   markdown)
            self._table_records.append({"page": page_num, "table": table_idx, "method": "markdown"})
            self._summary["tables_markdown"] += 1
        except Exception:
            pass

    def save_table_vlm(self, page_num: int, table_idx: int, png_bytes: bytes, caption: str) -> None:
        try:
            d    = self._dir("03_tables")
            stem = f"page_{page_num:03d}_table{table_idx}_VLM"
            _write_bytes(d / f"{stem}.png", png_bytes)
            _write_text(d / f"{stem}.txt",  caption)
            self._table_records.append({"page": page_num, "table": table_idx, "method": "vlm"})
            self._summary["tables_vlm"] += 1
        except Exception:
            pass

    def save_tables_summary(self) -> None:
        try:
            _write_json(self._dir("03_tables") / "tables_summary.json", {
                "markdown": self._summary["tables_markdown"],
                "vlm":      self._summary["tables_vlm"],
                "records":  self._table_records,
            })
        except Exception:
            pass

    # ── stage 04: chunks ─────────────────────────────────────────

    def save_chunks_for_section(
        self,
        section_heading: str,
        parents: list[str],
        children_per_parent: list[list[str]],
    ) -> None:
        try:
            d = self._dir("04_chunks")
            safe = re.sub(r"[^\w\s-]", "", section_heading)[:40].strip().replace(" ", "_") or "preamble"
            # avoid collisions if heading appears on multiple pages
            idx = sum(1 for r in self._chunk_records if r.get("section", "").startswith(safe))
            prefix = f"{safe}_{idx:02d}" if idx else safe

            for pi, parent in enumerate(parents, 1):
                _write_text(d / f"{prefix}_parent_{pi:02d}.txt", parent)
                self._summary["parent_chunks"] += 1
                kids = children_per_parent[pi - 1] if pi - 1 < len(children_per_parent) else []
                for ci, child in enumerate(kids, 1):
                    label = chr(ord("a") + ci - 1) if ci <= 26 else str(ci)
                    _write_text(d / f"{prefix}_parent_{pi:02d}_child_{label}.txt", child)
                    self._summary["child_chunks"] += 1
            self._chunk_records.append({"section": safe, "parents": len(parents)})
        except Exception:
            pass

    def save_chunks_summary(self) -> None:
        try:
            _write_json(self._dir("04_chunks") / "chunks_summary.json", {
                "parent_chunks": self._summary["parent_chunks"],
                "child_chunks":  self._summary["child_chunks"],
                "sections":      self._chunk_records,
            })
        except Exception:
            pass

    # ── stage 05: vector store ───────────────────────────────────

    def save_upsert_summary(self, count: int, collection: str) -> None:
        try:
            self._summary["vectors_upserted"] = count
            _write_json(self._dir("05_vector_store") / "upsert_summary.json", {
                "collection":       collection,
                "vectors_upserted": count,
                "vector_dims":      1024,
                "model":            "BAAI/bge-m3",
            })
        except Exception:
            pass

    # ── stage 06: document summary ───────────────────────────────

    def save_summary(
        self,
        summary_text: str,
        batches: int,
        parent_chunks: int,
        model: str,
        chars: int,
    ) -> None:
        try:
            d = self._dir("06_summary")
            _write_text(d / "summary.txt", summary_text)
            _write_json(d / "summary_meta.json", {
                "status":        "ok",
                "model":         model,
                "parent_chunks": parent_chunks,
                "batches":       batches,
                "chars":         chars,
            })
            self._summary["summary_status"]  = "ok"
            self._summary["summary_chars"]   = chars
            self._summary["summary_batches"] = batches
        except Exception:
            pass

    def save_summary_error(self, reason: str) -> None:
        try:
            d = self._dir("06_summary")
            _write_json(d / "summary_meta.json", {
                "status": "failed",
                "reason": reason,
            })
            self._summary["summary_status"] = f"failed: {reason}"
        except Exception:
            pass


# ── Query audit ──────────────────────────────────────────────────

class QueryAudit:
    """Saves all query-pipeline artifacts for one question."""

    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query      = query
        self._t0        = time.perf_counter()
        self._rewrite_count = 0

        ts     = str(int(time.time()))
        safe_q = "".join(
            c if c.isalnum() or c in " _-" else "" for c in query[:40]
        ).strip().replace(" ", "_")
        self.root = RUNS_DIR / session_id / "queries" / f"{ts}_{safe_q}"
        self.root.mkdir(parents=True, exist_ok=True)

        self._summary: dict = {
            "session_id":       session_id,
            "query":            query,
            "rewritten":        False,
            "rewritten_query":  None,
            "is_relevant":      None,
            "phase":            "phase1",
            "vision_triggered": False,
            "chunks_prefetch":  0,
            "chunks_reranked":  0,
            "elapsed_s":        None,
        }

    # ── save helpers ─────────────────────────────────────────────

    def save_query(self) -> None:
        _write_text(self.root / "01_query.txt", self.query)

    def save_dense_vector(self, vector: list[float], timing_ms: float) -> None:
        _write_json(self.root / "02_dense_vector.json", {
            "dims":      len(vector),
            "first_5":   [round(v, 6) for v in vector[:5]],
            "last_5":    [round(v, 6) for v in vector[-5:]],
            "norm":      round(sum(v * v for v in vector) ** 0.5, 6),
            "timing_ms": round(timing_ms, 2),
        })

    def save_sparse_vector(self, sparse: dict) -> None:
        indices = sparse.get("indices", [])
        values  = sparse.get("values",  [])
        _write_json(self.root / "03_sparse_vector.json", {
            "non_zero_terms":  len(indices),
            "top_10_indices":  indices[:10],
            "top_10_values":   [round(v, 6) for v in values[:10]],
        })

    def save_prefetch_results(self, docs: list[Any]) -> None:
        records = []
        for i, d in enumerate(docs):
            meta = getattr(d, "metadata", {})
            records.append({
                "rank":    i + 1,
                "type":    meta.get("type", "text"),
                "page":    meta.get("page"),
                "score":   meta.get("score"),
                "snippet": getattr(d, "page_content", "")[:200],
            })
        self._summary["chunks_prefetch"] = len(records)
        _write_json(self.root / "04_prefetch_results.json", {
            "count": len(records), "results": records,
        })

    def save_reranked_results(self, docs: list[Any]) -> None:
        records = []
        for i, d in enumerate(docs):
            meta = getattr(d, "metadata", {})
            records.append({
                "rank":         i + 1,
                "type":         meta.get("type", "text"),
                "page":         meta.get("page"),
                "rerank_score": meta.get("rerank_score"),
                "snippet":      getattr(d, "page_content", "")[:200],
            })
        self._summary["chunks_reranked"] = len(records)
        _write_json(self.root / "05_reranked_results.json", {
            "count": len(records), "results": records,
        })

    def save_relevancy_check(self, is_relevant: bool, reason: str) -> None:
        self._summary["is_relevant"] = is_relevant
        _write_json(self.root / "06_relevancy_check.json", {
            "is_relevant": is_relevant,
            "reason":      reason,
        })

    def save_rewritten_query(self, rewritten: str) -> None:
        self._summary["rewritten"]      = True
        self._summary["rewritten_query"] = rewritten
        self._rewrite_count += 1
        _write_text(self.root / "07_rewritten_query.txt", rewritten)

    def save_rerank_after_rewrite(self, docs: list[Any]) -> None:
        records = []
        for i, d in enumerate(docs):
            meta = getattr(d, "metadata", {})
            records.append({
                "rank":         i + 1,
                "type":         meta.get("type", "text"),
                "page":         meta.get("page"),
                "rerank_score": meta.get("rerank_score"),
                "snippet":      getattr(d, "page_content", "")[:200],
            })
        _write_json(self.root / "08_rerank_after_rewrite.json", {
            "count": len(records), "results": records,
        })

    def save_context_sent(self, context: str) -> None:
        _write_text(self.root / "09_context_sent.txt", context)

    def save_phase1_answer(self, answer: str) -> None:
        _write_text(self.root / "10_phase1_answer.txt", answer)

    def save_vision_triggered(self, image_paths: list[str]) -> None:
        self._summary["phase"]           = "phase2_vision"
        self._summary["vision_triggered"] = True
        _write_json(self.root / "11_vision_triggered.json", {"image_paths": image_paths})
        img_dir = self.root / "12_vision_images"
        img_dir.mkdir(exist_ok=True)
        for p in image_paths:
            try:
                shutil.copy2(p, img_dir / Path(p).name)
            except Exception:
                pass

    def save_final_answer(self, answer: str) -> None:
        _write_text(self.root / "13_final_answer.txt", answer)

    def finish(self) -> None:
        try:
            self._summary["elapsed_s"] = round(time.perf_counter() - self._t0, 2)
            _write_json(self.root / "query_summary.json", self._summary)
        except Exception:
            pass
