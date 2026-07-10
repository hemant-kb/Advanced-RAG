"""
Document ingestion pipeline.

Three processing modes (set via PIPELINE_MODE in config / per session from UI):

  low_cost     - tables as markdown only, images always VLM
  hybrid       - tables: rule-based routing (simple->markdown, complex->VLM)
  high_quality - all tables via VLM, images always VLM

Text extraction uses PyMuPDF4LLM (layout-aware, handles multi-column).
Images always use VLM captioning (charts/figures need semantic descriptions).
Text pipeline:
  1. Full-doc markdown assembled across all pages (preserves cross-page sections)
  2. Preprocessed (ligatures, hyphenation, whitespace)
  3. Split by Markdown headings (#/##/###) → each section is a semantic unit
  4. Within each section: tiktoken (cl100k_base) parent splitter → child splitter
  Parent chunks (~512 tokens) sent to LLM; child chunks (~256 tokens) embedded+searched.
  Section boundaries are never crossed.
Tables and images are still processed per-page in parallel threads.
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import re
from pathlib import Path

import fitz
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from backend.config import (
    CHART_CAPTION_PROMPT,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_SUMMARY_MODEL,
    HEADER_FOOTER_MARGIN,
    IMAGE_CAPTION_PROMPT,
    IMAGE_STORE_DIR,
    MIN_IMAGE_AREA,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PIPELINE_MAX_WORKERS,
    PIPELINE_MODE,
    SUMMARY_BATCH_CHARS,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_FINAL_PROMPT,
    TABLE_MIN_COLS,
    TABLE_MIN_ROWS,
    TABLE_VLM_EMPTY_RATIO,
    TABLE_VLM_HEADER_DEPTH,
    TABLE_VLM_MAX_COLS,
    TABLE_VLM_MAX_ROWS,
    TABLE_VLM_PROMPT,
    VISION_MODEL,
)
from backend.rag.audit import IngestionAudit
# vector_store and vision_llm are imported lazily inside ingest_pdf()
# so that parse/save-tables/save-images commands don't trigger BGE model
# loading or OpenAI client initialization at import time.


# ── tiktoken splitters (lazy-initialised on first use) ──────────
# cl100k_base is the tokenizer family used by text-embedding-3-small,
# so chunk sizes are exact embedding-token counts, preventing silent
# truncation at the model's 8191-token hard limit.

_child_splitter:  RecursiveCharacterTextSplitter | None = None
_parent_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitters() -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    global _child_splitter, _parent_splitter
    if _child_splitter is None or _parent_splitter is None:
        _child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            add_start_index=True,
        )
        _parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            add_start_index=True,
        )
    return _child_splitter, _parent_splitter


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _preprocess_text(text: str) -> str:
    """
    Clean raw PyMuPDF4LLM output before chunking.
    Operations are conservative — no stemming, no stopword removal.
    """
    # Fix ligatures (common in academic/typeset PDFs)
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)

    # Fix hyphenated line breaks: "trans-\nformer" -> "transformer"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Normalize whitespace — collapse multiple spaces/tabs but keep newlines
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove zero-width / invisible characters
    text = re.sub(r"[​‌‍﻿]", "", text)

    # Strip inline citation markers like [1], [23], [1,2,3] — they clutter embeddings
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # Remove isolated page numbers (a lone number on its own line)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Remove empty bullet/list lines — PDF layout often produces "- \n- \n" artifacts
    # where the bullet marker and content are on separate lines, or items have no text.
    text = re.sub(r"^[ \t]*[-*•]\s*$", "", text, flags=re.MULTILINE)

    # Collapse runs of 3+ blank lines down to 2 (keeps paragraph breaks readable)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Table complexity detection (for hybrid mode)
# ---------------------------------------------------------------------------

def _estimate_header_depth(table) -> int:
    """
    Estimate how many header rows the table has.
    Heuristic: consecutive rows from the top where all cells are short strings
    with no numeric-looking values.
    """
    depth = 0
    for row in table.rows[:3]:
        cells = [str(c or "").strip() for c in row.cells]
        if not cells:
            break
        has_numbers = any(re.search(r"\d{2,}", c) for c in cells)
        all_short   = all(len(c) < 40 for c in cells)
        if all_short and not has_numbers:
            depth += 1
        else:
            break
    return depth


def _is_complex_table(table) -> bool:
    """
    Return True if the table should be processed with VLM instead of markdown.
    Used in hybrid mode.
    """
    rows = len(table.rows)
    cols = len(table.rows[0].cells) if table.rows else 0

    if rows > TABLE_VLM_MAX_ROWS:
        return True

    if cols > TABLE_VLM_MAX_COLS:
        return True

    # Detect merged-cell artifacts: high ratio of empty cells
    total_cells = rows * cols
    if total_cells > 0:
        empty = sum(
            1 for row in table.rows
            for cell in row.cells
            if not str(cell or "").strip()
        )
        if empty / total_cells > TABLE_VLM_EMPTY_RATIO:
            return True

    if _estimate_header_depth(table) > TABLE_VLM_HEADER_DEPTH:
        return True

    return False


# ---------------------------------------------------------------------------
# VLM helpers (images + tables)
# ---------------------------------------------------------------------------

def _png_from_table_bbox(page: fitz.Page, bbox: tuple) -> bytes:
    """Render the table region on the page to PNG at 2x resolution."""
    rect = fitz.Rect(bbox)
    mat  = fitz.Matrix(2, 2)
    clip = page.get_pixmap(matrix=mat, clip=rect)
    return clip.tobytes("png")


# Lazy singleton — keeps import-time light (parse-only commands never touch
# OpenAI) while avoiding a new client per captioned table/image.
_vision_llm = None

def _get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        from langchain_openai import ChatOpenAI
        _vision_llm = ChatOpenAI(model=VISION_MODEL)
    return _vision_llm


@traceable(name="vlm_caption", run_type="llm",
           metadata={"component": "document_pipeline", "step": "vlm_caption"})
def _vlm_caption(png_bytes: bytes, prompt: str) -> str:
    vision_llm = _get_vision_llm()
    b64 = base64.b64encode(png_bytes).decode()
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }
    try:
        return (vision_llm.invoke([msg]).content or "").strip()
    except Exception as exc:
        return f"[caption failed: {exc}]"


# ---------------------------------------------------------------------------
# Header / footer detection
# ---------------------------------------------------------------------------

def _is_header_footer(bbox: tuple, page_height: float) -> bool:
    _, y0, _, y1 = bbox
    return y1 <= page_height * HEADER_FOOTER_MARGIN or y0 >= page_height * (1 - HEADER_FOOTER_MARGIN)


def _first_heading(md_text: str) -> str:
    for line in md_text.splitlines():
        m = re.match(r"^#{1,3}\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def _split_by_headings(text: str) -> list[tuple[dict, str]]:
    """
    Split a full-document markdown string into (heading_hierarchy, body) sections.

    Tracks h1/h2/h3 independently so subsection names like "Summary" or
    "Recommendations" are disambiguated by their parent headings.

    Returns list of (hierarchy_dict, section_text) where hierarchy_dict has keys:
      "h1", "h2", "h3" — each is the most recent heading at that level, or "".
      "section_heading" — the immediate heading label (deepest non-empty level).
    """
    sections: list[tuple[dict, str]] = []
    h1 = h2 = h3 = ""
    current_lines: list[str] = []

    def _flush(h1, h2, h3):
        body = "\n".join(current_lines).strip()
        if body:
            immediate = h3 or h2 or h1
            sections.append(({"h1": h1, "h2": h2, "h3": h3, "section_heading": immediate}, body))

    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.+)", line.strip())
        if m:
            _flush(h1, h2, h3)
            current_lines = []
            level = len(m.group(1))
            label = m.group(2).strip()
            if level == 1:
                h1, h2, h3 = label, "", ""
            elif level == 2:
                h2, h3 = label, ""
            else:
                h3 = label
        else:
            current_lines.append(line)

    _flush(h1, h2, h3)
    return sections


# ---------------------------------------------------------------------------
# Chart heuristic
# ---------------------------------------------------------------------------

def _looks_like_chart(page: fitz.Page, img_bbox: tuple) -> bool:
    x0, y0, x1, y1 = img_bbox
    vicinity    = fitz.Rect(x0 - 10, y0 - 30, x1 + 10, y1 + 30)
    nearby_text = page.get_text("text", clip=vicinity).lower()
    signals     = ["%", "axis", "fig", "figure", "chart", "graph", "plot", "revenue", "growth"]
    return any(s in nearby_text for s in signals)


def _is_extreme_aspect_ratio(w: int, h: int, max_ratio: float = 10.0) -> bool:
    """Return True for very wide/short or very tall/narrow images (rules, dividers, borders)."""
    if h == 0 or w == 0:
        return True
    return max(w / h, h / w) > max_ratio


def _is_solid_color(pix: fitz.Pixmap, threshold: float = 0.95) -> bool:
    """Return True if the pixmap is almost entirely one color (background fills, watermarks)."""
    try:
        n = pix.n - pix.alpha
        if n <= 0:
            return False
        samples = pix.samples
        if len(samples) < n:
            return False
        first_pixel = samples[:n]
        step    = n * 50
        matches = 0
        total   = 0
        for i in range(0, len(samples) - n + 1, step):
            total += 1
            if samples[i : i + n] == first_pixel:
                matches += 1
        return total > 0 and (matches / total) >= threshold
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

@traceable(name="process_page", run_type="chain",
           metadata={"component": "document_pipeline", "step": "page_extraction"})
def _process_page(
    page: fitz.Page,
    doc: fitz.Document,
    page_num: int,
    source: str,
    doc_title: str,
    page_md: str,
    mode: str,
    image_dir: Path,
    audit: IngestionAudit | None = None,
) -> tuple[list[Document], str]:
    docs: list[Document] = []
    page_height = page.rect.height
    page_width  = page.rect.width
    base_meta   = {"source": source, "doc_title": doc_title, "page": page_num}

    # ----------------------------------------------------------------
    # TABLES
    # ----------------------------------------------------------------
    table_bboxes: list[tuple] = []
    _table_idx = 0
    try:
        for table in page.find_tables().tables:
            if not table.rows or len(table.rows) < TABLE_MIN_ROWS:
                continue
            if not table.rows[0].cells or len(table.rows[0].cells) < TABLE_MIN_COLS:
                continue
            bbox = table.bbox
            if _is_header_footer(bbox, page_height):
                continue
            table_bboxes.append(bbox)
            _table_idx += 1

            rows = len(table.rows)
            cols = len(table.rows[0].cells)

            use_vlm = (
                mode == "high_quality"
                or (mode == "hybrid" and _is_complex_table(table))
            )

            png_bytes_table = _png_from_table_bbox(page, bbox)

            if use_vlm:
                content = _vlm_caption(png_bytes_table, TABLE_VLM_PROMPT)
                method  = "vlm"
                if audit:
                    audit.save_table_vlm(page_num, _table_idx, png_bytes_table, content)
            else:
                try:
                    df        = table.to_pandas()
                    content   = df.to_markdown(index=False)
                    json_data = df.to_dict(orient="records")
                except Exception:
                    rows_data  = [[cell for cell in row.cells] for row in table.rows]
                    header_row = [str(c or "") for c in rows_data[0]]
                    content = "| " + " | ".join(header_row) + " |\n"
                    content += "| " + " | ".join("---" for _ in header_row) + " |\n"
                    for row in rows_data[1:]:
                        content += "| " + " | ".join(str(c or "") for c in row) + " |\n"
                    json_data = []
                method = "markdown"
                if audit:
                    audit.save_table_markdown(page_num, _table_idx, png_bytes_table, content)

            meta = {
                **base_meta,
                "type": "table",
                "bbox": list(bbox),
                "row_count": rows,
                "col_count": cols,
                "table_method": method,
            }
            if method == "markdown":
                meta["json_data"] = json.dumps(json_data)

            docs.append(Document(page_content=content, metadata=meta))

    except Exception:
        pass

    # ----------------------------------------------------------------
    # IMAGES / CHARTS
    # Captions are stored as page_content (for retrieval).
    # PNG is saved to disk; image_path stored in metadata (for visual LLM fallback).
    # No base64 in Qdrant — keeps the vector store lean.
    # Filters: min area, header/footer margin, extreme aspect ratio, solid color.
    # ----------------------------------------------------------------
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pix  = fitz.Pixmap(doc, xref)
            w, h = pix.width, pix.height

            if w * h < MIN_IMAGE_AREA:
                if audit:
                    audit.save_image_skipped(page_num, xref, f"too_small_{w}x{h}")
                pix = None
                continue

            if _is_extreme_aspect_ratio(w, h):
                if audit:
                    audit.save_image_skipped(page_num, xref, f"extreme_aspect_{w}x{h}")
                pix = None
                continue

            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            if _is_solid_color(pix):
                if audit:
                    audit.save_image_skipped(page_num, xref, f"solid_color_{w}x{h}")
                pix = None
                continue

            png_bytes = pix.tobytes("png")
            pix = None

            img_rects = page.get_image_rects(xref)
            img_bbox  = tuple(img_rects[0]) if img_rects else (0, 0, page_width, page_height)
            if _is_header_footer(img_bbox, page_height):
                if audit:
                    audit.save_image_skipped(page_num, xref, "header_footer_margin")
                continue

            # Save PNG to disk
            img_filename = f"page{page_num:03d}_xref{xref}.png"
            img_path     = image_dir / img_filename
            img_path.write_bytes(png_bytes)

            is_chart = _looks_like_chart(page, img_bbox)
            prompt   = CHART_CAPTION_PROMPT if is_chart else IMAGE_CAPTION_PROMPT
            caption  = _vlm_caption(png_bytes, prompt)
            if not caption:
                img_path.unlink(missing_ok=True)
                continue

            if audit:
                audit.save_image_kept(page_num, xref, png_bytes, caption)

            docs.append(Document(
                page_content=caption,
                metadata={
                    **base_meta,
                    "type": "chart" if is_chart else "image",
                    "bbox": list(img_bbox),
                    "is_chart": is_chart,
                    "width": w,
                    "height": h,
                    "image_path": str(img_path),
                },
            ))
        except Exception:
            continue

    # ----------------------------------------------------------------
    # TEXT — stripped markdown returned for full-doc assembly in ingest_pdf.
    # Table pipe lines removed here so they are not double-counted when
    # ingest_pdf assembles the full document and splits by headings.
    # ----------------------------------------------------------------
    clean_lines = []
    in_table    = False
    for line in page_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
            continue
        if in_table and stripped == "":
            in_table = False
            continue
        in_table = False
        clean_lines.append(line)

    page_text_raw = "\n".join(clean_lines).strip()
    if audit and page_text_raw:
        audit.save_page_text(page_num, page_md, page_text_raw)

    return docs, page_text_raw


# ---------------------------------------------------------------------------
# Document summarisation (called at end of ingest_pdf)
# ---------------------------------------------------------------------------

import logging as _logging
_dp_logger = _logging.getLogger("rag.document_pipeline")


def _generate_document_summary(
    all_docs: list[Document],
    session_id: str,
    source: str,
    doc_title: str,
    audit: IngestionAudit | None = None,
) -> str | None:
    """
    Generate a full-document summary by:
      1. Collecting unique parent text chunks in page order.
      2. Batching them into windows of SUMMARY_BATCH_CHARS chars.
      3. Summarising each batch in parallel via Groq (SUMMARY_CHUNK_PROMPT).
      4. Combining chunk summaries into a final summary (SUMMARY_FINAL_PROMPT).
      5. Storing the summary as a Document with type=document_summary.

    All errors are caught and logged — never raises so ingest is not affected.
    """
    try:
        import concurrent.futures as _cf
        import httpx
        from openai import OpenAI

        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            http_client=httpx.Client(verify=False),
        )

        # 1. Collect unique parent chunks, sorted by page then by order seen
        seen_parents: set[str] = set()
        ordered_parents: list[tuple[int, str]] = []
        for doc in all_docs:
            if doc.metadata.get("type") != "text":
                continue
            parent = doc.metadata.get("parent_content") or doc.page_content
            if parent not in seen_parents:
                seen_parents.add(parent)
                page = doc.metadata.get("page", 0)
                ordered_parents.append((page, parent))

        if not ordered_parents:
            _dp_logger.warning("[summary] No text chunks found — skipping summary for %s", source)
            if audit:
                audit.save_summary_error("no text chunks found")
            return None

        # Sort by page to maintain document order
        ordered_parents.sort(key=lambda x: x[0])
        parent_texts = [p for _, p in ordered_parents]

        # 2. Greedy batch into windows of SUMMARY_BATCH_CHARS chars
        batches: list[str] = []
        current_batch: list[str] = []
        current_len = 0
        for text in parent_texts:
            if current_len + len(text) > SUMMARY_BATCH_CHARS and current_batch:
                batches.append("\n\n".join(current_batch))
                current_batch = [text]
                current_len = len(text)
            else:
                current_batch.append(text)
                current_len += len(text)
        if current_batch:
            batches.append("\n\n".join(current_batch))

        _dp_logger.info("[summary] Summarising %s: %d parent chunks → %d batches",
                        source, len(parent_texts), len(batches))

        # 3. Summarise each batch in parallel
        def _summarise_batch(batch_text: str) -> str:
            try:
                prompt = SUMMARY_CHUNK_PROMPT.format(text=batch_text)
                resp = groq_client.chat.completions.create(
                    model=GROQ_SUMMARY_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                _dp_logger.warning("[summary] Batch summarisation failed: %s", exc)
                return ""

        chunk_summaries: list[str] = []
        with _cf.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_summarise_batch, b) for b in batches]
            for f in _cf.as_completed(futures):
                try:
                    result = f.result()
                    if result:
                        chunk_summaries.append(result)
                except Exception as exc:
                    _dp_logger.warning("[summary] Batch future failed: %s", exc)

        if not chunk_summaries:
            _dp_logger.warning("[summary] All batch summaries empty for %s", source)
            if audit:
                audit.save_summary_error("all Groq batch calls failed")
            return None

        # 4. Final summary from chunk summaries
        combined = "\n\n---\n\n".join(chunk_summaries)
        final_prompt = SUMMARY_FINAL_PROMPT.format(summaries=combined)
        try:
            final_resp = groq_client.chat.completions.create(
                model=GROQ_SUMMARY_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
            )
            final_summary = final_resp.choices[0].message.content.strip()
        except Exception as exc:
            _dp_logger.warning("[summary] Final summary call failed: %s", exc)
            if audit:
                audit.save_summary_error(f"final Groq call failed: {exc}")
            return None

        if not final_summary:
            _dp_logger.warning("[summary] Final summary was empty for %s", source)
            if audit:
                audit.save_summary_error("final summary was empty")
            return None

        # 5. Store as a document_summary chunk
        summary_doc = Document(
            page_content=final_summary,
            metadata={
                "type": "document_summary",
                "source": source,
                "doc_title": doc_title,
                "page": 0,
                "section_heading": "__summary__",
            },
        )

        from backend.rag.vector_store import add_documents
        add_documents([summary_doc], session_id)
        _dp_logger.info("[summary] Stored document summary for %s (%d chars)", source, len(final_summary))

        if audit:
            audit.save_summary(
                summary_text=final_summary,
                batches=len(batches),
                parent_chunks=len(parent_texts),
                model=GROQ_SUMMARY_MODEL,
                chars=len(final_summary),
            )

        return final_summary

    except Exception as exc:
        _dp_logger.warning("[summary] _generate_document_summary failed for %s: %s", source, exc)
        if audit:
            audit.save_summary_error(str(exc))
        return None


# ---------------------------------------------------------------------------
# Ingestion phases
# ---------------------------------------------------------------------------

def _extract_pages(
    doc: fitz.Document,
    source: str,
    doc_title: str,
    page_mds: list[str],
    mode: str,
    image_dir: Path,
    audit: IngestionAudit | None,
) -> tuple[list[Document], dict[int, str]]:
    """
    Phase 1: tables + images per page, in parallel threads.
    Returns (non-text docs, {page_num: stripped page markdown}).
    """
    total_pages = len(doc)
    page_args = [
        (doc[i], doc, i + 1, source, doc_title, page_mds[i], mode, image_dir, audit)
        for i in range(total_pages)
    ]

    docs: list[Document] = []
    page_text_by_num: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=PIPELINE_MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(_process_page, p, d, n, s, t, md, m, idir, aud): n
            for p, d, n, s, t, md, m, idir, aud in page_args
        }
        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                page_docs, page_raw = future.result()
                docs.extend(page_docs)
                if page_raw:
                    page_text_by_num[page_num] = page_raw
            except Exception:
                pass

    return docs, page_text_by_num


def _build_text_chunks(
    page_text_by_num: dict[int, str],
    source: str,
    doc_title: str,
    audit: IngestionAudit | None,
) -> list[Document]:
    """
    Phase 2: heading-aware text splitting over the full document.
    Pages are assembled in order, preprocessed, split by Markdown headings,
    then each section is split into parent (LLM context) and child (embedded)
    chunks that never cross section boundaries.
    """
    child_splitter, parent_splitter = _get_splitters()

    # Build a list of (page_num, line) to track page boundaries
    ordered_lines: list[tuple[int, str]] = []
    for pn in sorted(page_text_by_num.keys()):
        for line in page_text_by_num[pn].splitlines():
            ordered_lines.append((pn, line))

    full_text = "\n".join(ln for _, ln in ordered_lines)
    full_text = _preprocess_text(full_text)

    # Split by headings — each section is (hierarchy_dict, section_body)
    sections = _split_by_headings(full_text)

    # For each section, find its first page by locating heading text in the
    # ordered_lines list. Single pass resolves every heading at once instead
    # of rescanning the whole document per section. Carry forward if not found.
    _pending_headings = {
        h["section_heading"] for h, _ in sections if h["section_heading"]
    }
    _heading_first_page: dict[str, int] = {}
    for pn, ln in ordered_lines:
        if not _pending_headings:
            break
        for heading in [h for h in _pending_headings if h in ln]:
            _heading_first_page[heading] = pn
            _pending_headings.discard(heading)

    docs: list[Document] = []
    _last_page = 1
    audit_chunk_sections: list[tuple[str, list[str], list[list[str]]]] = []

    for hierarchy, section_body in sections:
        immediate = hierarchy["section_heading"]

        if immediate and immediate in _heading_first_page:
            _last_page = _heading_first_page[immediate]
        first_page = _last_page

        base_meta = {
            "source":          source,
            "doc_title":       doc_title,
            "page":            first_page,
            "section_heading": immediate,
            "h1":              hierarchy["h1"],
            "h2":              hierarchy["h2"],
            "h3":              hierarchy["h3"],
        }

        # Parent → child within section boundary.
        # page_content = clean child text (stored, shown in UI, BM25-indexed).
        # Contextual prefix is built at embed time in vector_store.add_documents
        # so the stored text stays clean and the embedding format can change
        # without re-parsing the PDF.
        parents = parent_splitter.split_text(section_body)
        _children_per_parent: list[list[str]] = []
        for parent_text in parents:
            kids = child_splitter.split_text(parent_text)
            _children_per_parent.append(kids)
            for child in kids:
                docs.append(Document(
                    page_content=child,
                    metadata={
                        **base_meta,
                        "type":           "text",
                        "parent_content": parent_text,
                    },
                ))

        audit_chunk_sections.append((immediate or "(preamble)", parents, _children_per_parent))

    if audit:
        for sec_heading, parents, children_per_parent in audit_chunk_sections:
            audit.save_chunks_for_section(sec_heading, parents, children_per_parent)

    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@traceable(name="ingest_pdf", run_type="chain",
           metadata={"component": "document_pipeline", "step": "ingestion"})
def ingest_pdf(file_path: str, session_id: str, mode: str | None = None) -> dict:
    """
    Ingest a PDF end-to-end into the session's Qdrant collection.
    mode overrides PIPELINE_MODE from config if provided.

    Text pipeline:
      - Pages processed in parallel for tables + images (fast, IO-bound)
      - Per-page stripped markdown collected, assembled into one full-doc string
      - Full doc preprocessed then split by Markdown headings
      - Each section split into parent then child chunks (tiktoken cl100k_base)
      - Parent/child chunks never cross section boundaries
    """
    mode   = mode or PIPELINE_MODE
    source = Path(file_path).name

    from backend.rag.vector_store import add_documents, list_sources
    existing = list_sources(session_id)
    if source in existing:
        return {
            "source": source, "chunks_created": 0,
            "text_chunks": 0, "image_chunks": 0, "table_chunks": 0,
            "skipped": True,
        }

    # Start audit
    audit = IngestionAudit(session_id)
    audit.start(file_path, mode)

    # Create per-session image directory
    image_dir = Path(IMAGE_STORE_DIR) / session_id
    image_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []

    try:
        with fitz.open(file_path) as doc:
            if doc.is_encrypted:
                try:
                    doc.authenticate("")
                except Exception as exc:
                    raise ValueError(f"Cannot open encrypted PDF: {exc}")

            doc_title   = (doc.metadata or {}).get("title") or source
            page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, ignore_images=True)
            page_mds    = [ch["text"] for ch in page_chunks]
            total_pages = len(doc)

            # Phase 1: tables + images — parallel per-page
            page_docs, page_text_by_num = _extract_pages(
                doc, source, doc_title, page_mds, mode, image_dir, audit,
            )
            all_docs.extend(page_docs)

            # Phase 2: heading-aware text splitting over the full document
            all_docs.extend(
                _build_text_chunks(page_text_by_num, source, doc_title, audit)
            )

        if not all_docs:
            raise ValueError("No content extracted from PDF.")

        added        = add_documents(all_docs, session_id)
        text_chunks  = sum(1 for d in all_docs if d.metadata.get("type") == "text")
        image_chunks = sum(1 for d in all_docs if d.metadata.get("type") in ("image", "chart"))
        table_chunks = sum(1 for d in all_docs if d.metadata.get("type") == "table")

        _generate_document_summary(all_docs, session_id, source, doc_title, audit=audit)

        from backend.config import get_collection_name
        audit.save_images_summary()
        audit.save_tables_summary()
        audit.save_chunks_summary()
        audit.save_extraction_summary(total_pages)
        audit.save_upsert_summary(added, get_collection_name(session_id))
        audit.finish()

        return {
            "source":         source,
            "mode":           mode,
            "chunks_created": added,
            "text_chunks":    text_chunks,
            "image_chunks":   image_chunks,
            "table_chunks":   table_chunks,
        }

    except Exception as exc:
        audit.finish(error=str(exc))
        raise
