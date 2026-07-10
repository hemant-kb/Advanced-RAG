"""
Qdrant vector store — OpenAI embeddings + Cohere reranker.

Embedding:  text-embedding-3-small  (OpenAI API, 1536-dim, 8191-token context)
Sparse:     FastEmbed Qdrant/bm25   (local, independent of the dense embedder)
Reranker:   Cohere rerank-v4.0-fast (API, ~300ms)
Search:     dense + sparse BM25 (RRF fusion) -> Cohere rerank top-N
"""
from __future__ import annotations

import logging
import traceback

from langchain_core.documents import Document

logger = logging.getLogger("rag.vector_store")
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

from backend.config import (
    EMBEDDING_MODEL,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    HYBRID_PREFETCH_LIMIT,
    QDRANT_API_KEY,
    QDRANT_SEARCH_LIMIT,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_TIMEOUT,
    QDRANT_URL,
    QDRANT_VECTOR_SIZE,
    RERANK_TOP_N,
    get_collection_name,
)

# ── OpenAI embedder — lazy-loaded on first use ───────────────────
_dense_embedder = None

def _get_embedder():
    global _dense_embedder
    if _dense_embedder is None:
        from langchain_openai import OpenAIEmbeddings
        _dense_embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _dense_embedder


# ── BM25 sparse embedder — lazy singleton (loading it per call costs
# a full model init on every query/upsert) ────────────────────────
_sparse_embedder = None

def _get_sparse_embedder():
    global _sparse_embedder
    if _sparse_embedder is None:
        from fastembed import SparseTextEmbedding
        _sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_embedder

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)


# ── Embedding helpers ────────────────────────────────────────────

def _embed_documents(texts: list[str]) -> list[list[float]]:
    return _get_embedder().embed_documents(texts)


def _embed_query(text: str) -> list[float]:
    return _get_embedder().embed_query(text)


# ── Collection management ────────────────────────────────────────

def _ensure_payload_index(collection_name: str) -> None:
    """Create a keyword index on metadata.type if it doesn't already exist.
    Required for filtered scroll queries (Qdrant rejects unindexed keyword filters)."""
    try:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.type",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # Already exists or collection gone — both are fine


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )
        _ensure_payload_index(collection_name)
    return collection_name


def delete_collection(session_id: str) -> None:
    name = get_collection_name(session_id)
    if qdrant_client.collection_exists(name):
        qdrant_client.delete_collection(name)


def collection_exists(session_id: str) -> bool:
    return qdrant_client.collection_exists(get_collection_name(session_id))


# ── Document upsert ──────────────────────────────────────────────

def _build_embedding_text(doc: Document) -> str:
    """
    Build the text sent to the dense embedding model.
    For text chunks: prepend doc title + heading hierarchy + page so that
    semantically thin chunks (bullet lists, short paragraphs) retrieve correctly.
    For tables/images: embed as-is (caption/markdown already self-contained).
    """
    if doc.metadata.get("type") != "text":
        return doc.page_content

    meta = doc.metadata
    parts: list[str] = []
    if meta.get("doc_title"):
        parts.append(f"Title: {meta['doc_title']}")
    for level in ("h1", "h2", "h3"):
        if meta.get(level):
            parts.append(f"{level.upper()}: {meta[level]}")
    parts.append(doc.page_content)
    return "\n".join(parts)


@traceable(name="add_documents", run_type="tool",
           metadata={"component": "vector_store", "operation": "upsert", "db": "qdrant"})
def add_documents(docs: list[Document], session_id: str) -> int:
    if not docs:
        return 0

    collection_name = get_vectorstore(session_id)

    # Dense: embed contextual text (title + hierarchy + page + content)
    embed_texts = [_build_embedding_text(d) for d in docs]
    vectors = _embed_documents(embed_texts)

    # Sparse BM25: index clean page_content only — no prefix noise in keyword index
    bm25_texts = [d.page_content for d in docs]
    try:
        sparse_vecs = list(_get_sparse_embedder().embed(bm25_texts))
    except Exception:
        sparse_vecs = [None] * len(docs)

    points = []
    for i, (doc, vector) in enumerate(zip(docs, vectors)):
        payload = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        sv = sparse_vecs[i] if i < len(sparse_vecs) else None
        named_vectors: dict = {"": vector}
        if sv is not None:
            named_vectors[QDRANT_SPARSE_VECTOR_NAME] = qmodels.SparseVector(
                indices=sv.indices.tolist(),
                values=sv.values.tolist(),
            )
        points.append(qmodels.PointStruct(
            id=_doc_id(session_id, i, doc),
            vector=named_vectors,
            payload=payload,
        ))

    # Upsert in batches of 100
    for batch_start in range(0, len(points), 100):
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points[batch_start:batch_start + 100],
        )

    return len(docs)


def _doc_id(session_id: str, idx: int, doc: Document) -> str:
    import hashlib
    doc_type = doc.metadata.get("type", "text")
    key = f"{session_id}:{doc_type}:{doc.metadata.get('source','')}:{doc.metadata.get('page','')}:{idx}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Sparse (BM25) embedding ──────────────────────────────────────

def _embed_query_sparse(query: str) -> dict:
    try:
        result = list(_get_sparse_embedder().embed([query]))[0]
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}
    except Exception:
        return {}


# ── Structural query detection ───────────────────────────────────
# Queries asking for named document sections (conclusion, introduction, etc.)
# are navigational — the reranker scores semantic relevance, not section identity,
# so it reliably drops the correct chunk when it has a short/dense body.
# This set is matched case-insensitively against the query.

_STRUCTURAL_SECTIONS = {
    "conclusion", "conclusions",
    "introduction", "abstract",
    "summary", "overview",
    "related work", "background",
    "methodology", "methods",
    "results", "discussion",
    "future work", "limitations",
    "references", "appendix",
}


def _structural_section(query: str) -> str | None:
    """Return the section keyword if the query is asking for a named section, else None."""
    q = query.lower()
    for section in _STRUCTURAL_SECTIONS:
        if section in q:
            return section
    return None


def _promote_structural_matches(
    query: str,
    prefetch_docs: list[Document],
    reranked: list[Document],
    top_n: int,
) -> list[Document]:
    """
    For structural queries (e.g. "what is the conclusion"), scan the full prefetch
    pool for chunks whose section heading matches the target section and pin them
    into the result set, displacing the lowest-scored reranked chunks if needed.

    This prevents the reranker from dropping the correct structural chunk because
    it scored a semantically similar but wrong chunk (e.g. Acknowledgements) higher.
    """
    section = _structural_section(query)
    if not section:
        return reranked

    promoted: list[Document] = []
    reranked_ids = {id(d) for d in reranked}

    for doc in prefetch_docs:
        if id(doc) in reranked_ids:
            continue
        heading = " ".join([
            doc.metadata.get("h1", ""),
            doc.metadata.get("h2", ""),
            doc.metadata.get("h3", ""),
            doc.metadata.get("section_heading", ""),
        ]).lower()
        if section in heading:
            doc.metadata["rerank_score"] = doc.metadata.get("rerank_score", 0.0)
            doc.metadata["structurally_promoted"] = True
            promoted.append(doc)

    if not promoted:
        return reranked

    logger.info("[search] structural promotion: pinning %d chunk(s) for section '%s'",
                len(promoted), section)

    # Merge: promoted first, then remaining reranked up to top_n total
    seen_ids: set[int] = {id(d) for d in promoted}
    merged = list(promoted)
    for d in reranked:
        if id(d) not in seen_ids:
            merged.append(d)
    return merged[:top_n]


# ── Cohere reranker ──────────────────────────────────────────────

def _rerank_text(doc: Document) -> str:
    """Use parent_content for text chunks so the reranker sees full context, not just the child slice."""
    if doc.metadata.get("type") == "text":
        return doc.metadata.get("parent_content") or doc.page_content
    return doc.page_content


_cohere_http = None

def _get_cohere_http():
    """Persistent HTTP client — avoids a new TLS handshake per rerank call."""
    global _cohere_http
    if _cohere_http is None:
        import httpx
        _cohere_http = httpx.Client(verify=False, timeout=httpx.Timeout(connect=10, read=60, write=10, pool=5))
    return _cohere_http


def _cohere_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Rerank using Cohere rerank-v4.0-fast API (~300ms vs 23-27s for local BGE)."""
    if not docs:
        return docs
    passages = [_rerank_text(d)[:2048] for d in docs]
    try:
        resp = _get_cohere_http().post(
            "https://api.cohere.com/v2/rerank",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":      COHERE_RERANK_MODEL,
                "query":      query,
                "documents":  passages,
                "top_n":      top_n,
                "return_documents": False,
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        reranked = []
        for r in results:
            doc = docs[r["index"]]
            doc.metadata["rerank_score"] = round(float(r["relevance_score"]), 6)
            reranked.append(doc)
        return reranked
    except Exception:
        logger.error("[cohere_rerank] failed, falling back to order-preserved top-N\n%s",
                     traceback.format_exc())
        for i, doc in enumerate(docs[:top_n]):
            doc.metadata["rerank_score"] = round(1.0 - i * 0.1, 6)
        return docs[:top_n]


# ── Hybrid search + Cohere rerank ────────────────────────────────

def _points_to_docs(points) -> list[Document]:
    return [
        Document(
            page_content=(p.payload or {}).get("page_content", ""),
            metadata=(p.payload or {}).get("metadata", {}),
        )
        for p in points
    ]


def _dense_query(collection_name: str, dense_vector: list[float]):
    """Dense-only Qdrant query — the shared primary/fallback path."""
    return qdrant_client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="",
        limit=HYBRID_PREFETCH_LIMIT,
        with_payload=True,
    ).points


@traceable(name="hybrid_search", run_type="retriever",
           metadata={"component": "vector_store", "operation": "hybrid_search", "db": "qdrant"})
def search(
    query: str,
    session_id: str,
    k: int = QDRANT_SEARCH_LIMIT,
    _prefetch_cb=None,
    _reranked_cb=None,
    _timing_cb=None,   # optional: called with {"qdrant_ms": N, "rerank_ms": N, "embed_ms": N, "mode": str, "dense_vector": list, "sparse_vec": dict|None}
) -> list[Document]:
    """
    Dense + sparse BM25 (RRF fusion) -> Cohere rerank.
    Returns up to RERANK_TOP_N documents.
    """
    import time as _t
    if not collection_exists(session_id):
        return []

    collection_name = get_collection_name(session_id)
    _t_embed0 = _t.perf_counter()
    dense_vector = _embed_query(query)
    sparse_vec   = _embed_query_sparse(query)
    embed_ms = int((_t.perf_counter() - _t_embed0) * 1000)

    docs: list[Document] = []
    _qdrant_ms = 0
    _search_mode = "dense"
    try:
        _t_q0 = _t.perf_counter()
        if sparse_vec:
            _search_mode = "hybrid_rrf"
            logger.info("[search] attempting hybrid (dense+sparse RRF) on %s", collection_name)
            results = qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    qmodels.Prefetch(
                        query=dense_vector,
                        using="",
                        limit=HYBRID_PREFETCH_LIMIT,
                    ),
                    qmodels.Prefetch(
                        query=qmodels.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"],
                        ),
                        using=QDRANT_SPARSE_VECTOR_NAME,
                        limit=HYBRID_PREFETCH_LIMIT,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=HYBRID_PREFETCH_LIMIT,
                with_payload=True,
            ).points
            logger.info("[search] hybrid OK — %d points returned", len(results))
        else:
            _search_mode = "dense"
            logger.info("[search] no sparse vec — dense-only on %s", collection_name)
            results = _dense_query(collection_name, dense_vector)
            logger.info("[search] dense-only OK — %d points returned", len(results))

        _qdrant_ms = int((_t.perf_counter() - _t_q0) * 1000)
        docs = _points_to_docs(results)

    except Exception:
        _qdrant_ms = int((_t.perf_counter() - _t_q0) * 1000)
        logger.error("[search] hybrid query FAILED — falling back to dense-only\n%s",
                     traceback.format_exc())
        _search_mode = "dense_fallback"
        try:
            _t_q0b = _t.perf_counter()
            docs = _points_to_docs(_dense_query(collection_name, dense_vector))
            _qdrant_ms = int((_t.perf_counter() - _t_q0b) * 1000)
            logger.info("[search] dense fallback OK — %d docs", len(docs))
        except Exception:
            logger.error("[search] dense fallback ALSO FAILED\n%s", traceback.format_exc())
            docs = []

    if not docs:
        if _timing_cb:
            try:
                _timing_cb({"qdrant_ms": _qdrant_ms, "rerank_ms": 0, "embed_ms": embed_ms, "mode": _search_mode, "dense_vector": dense_vector, "sparse_vec": sparse_vec})
            except Exception:
                pass
        return []

    if _prefetch_cb:
        try:
            _prefetch_cb(list(docs))
        except Exception:
            pass

    _t_r0 = _t.perf_counter()
    reranked = _cohere_rerank(query, docs, top_n=RERANK_TOP_N)
    rerank_ms = int((_t.perf_counter() - _t_r0) * 1000)

    # Structural promotion: pin heading-matched chunks for navigational queries
    # (e.g. "what is the conclusion") before the reranker drops them.
    reranked = _promote_structural_matches(query, docs, reranked, top_n=RERANK_TOP_N)

    if _reranked_cb:
        try:
            _reranked_cb(list(reranked))
        except Exception:
            pass

    if _timing_cb:
        try:
            _timing_cb({"qdrant_ms": _qdrant_ms, "rerank_ms": rerank_ms, "embed_ms": embed_ms, "mode": _search_mode, "dense_vector": dense_vector, "sparse_vec": sparse_vec})
        except Exception:
            pass

    return reranked


def list_sources(session_id: str) -> list[str]:
    name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(name):
        return []
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for p in points:
            src = (p.payload or {}).get("metadata", {}).get("source")
            if src:
                seen.add(src)
        if offset is None:
            break
    return sorted(seen)


def get_document_summary(session_id: str) -> str | None:
    """Retrieve the stored document summary chunk, or None if not yet generated."""
    name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(name):
        return None
    # Ensure the payload index exists — collections created before this fix won't have it.
    _ensure_payload_index(name)
    try:
        results = qdrant_client.scroll(
            collection_name=name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="metadata.type",
                    match=qmodels.MatchValue(value="document_summary"),
                )]
            ),
            limit=1,
            with_payload=True,
        )
        points = results[0]
        if points:
            return (points[0].payload or {}).get("page_content")
    except Exception:
        logger.error("[get_document_summary] scroll failed for session=%s\n%s",
                     session_id, traceback.format_exc())
    return None
